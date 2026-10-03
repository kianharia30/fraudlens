from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import Config
from fraudlens.data.loading import load_merged
from fraudlens.data.synthetic import make_synthetic
from scripts.update_readme import main as update_readme
from scripts.update_readme import replace_block


def test_left_join_keeps_every_transaction_and_compacts_dtypes(tmp_config: Config) -> None:
    tx, ident = make_synthetic(n_rows=500, n_cards=30, seed=1)
    tx.to_csv(tmp_config.paths.raw_dir / tmp_config.data.transaction_file, index=False)
    ident.to_csv(tmp_config.paths.raw_dir / tmp_config.data.identity_file, index=False)
    df = load_merged(tmp_config)
    assert len(df) == len(tx)
    assert df["DeviceType"].isna().sum() == len(tx) - len(ident)
    assert df["V258"].dtype == np.float32
    assert df["TransactionAmt"].dtype == np.float64  # money stays exact
    assert isinstance(df["P_emaildomain"].dtype, pd.CategoricalDtype)


def test_duplicate_identity_rows_are_rejected(tmp_config: Config) -> None:
    tx, ident = make_synthetic(n_rows=100, n_cards=10, seed=1)
    tx.to_csv(tmp_config.paths.raw_dir / tmp_config.data.transaction_file, index=False)
    pd.concat([ident, ident.iloc[:1]]).to_csv(
        tmp_config.paths.raw_dir / tmp_config.data.identity_file, index=False
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_merged(tmp_config)


def test_replace_block() -> None:
    text = "a\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nb"
    assert (
        replace_block(text, "RESULTS", "new")
        == "a\n<!-- RESULTS:START -->\nnew\n<!-- RESULTS:END -->\nb"
    )
    with pytest.raises(ValueError):
        replace_block("no markers", "RESULTS", "x")


def test_readme_refuses_synthetic_metrics(trained_config: Config, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("<!-- RESULTS:START -->\n{{PR_AUC}}\n<!-- RESULTS:END -->\n")
    cfg_path = trained_config.paths.metrics_file.parents[1] / "configs" / "config.yaml"
    assert (
        update_readme(
            [
                "--config",
                str(cfg_path),
                "--readme",
                str(readme),
                "--model-card",
                str(tmp_path / "card.md"),
            ]
        )
        == 1
    )
    assert "{{PR_AUC}}" in readme.read_text()


def test_readme_filled_from_real_metrics(trained_config: Config, tmp_path: Path) -> None:
    metrics = json.loads(trained_config.paths.metrics_file.read_text()) | {"dataset": "IEEE-CIS"}
    fake_root = tmp_path / "proj"
    (fake_root / "configs").mkdir(parents=True)
    (fake_root / "docs").mkdir()
    (fake_root / "docs" / "metrics.json").write_text(json.dumps(metrics))
    src_cfg = trained_config.paths.metrics_file.parents[1] / "configs" / "config.yaml"
    (fake_root / "configs" / "config.yaml").write_text(
        src_cfg.read_text().replace(str(trained_config.paths.metrics_file), "docs/metrics.json")
    )
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- RESULTS:START -->\n{{PR_AUC}}\n<!-- RESULTS:END -->\n")
    assert (
        update_readme(
            [
                "--config",
                str(fake_root / "configs" / "config.yaml"),
                "--readme",
                str(readme),
                "--model-card",
                str(tmp_path / "card.md"),
            ]
        )
        == 0
    )
    text = readme.read_text()
    assert "{{PR_AUC}}" not in text and f"{metrics['test']['LightGBM']['pr_auc']:.4f}" in text
