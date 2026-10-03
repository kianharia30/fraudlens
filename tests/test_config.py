from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from fraudlens.config import CONFIG_ENV_VAR, load_config
from fraudlens.seed import set_global_seed


def test_default_config_loads_and_is_sane() -> None:
    cfg = load_config()
    assert cfg.split.test_frac == pytest.approx(0.15)
    assert cfg.costs.false_alarm_cost == 5.0
    assert cfg.costs.review_cost == 2.0
    assert cfg.costs.missed_fraud_multiplier == 1.0
    assert all(p.is_absolute() for p in cfg.paths.model_dump().values())


def test_relative_paths_resolve_against_project_root(tmp_project: Path) -> None:
    cfg = load_config(tmp_project / "configs" / "config.yaml")
    assert cfg.paths.raw_dir == (tmp_project / "data" / "raw").resolve()


def test_env_var_selects_config(tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_project / "configs" / "config.yaml"))
    assert load_config().paths.raw_dir.is_relative_to(tmp_project.resolve())


def _write_with(tmp_project: Path, section: str, key: str, value: object) -> Path:
    path = tmp_project / "configs" / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw[section][key] = value
    path.write_text(yaml.safe_dump(raw))
    return path


def test_split_must_leave_a_test_pool(tmp_project: Path) -> None:
    path = _write_with(tmp_project, "split", "valid_frac", 0.30)  # 0.70 + 0.30 = 1.0
    with pytest.raises(ValidationError, match="test pool"):
        load_config(path)


def test_unknown_keys_are_rejected(tmp_project: Path) -> None:
    path = _write_with(tmp_project, "costs", "false_alarm_cots", 5.0)  # typo
    with pytest.raises(ValidationError):
        load_config(path)


def test_negative_cost_is_rejected(tmp_project: Path) -> None:
    path = _write_with(tmp_project, "costs", "review_cost", -1.0)
    with pytest.raises(ValidationError):
        load_config(path)


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_seed_is_reproducible() -> None:
    a = set_global_seed(123).random(5)
    legacy_a = np.random.rand(3)
    b = set_global_seed(123).random(5)
    legacy_b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(legacy_a, legacy_b)
