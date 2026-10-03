"""Shared pytest fixtures. Tests never need the real IEEE-CIS dataset."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraudlens.config import DEFAULT_CONFIG_PATH, Config, load_config
from fraudlens.data.build import build_dataset
from fraudlens.data.synthetic import make_synthetic
from fraudlens.models.train import run_training
from fraudlens.scoring import FraudScorer


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """A throwaway project root containing a copy of the default config."""
    (tmp_path / "configs").mkdir()
    shutil.copy(DEFAULT_CONFIG_PATH, tmp_path / "configs" / "config.yaml")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def tmp_config(tmp_project: Path) -> Config:
    """Config whose relative paths resolve inside ``tmp_project``."""
    return load_config(tmp_project / "configs" / "config.yaml")


def _small_config(root: Path) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["project"]["dataset"] = "synthetic"
    raw["model"].update(optuna_trials=2, optuna_timeout_seconds=60, max_boost_rounds=200)
    raw["app"]["demo_pool_max_genuine"] = 500
    raw["thresholds"]["grid_size"] = 50
    (root / "configs").mkdir(parents=True)
    path = root / "configs" / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture(scope="session")
def trained_config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """End-to-end run on synthetic data: CSVs -> features -> splits -> trained bundle."""
    root = tmp_path_factory.mktemp("project")
    cfg = load_config(_small_config(root))
    cfg.paths.raw_dir.mkdir(parents=True)
    tx, ident = make_synthetic(n_rows=6000, n_cards=400, seed=0)
    tx.to_csv(cfg.paths.raw_dir / cfg.data.transaction_file, index=False)
    ident.to_csv(cfg.paths.raw_dir / cfg.data.identity_file, index=False)
    build_dataset(cfg)
    run_training(cfg)
    return cfg


@pytest.fixture(scope="session")
def scorer(trained_config: Config) -> FraudScorer:
    return FraudScorer.load(trained_config)


@pytest.fixture(scope="session")
def test_block(trained_config: Config) -> pd.DataFrame:
    return pd.read_parquet(trained_config.paths.processed_dir / "test.parquet")
