"""Typed configuration loading.

All tunable values (paths, seeds, split fractions, costs) live in ``configs/config.yaml``.
This module parses that file into validated, immutable pydantic models so that a typo or
an invalid value fails loudly at start-up rather than silently mid-pipeline.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_ENV_VAR = "FRAUDLENS_CONFIG"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # <root>/src/fraudlens -> <root>
DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "configs" / "config.yaml"


class _Frozen(BaseModel):
    """Base model: immutable and rejects unknown keys (catches typos in YAML)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProjectConfig(_Frozen):
    name: str
    seed: int = Field(ge=0)
    dataset: str  # recorded in metrics.json; "synthetic" marks numbers as meaningless


class PathsConfig(_Frozen):
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    figures_dir: Path
    reports_dir: Path
    metrics_file: Path
    benchmark_file: Path

    def resolved(self, root: Path) -> PathsConfig:
        """Return a copy with every relative path made absolute against ``root``."""
        return PathsConfig(
            **{
                name: (value if value.is_absolute() else (root / value).resolve())
                for name, value in self.model_dump().items()
            }
        )


class DataConfig(_Frozen):
    transaction_file: str
    identity_file: str
    id_col: str
    time_col: str
    target_col: str
    amount_col: str


class SplitConfig(_Frozen):
    train_frac: float = Field(gt=0, lt=1)
    valid_frac: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _leave_room_for_test(self) -> SplitConfig:
        if self.train_frac + self.valid_frac >= 1.0:
            raise ValueError("train_frac + valid_frac must be < 1 so a test pool remains")
        return self

    @property
    def test_frac(self) -> float:
        return 1.0 - self.train_frac - self.valid_frac


class FeaturesConfig(_Frozen):
    card_uid_cols: list[str] = Field(min_length=1)
    card_key_cols: list[str] = Field(min_length=1)
    novelty_cols: list[str]
    velocity_windows_seconds: dict[str, int]
    max_missing_frac: float = Field(gt=0, le=1)
    freq_encode_cols: list[str]
    max_ordinal_cardinality: int = Field(ge=1)


class ModelConfig(_Frozen):
    optuna_trials: int = Field(ge=1)
    optuna_timeout_seconds: int = Field(ge=1)
    learning_rate: float = Field(gt=0, le=1)
    max_boost_rounds: int = Field(ge=1)
    early_stopping_rounds: int = Field(ge=1)
    num_threads: int = Field(ge=0)
    calibration_method: Literal["isotonic", "sigmoid"]
    recall_at_precision: float = Field(gt=0, lt=1)
    baseline_max_rows: int = Field(ge=100)
    baseline_max_iter: int = Field(ge=1)


class CostsConfig(_Frozen):
    currency: str
    missed_fraud_multiplier: float = Field(ge=0)
    false_alarm_cost: float = Field(ge=0)
    review_cost: float = Field(ge=0)
    review_catch_rate: float = Field(ge=0, le=1)


class ThresholdsConfig(_Frozen):
    grid_size: int = Field(ge=10)
    max_review_rate: float = Field(gt=0, le=1)
    max_block_rate: float = Field(gt=0, le=1)


class ServiceConfig(_Frozen):
    host: str
    port: int = Field(ge=1, le=65535)
    top_n_reasons: int = Field(ge=1)


class AppConfig(_Frozen):
    demo_fraud_weight: float = Field(ge=0, le=1)
    animation_step_seconds: float = Field(ge=0)
    demo_pool_max_genuine: int = Field(ge=1)


class Config(_Frozen):
    """Root configuration object."""

    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    features: FeaturesConfig
    model: ModelConfig
    costs: CostsConfig
    thresholds: ThresholdsConfig
    service: ServiceConfig
    app: AppConfig


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a config file.

    Resolution order: explicit ``path`` > ``$FRAUDLENS_CONFIG`` > ``configs/config.yaml``.
    Relative paths inside the file are resolved against the project root, defined as
    the parent of the directory containing the config file.

    Raises:
        FileNotFoundError: if the config file does not exist.
        pydantic.ValidationError: if the file contents are invalid.
    """
    config_path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = Config.model_validate(raw)
    root = config_path.parent.parent
    return cfg.model_copy(update={"paths": cfg.paths.resolved(root)})


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached default config for long-running processes (API, app)."""
    return load_config()
