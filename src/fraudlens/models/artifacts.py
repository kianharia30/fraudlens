"""Versioned model bundles.

Layout::

    artifacts/models/
        LATEST                      # text file: name of the newest bundle directory
        20261002-211500-1a2b3c4/
            bundle.joblib           # FeaturePipeline + LightGBM model + calibrator + thresholds
            baseline.joblib         # logistic-regression baseline (evaluation only)
            metadata.json           # human-readable: data window, metrics, params, git hash
            demo_pool.parquet       # held-out test rows the demo app samples from
            eval_scores.parquet     # test-block scores/labels/amounts for the performance tab
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb

from fraudlens.config import Config
from fraudlens.features.pipeline import FeaturePipeline
from fraudlens.models.calibration import ScoreCalibrator
from fraudlens.models.policy import Thresholds

LATEST_FILE = "LATEST"
BUNDLE_FILE = "bundle.joblib"
BASELINE_FILE = "baseline.joblib"
METADATA_FILE = "metadata.json"
DEMO_POOL_FILE = "demo_pool.parquet"
EVAL_SCORES_FILE = "eval_scores.parquet"


@dataclass
class ModelBundle:
    """Everything needed to score a transaction, saved and loaded as one unit."""

    pipeline: FeaturePipeline
    model_string: str  # LightGBM text model, truncated at the best iteration
    calibrator: ScoreCalibrator
    thresholds: Thresholds
    metadata: dict[str, Any]

    def booster(self) -> lgb.Booster:
        return lgb.Booster(model_str=self.model_string)

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unknown"))


def git_hash() -> str:
    """Short git commit of the working tree, with ``-dirty`` if there are local changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def new_version() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{git_hash().split('-')[0]}"


def save_bundle(bundle: ModelBundle, cfg: Config, extras: dict[str, Any] | None = None) -> Path:
    """Write a bundle (+ optional extra joblib objects) and point LATEST at it."""
    root = cfg.paths.artifacts_dir
    out = root / bundle.version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / BUNDLE_FILE, compress=3)
    for name, obj in (extras or {}).items():
        joblib.dump(obj, out / name, compress=3)
    (out / METADATA_FILE).write_text(json.dumps(bundle.metadata, indent=2, default=str) + "\n")
    (root / LATEST_FILE).write_text(bundle.version)
    return out


def overwrite_bundle(bundle: ModelBundle, bundle_dir: Path) -> None:
    """Rewrite an existing bundle in place (e.g. after re-tuning thresholds)."""
    joblib.dump(bundle, bundle_dir / BUNDLE_FILE, compress=3)
    (bundle_dir / METADATA_FILE).write_text(
        json.dumps(bundle.metadata, indent=2, default=str) + "\n"
    )


def latest_bundle_dir(cfg: Config) -> Path:
    root = cfg.paths.artifacts_dir
    pointer = root / LATEST_FILE
    if not pointer.is_file():
        raise FileNotFoundError(f"No trained model found in {root}. Run `make train` first.")
    return root / pointer.read_text().strip()


def load_bundle(path: Path) -> ModelBundle:
    bundle = joblib.load(path / BUNDLE_FILE)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path / BUNDLE_FILE} does not contain a ModelBundle")
    return bundle
