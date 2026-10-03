"""Export the held-out test rows the demo app samples from (with precomputed scores)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.config import Config
from fraudlens.models.artifacts import DEMO_POOL_FILE
from fraudlens.models.policy import decide_codes
from fraudlens.scoring import FraudScorer

# Shown on the app's raw-input card even if the model dropped them as features.
DISPLAY_COLS: tuple[str, ...] = (
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "addr1",
    "addr2",
    "dist1",
)
DECISION_NAMES = np.array(["APPROVE", "REVIEW", "BLOCK"])


def build_demo_pool(test: pd.DataFrame, scorer: FraudScorer, cfg: Config) -> pd.DataFrame:
    """All test-block frauds plus a seeded sample of genuine rows, scored."""
    y_col = cfg.data.target_col
    fraud = test[test[y_col] == 1]
    genuine = test[test[y_col] == 0]
    n = min(len(genuine), cfg.app.demo_pool_max_genuine)
    genuine = genuine.sample(n=n, random_state=cfg.project.seed)
    pool = pd.concat([fraud, genuine]).sort_values(cfg.data.time_col)

    cols = list(dict.fromkeys([*DISPLAY_COLS, *scorer.pipeline.input_columns, y_col]))
    pool = pool[[c for c in cols if c in pool.columns]].reset_index(drop=True)
    raw, calibrated = scorer.score_frame(pool)
    pool["raw_score"] = raw
    pool["fraud_score"] = calibrated
    pool["decision"] = DECISION_NAMES[decide_codes(calibrated, scorer.thresholds)]
    return pool


def save_demo_pool(pool: pd.DataFrame, bundle_dir: Path) -> Path:
    path = bundle_dir / DEMO_POOL_FILE
    pool.to_parquet(path, index=False)
    return path
