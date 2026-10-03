"""Logistic-regression baseline: median imputation, standard scaling, class weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraudlens.config import Config
from fraudlens.logging_utils import get_logger

logger = get_logger(__name__)


def build_baseline(cfg: Config) -> Pipeline:
    return Pipeline(
        [
            # keep_empty_features keeps the column count stable if a column is all-NaN.
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=cfg.model.baseline_max_iter,
                    random_state=cfg.project.seed,
                ),
            ),
        ]
    )


def fit_baseline(X: pd.DataFrame, y: np.ndarray, cfg: Config) -> Pipeline:
    """Fit on the most recent ``baseline_max_rows`` training rows (bounded RAM on a laptop)."""
    n = cfg.model.baseline_max_rows
    if len(X) > n:
        logger.info("Baseline: using the most recent %d of %d training rows", n, len(X))
        X, y = X.iloc[-n:], y[-n:]
    # Extreme values in anonymised columns can overflow float32 when squared during scaling.
    x64 = X.astype(np.float64).clip(-1e9, 1e9)
    model = build_baseline(cfg)
    model.fit(x64, y)
    return model


def predict_baseline(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X.astype(np.float64).clip(-1e9, 1e9))[:, 1])
