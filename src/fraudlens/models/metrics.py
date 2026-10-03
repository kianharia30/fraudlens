"""Evaluation metrics for imbalanced fraud detection.

Accuracy is deliberately absent: with ~3.5% fraud, approving everything is ~96.5% "accurate".
PR-AUC (average precision) is the headline because it focuses on the positive class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


def recall_at_precision(y: np.ndarray, scores: np.ndarray, target: float) -> tuple[float, float]:
    """Highest recall achievable with precision >= target, and the score threshold for it.

    Returns ``(0.0, nan)`` if no threshold reaches the target precision.
    """
    precision, recall, thresholds = precision_recall_curve(y, scores)
    # precision/recall have one more element than thresholds (the final point has none).
    ok = precision[:-1] >= target
    if not ok.any():
        return 0.0, float("nan")
    best = int(np.argmax(np.where(ok, recall[:-1], -1)))
    return float(recall[best]), float(thresholds[best])


def ranking_metrics(y: np.ndarray, scores: np.ndarray, precision_target: float) -> dict[str, float]:
    """Threshold-free metrics; depend only on how scores rank transactions."""
    rec, thr = recall_at_precision(y, scores, precision_target)
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        f"recall_at_{int(precision_target * 100)}_precision": rec,
        f"threshold_at_{int(precision_target * 100)}_precision": thr,
    }


def probability_metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Calibration-sensitive metrics; meaningful only for calibrated probabilities."""
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_predicted": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
    }


def confusion(y: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    """Confusion matrix counts for a binary flag (e.g. 'blocked' or 'reviewed or blocked')."""
    y = np.asarray(y).astype(bool)
    f = np.asarray(flagged).astype(bool)
    return {
        "tp": int((y & f).sum()),
        "fp": int((~y & f).sum()),
        "fn": int((y & ~f).sum()),
        "tn": int((~y & ~f).sum()),
    }


def evaluate_scores(
    y: np.ndarray, ranking_scores: np.ndarray, probs: np.ndarray, precision_target: float
) -> dict[str, Any]:
    """All metrics for one model on one block."""
    return {
        "n": len(y),
        "n_fraud": int(np.sum(y)),
        **ranking_metrics(y, ranking_scores, precision_target),
        **probability_metrics(y, probs),
    }


def reliability_table(
    y: np.ndarray, probs: np.ndarray, n_bins: int = 10, min_count: int = 20
) -> pd.DataFrame:
    """Fixed-width bins of predicted probability vs observed fraud rate.

    Fixed-width (not quantile) bins: with ~3.5% fraud, quantile bins all crowd near zero
    and hide the high-score region, which is where decisions are made. Sparse bins
    (< ``min_count`` rows) are dropped as too noisy to read.
    """
    df = pd.DataFrame({"p": probs, "y": y})
    df["bin"] = np.minimum((df["p"] * n_bins).astype(int), n_bins - 1)
    t = df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return t[t["n"] >= min_count]
