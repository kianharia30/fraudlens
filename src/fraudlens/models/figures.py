"""Evaluation figures written to docs/figures (all computed on the held-out test block)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.metrics import precision_recall_curve, roc_curve

from fraudlens import style
from fraudlens.models.metrics import reliability_table

_RC = {
    "figure.facecolor": style.SURFACE,
    "axes.facecolor": style.SURFACE,
    "axes.edgecolor": style.AXIS,
    "axes.labelcolor": style.TEXT_SECONDARY,
    "axes.titlecolor": style.TEXT_PRIMARY,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": style.GRID,
    "grid.linewidth": 0.6,
    "xtick.color": style.TEXT_SECONDARY,
    "ytick.color": style.TEXT_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": style.TEXT_PRIMARY,
    "font.size": 10,
    "lines.linewidth": 2,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _new(size: tuple[float, float] = (6.0, 4.2)) -> tuple[plt.Figure, Axes]:
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=size)
    return fig, ax


def pr_curve(
    y: np.ndarray, scores: Mapping[str, np.ndarray], ap: Mapping[str, float], out: Path
) -> Path:
    with plt.rc_context(_RC):
        fig, ax = _new()
        for name, s in scores.items():
            p, r, _ = precision_recall_curve(y, s)
            ax.plot(r, p, color=style.MODEL_COLORS[name], label=f"{name} (PR-AUC {ap[name]:.3f})")
        ax.axhline(
            float(np.mean(y)), color=style.TEXT_MUTED, ls="--", lw=1, label="No-skill (fraud rate)"
        )
        ax.set(
            xlabel="Recall (share of fraud caught)",
            ylabel="Precision",
            title="Precision-recall, test block",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(
            loc="upper right"
        )  # high precision at high recall is unreachable, so this corner is empty
        return _save(fig, out)


def roc(
    y: np.ndarray, scores: Mapping[str, np.ndarray], auc: Mapping[str, float], out: Path
) -> Path:
    with plt.rc_context(_RC):
        fig, ax = _new()
        for name, s in scores.items():
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(
                fpr, tpr, color=style.MODEL_COLORS[name], label=f"{name} (ROC-AUC {auc[name]:.3f})"
            )
        ax.plot([0, 1], [0, 1], color=style.TEXT_MUTED, ls="--", lw=1, label="Random")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC, test block")
        ax.legend(loc="lower right")
        return _save(fig, out)


def calibration(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, out: Path) -> Path:
    with plt.rc_context(_RC):
        fig, ax = _new((5.2, 5.0))
        for label, probs, color in (
            ("Raw LightGBM (class-weighted)", raw, style.SERIES_2),
            ("Calibrated (isotonic, fitted on validation)", calibrated, style.SERIES_1),
        ):
            t = reliability_table(y, probs)
            ax.plot(
                t["predicted"],
                t["observed"],
                marker="o",
                ms=6,
                color=color,
                label=label,
                markeredgecolor=style.SURFACE,
                markeredgewidth=1.5,
            )
        ax.plot([0, 1], [0, 1], color=style.TEXT_MUTED, ls="--", lw=1, label="Perfect calibration")
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean predicted probability (10 bins, n ≥ 20)",
            ylabel="Observed fraud rate",
            title="Calibration, test block",
        )
        ax.legend(loc="upper left")
        return _save(fig, out)


def feature_importance(importance: pd.Series, out: Path, top: int = 20) -> Path:
    imp = importance.head(top)[::-1] / importance.sum()
    with plt.rc_context(_RC):
        fig, ax = _new((6.4, 0.28 * len(imp) + 1.2))
        ax.barh(imp.index, imp.to_numpy(), color=style.SERIES_1, height=0.7)
        ax.grid(axis="y", visible=False)
        ax.set(xlabel="Share of total gain", title=f"Top {len(imp)} features (LightGBM gain)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        return _save(fig, out)


def score_distribution(
    y: np.ndarray, probs: np.ndarray, review: float, block: float, out: Path
) -> Path:
    """Small multiples (genuine above fraud): class sizes differ ~30x, so overlaid
    histograms would hide the fraud class and blend colours."""
    bins = np.linspace(0, 1, 41)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)
        for ax, (label, mask) in zip(axes, (("Genuine", y == 0), ("Fraud", y == 1)), strict=True):
            ax.hist(
                probs[mask],
                bins=bins,
                color=style.CLASS_COLORS[label],
                edgecolor=style.SURFACE,
                linewidth=1,
            )
            ax.set_yscale("log")
            low, high = ax.get_ylim()
            ax.set_ylim(low, high * 40)  # headroom for threshold labels
            ax.set_title(f"{label} transactions (n={int(mask.sum()):,})", loc="left", fontsize=10)
            ax.set_ylabel("Count (log)")
            for t in (review, block):
                if t <= 1:
                    ax.axvline(t, color=style.TEXT_PRIMARY, lw=1, ls=":")
        for name, t in (("REVIEW ≥", review), ("BLOCK ≥", block)):
            if t <= 1:
                axes[0].annotate(
                    f"{name} {t:.2f}",
                    (t, 0.96),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="top",
                    color=style.TEXT_SECONDARY,
                    fontsize=9,
                )
        axes[1].set_xlabel("Calibrated fraud score (test block)")
        axes[1].set_xlim(0, 1)
        fig.suptitle(
            "Score distribution by true class",
            x=0.02,
            ha="left",
            fontweight="bold",
            color=style.TEXT_PRIMARY,
        )
        fig.tight_layout()
        return _save(fig, out)


def render_all(
    out: Path,
    y: np.ndarray,
    lgbm_raw: np.ndarray,
    lgbm_calibrated: np.ndarray,
    logreg: np.ndarray,
    metrics_test: Mapping[str, Mapping[str, float]],
    importance: pd.Series,
    review: float,
    block: float,
) -> list[Path]:
    """Every README figure, from test-block scores. Used by training and `make figures`."""
    ranking = {"LightGBM": lgbm_raw, "Logistic regression": logreg}
    ap = {m: float(metrics_test[m]["pr_auc"]) for m in ranking}
    auc = {m: float(metrics_test[m]["roc_auc"]) for m in ranking}
    return [
        pr_curve(y, ranking, ap, out / "pr_curve.png"),
        roc(y, ranking, auc, out / "roc_curve.png"),
        calibration(y, lgbm_raw, lgbm_calibrated, out / "calibration.png"),
        feature_importance(importance, out / "feature_importance.png"),
        score_distribution(y, lgbm_calibrated, review, block, out / "score_distribution.png"),
    ]
