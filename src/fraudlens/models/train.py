"""Training orchestration (Phases 4-5).

Data usage, by block:
    train -> FeaturePipeline fit, logistic baseline fit, LightGBM fit
    valid -> Optuna selection + early stopping, calibration fit, threshold selection
    test  -> final reported numbers, figures, demo pool (never used for any choice)
"""

from __future__ import annotations

import gc
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudlens.config import Config
from fraudlens.data.split import load_split
from fraudlens.demo_pool import build_demo_pool, save_demo_pool
from fraudlens.features.pipeline import FeaturePipeline
from fraudlens.logging_utils import get_logger
from fraudlens.models import figures
from fraudlens.models.artifacts import (
    BASELINE_FILE,
    EVAL_SCORES_FILE,
    ModelBundle,
    git_hash,
    new_version,
    save_bundle,
)
from fraudlens.models.baseline import fit_baseline, predict_baseline
from fraudlens.models.calibration import ScoreCalibrator
from fraudlens.models.lgbm import TuningResult, feature_importance, predict_raw, tune_lgbm
from fraudlens.models.metrics import confusion, evaluate_scores
from fraudlens.models.policy import (
    Thresholds,
    decide_codes,
    do_nothing_thresholds,
    optimise_thresholds,
    summarise_policy,
)
from fraudlens.scoring import FraudScorer
from fraudlens.seed import set_global_seed

logger = get_logger(__name__)

LGBM, LOGREG = "LightGBM", "Logistic regression"


@dataclass
class Block:
    """Model matrix and the few raw columns evaluation needs, for one split block."""

    X: pd.DataFrame
    y: np.ndarray
    amount: np.ndarray
    time_range: tuple[int, int]


def _prepare_blocks(cfg: Config) -> tuple[FeaturePipeline, dict[str, Block], pd.DataFrame]:
    """Fit the pipeline on train; transform all blocks; return the raw test frame too."""
    d = cfg.data
    blocks: dict[str, Block] = {}
    pipeline: FeaturePipeline | None = None
    test_raw = pd.DataFrame()
    for name in ("train", "valid", "test"):
        df = load_split(cfg, name)
        if pipeline is None:
            pipeline = FeaturePipeline.from_config(cfg).fit(df)
        blocks[name] = Block(
            X=pipeline.transform(df),
            y=df[d.target_col].to_numpy().astype(np.int8),
            amount=df[d.amount_col].to_numpy(dtype=np.float64),
            time_range=(int(df[d.time_col].min()), int(df[d.time_col].max())),
        )
        if name == "test":
            test_raw = df
        else:
            del df
        gc.collect()
        logger.info("%s: X=%s", name, blocks[name].X.shape)
    assert pipeline is not None
    return pipeline, blocks, test_raw


def _policy_report(
    name: str,
    valid_scores: np.ndarray,
    test_scores: np.ndarray,
    blocks: dict[str, Block],
    cfg: Config,
) -> tuple[Thresholds, dict[str, Any]]:
    """Choose thresholds on valid, evaluate them on test."""
    v, t = blocks["valid"], blocks["test"]
    tc = cfg.thresholds
    thr, valid_cost = optimise_thresholds(
        valid_scores,
        v.y,
        v.amount,
        cfg.costs,
        tc.grid_size,
        max_review_rate=tc.max_review_rate,
        max_block_rate=tc.max_block_rate,
    )
    summary = summarise_policy(test_scores, t.y, t.amount, thr, cfg.costs)
    codes = decide_codes(test_scores, thr)
    logger.info(
        "%s thresholds review>=%.4f block>=%.4f | valid cost £%.0f | test cost £%.0f",
        name,
        thr.review,
        thr.block,
        valid_cost,
        summary.total_cost,
    )
    return thr, {
        "thresholds": thr.as_dict(),
        "valid_cost": valid_cost,
        "test": summary.as_dict(),
        "test_confusion_blocked": confusion(t.y, codes == 2),
        "test_confusion_flagged": confusion(t.y, codes >= 1),
    }


def _write_figures(
    cfg: Config,
    blocks: dict[str, Block],
    scores: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, Any],
    importance: pd.Series,
    thr: Thresholds,
) -> list[str]:
    paths = figures.render_all(
        cfg.paths.figures_dir,
        blocks["test"].y,
        scores["lgbm_raw"]["test"],
        scores["lgbm_cal"]["test"],
        scores["logreg"]["test"],
        metrics["test"],
        importance,
        thr.review,
        thr.block,
    )
    return [p.name for p in paths]


def policy_section(
    cfg: Config, blocks: dict[str, Block], scores: dict[str, dict[str, np.ndarray]]
) -> tuple[Thresholds, dict[str, Any]]:
    """Thresholds (chosen on valid) and test-block £ costs for both models vs do-nothing.

    ``scores`` needs ``lgbm_cal`` and ``logreg``, each with ``valid`` and ``test`` arrays.
    Shared by training and ``scripts/retune_thresholds.py``.
    """
    te = blocks["test"]
    thr, lgbm_policy = _policy_report(
        LGBM, scores["lgbm_cal"]["valid"], scores["lgbm_cal"]["test"], blocks, cfg
    )
    _, logreg_policy = _policy_report(
        LOGREG, scores["logreg"]["valid"], scores["logreg"]["test"], blocks, cfg
    )
    nothing = summarise_policy(
        scores["lgbm_cal"]["test"], te.y, te.amount, do_nothing_thresholds(), cfg.costs
    )
    return thr, {
        "currency": cfg.costs.currency,
        "config": cfg.costs.model_dump(),
        "budgets": {
            "max_review_rate": cfg.thresholds.max_review_rate,
            "max_block_rate": cfg.thresholds.max_block_rate,
        },
        "do_nothing": {"test": nothing.as_dict()},
        LGBM: lgbm_policy,
        LOGREG: logreg_policy,
        "savings_vs_do_nothing": {
            LGBM: nothing.total_cost - lgbm_policy["test"]["total_cost"],
            LOGREG: nothing.total_cost - logreg_policy["test"]["total_cost"],
        },
        "savings_vs_logreg": logreg_policy["test"]["total_cost"]
        - lgbm_policy["test"]["total_cost"],
    }


def _json_safe(obj: Any) -> Any:
    """Strict JSON: NaN/inf -> null, numpy scalars -> Python numbers."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_reports(cfg: Config, pipeline: FeaturePipeline, metrics: dict[str, Any]) -> None:
    """Metrics JSON (the README's single source of numbers) + dropped-column report."""
    cfg.paths.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.metrics_file.write_text(
        json.dumps(_json_safe(metrics), indent=2, allow_nan=False) + "\n"
    )
    report = pipeline.report()
    lines = [
        "# Data report (auto-generated by `make train`)",
        "",
        f"Model features: **{report['n_features']}**.",
        "",
        f"## Dropped: more than {report['max_missing_frac']:.0%} missing in the training block",
        "",
        "| Column | Missing (train) |",
        "|---|---|",
        *[f"| {c} | {f:.1%} |" for c, f in sorted(report["dropped_sparse"].items())],
        "",
        "## Dropped: constant in the training block",
        "",
        ", ".join(report["dropped_constant"]) or "None.",
        "",
        "## Raw string column replaced by its frequency encoding only (too many categories)",
        "",
        ", ".join(report["dropped_high_cardinality_raw"]) or "None.",
        "",
        "## Frequency-encoded (fitted on train)",
        "",
        ", ".join(report["frequency_encoded"]),
        "",
        "## Ordinal-encoded (fitted on train; unseen = -1, missing = NaN)",
        "",
        ", ".join(report["ordinal_encoded"]),
        "",
    ]
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports_dir / "DATA_REPORT.md").write_text("\n".join(lines))


def run_training(cfg: Config) -> Path:
    """Train, calibrate, choose thresholds, evaluate, and save a versioned bundle."""
    start = time.perf_counter()
    dataset_label = cfg.project.dataset
    set_global_seed(cfg.project.seed)
    pipeline, blocks, test_raw = _prepare_blocks(cfg)
    tr, va, te = blocks["train"], blocks["valid"], blocks["test"]

    logger.info("Fitting logistic-regression baseline")
    t0 = time.perf_counter()
    baseline = fit_baseline(tr.X, tr.y, cfg)
    baseline_seconds = time.perf_counter() - t0
    scores: dict[str, dict[str, np.ndarray]] = {
        "logreg": {b: predict_baseline(baseline, blocks[b].X) for b in ("valid", "test")}
    }

    logger.info("Tuning LightGBM (%d trials max)", cfg.model.optuna_trials)
    tuning: TuningResult = tune_lgbm(tr.X, tr.y, va.X, va.y, cfg)
    booster = tuning.booster
    scores["lgbm_raw"] = {b: predict_raw(booster, blocks[b].X) for b in ("valid", "test")}
    calibrator = ScoreCalibrator(cfg.model.calibration_method).fit(
        scores["lgbm_raw"]["valid"], va.y
    )
    scores["lgbm_cal"] = {b: calibrator.transform(scores["lgbm_raw"][b]) for b in ("valid", "test")}

    p_target = cfg.model.recall_at_precision
    metrics: dict[str, Any] = {"dataset": dataset_label}
    for b in ("valid", "test"):
        y = blocks[b].y
        metrics[b] = {
            # Ranking metrics use raw scores (isotonic ties would blur the ranking slightly);
            # probability metrics use calibrated scores.
            LGBM: evaluate_scores(y, scores["lgbm_raw"][b], scores["lgbm_cal"][b], p_target),
            LOGREG: evaluate_scores(y, scores["logreg"][b], scores["logreg"][b], p_target),
        }
    metrics["test"][LGBM]["uncalibrated_brier"] = evaluate_scores(
        te.y, scores["lgbm_raw"]["test"], scores["lgbm_raw"]["test"], p_target
    )["brier"]

    thr, metrics["costs"] = policy_section(cfg, blocks, scores)
    lgbm_policy = metrics["costs"][LGBM]

    importance = feature_importance(booster)
    metrics["figures"] = _write_figures(cfg, blocks, scores, metrics, importance, thr)
    metrics["training"] = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "seconds_total": round(time.perf_counter() - start, 1),
        "seconds_lgbm_tuning": round(tuning.seconds, 1),
        "seconds_baseline": round(baseline_seconds, 1),
        "optuna_trials": tuning.n_trials,
        "best_iteration": tuning.best_iteration,
        "valid_pr_auc_best_trial": tuning.valid_pr_auc,
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "n_features": len(pipeline.feature_names),
        "data_window": {
            b: {
                "rows": len(blocks[b].y),
                "fraud_rate": float(blocks[b].y.mean()),
                "TransactionDT": list(blocks[b].time_range),
            }
            for b in blocks
        },
    }

    version = new_version()
    metadata = {
        "version": version,
        "dataset": dataset_label,
        "params": tuning.params,
        "thresholds": thr.as_dict(),
        "metrics_test": metrics["test"][LGBM],
        "costs_test": lgbm_policy["test"],
        "training": metrics["training"],
        "feature_names": pipeline.feature_names,
        "top_features": importance.head(25).round(1).to_dict(),
        "optuna_trials": tuning.trials,
        "config": cfg.model_dump(mode="json"),
    }
    bundle = ModelBundle(
        pipeline=pipeline,
        model_string=booster.model_to_string(num_iteration=tuning.best_iteration),
        calibrator=calibrator,
        thresholds=thr,
        metadata=metadata,
    )
    out = save_bundle(bundle, cfg, extras={BASELINE_FILE: baseline})

    pd.DataFrame(
        {
            "y": te.y,
            "amount": te.amount,
            "lgbm_raw": scores["lgbm_raw"]["test"],
            "lgbm": scores["lgbm_cal"]["test"],
            "logreg": scores["logreg"]["test"],
        }
    ).to_parquet(out / EVAL_SCORES_FILE, index=False)

    del blocks
    gc.collect()
    scorer = FraudScorer(bundle, top_n=cfg.service.top_n_reasons)
    pool = build_demo_pool(test_raw, scorer, cfg)
    save_demo_pool(pool, out)
    metrics["model_version"] = version
    write_reports(cfg, pipeline, metrics)
    logger.info(
        "Done in %.0fs. Test PR-AUC %.4f (LR %.4f). Bundle: %s",
        time.perf_counter() - start,
        metrics["test"][LGBM]["pr_auc"],
        metrics["test"][LOGREG]["pr_auc"],
        out,
    )
    return out
