"""Re-choose the decision thresholds for the latest model (no retraining).

Use after changing ``costs`` or ``thresholds`` budgets in the config. Thresholds are
re-optimised on the **validation** block only; the test block is used only to report.
Updates the bundle, docs/metrics.json, figures and the demo pool (`make thresholds`
also refreshes the README and model card).

Usage:
    python scripts/retune_thresholds.py [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from fraudlens.config import load_config
from fraudlens.data.split import load_split
from fraudlens.demo_pool import build_demo_pool, save_demo_pool
from fraudlens.logging_utils import configure_logging, get_logger
from fraudlens.models import figures
from fraudlens.models.artifacts import (
    BASELINE_FILE,
    EVAL_SCORES_FILE,
    latest_bundle_dir,
    load_bundle,
    overwrite_bundle,
)
from fraudlens.models.baseline import predict_baseline
from fraudlens.models.lgbm import feature_importance
from fraudlens.models.train import LGBM, Block, policy_section, write_reports
from fraudlens.scoring import FraudScorer

logger = get_logger("retune_thresholds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    bundle_dir = latest_bundle_dir(cfg)
    bundle = load_bundle(bundle_dir)
    baseline = joblib.load(bundle_dir / BASELINE_FILE)
    scorer = FraudScorer(bundle, top_n=cfg.service.top_n_reasons)

    blocks: dict[str, Block] = {}
    scores: dict[str, dict[str, np.ndarray]] = {"lgbm_cal": {}, "logreg": {}}
    test_raw = pd.DataFrame()
    for name in ("valid", "test"):
        df = load_split(cfg, name)
        _, scores["lgbm_cal"][name] = scorer.score_frame(df)
        scores["logreg"][name] = predict_baseline(baseline, bundle.pipeline.transform(df))
        blocks[name] = Block(
            X=pd.DataFrame(),
            y=df[cfg.data.target_col].to_numpy().astype(np.int8),
            amount=df[cfg.data.amount_col].to_numpy(dtype=np.float64),
            time_range=(int(df[cfg.data.time_col].min()), int(df[cfg.data.time_col].max())),
        )
        if name == "test":
            test_raw = df

    thr, costs = policy_section(cfg, blocks, scores)
    bundle.thresholds = thr
    bundle.metadata["thresholds"] = thr.as_dict()
    bundle.metadata["costs_test"] = costs[LGBM]["test"]
    bundle.metadata["config"] = cfg.model_dump(mode="json")
    bundle.metadata["thresholds_retuned_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    overwrite_bundle(bundle, bundle_dir)

    metrics = json.loads(cfg.paths.metrics_file.read_text())
    metrics["costs"] = costs
    write_reports(cfg, bundle.pipeline, metrics)

    ev = pd.read_parquet(bundle_dir / EVAL_SCORES_FILE)
    figures.render_all(
        cfg.paths.figures_dir,
        ev["y"].to_numpy(),
        ev["lgbm_raw"].to_numpy(),
        ev["lgbm"].to_numpy(),
        ev["logreg"].to_numpy(),
        metrics["test"],
        feature_importance(bundle.booster()),
        thr.review,
        thr.block,
    )
    save_demo_pool(build_demo_pool(test_raw, FraudScorer(bundle), cfg), bundle_dir)
    logger.info("Thresholds now review>=%.4f block>=%.4f", thr.review, thr.block)
    logger.info("Run `make readme` (or `make thresholds`, which does both) to refresh the docs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
