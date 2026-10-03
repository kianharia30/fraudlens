"""Re-evaluate the latest saved model on the held-out test block (no retraining).

Re-scores data/processed/test.parquet with the saved bundle and checks the result
matches the metrics recorded at training time (a reproducibility check).

Usage:
    python scripts/evaluate.py [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from fraudlens.config import load_config
from fraudlens.data.split import load_split
from fraudlens.logging_utils import configure_logging, get_logger
from fraudlens.models.artifacts import latest_bundle_dir
from fraudlens.models.metrics import evaluate_scores
from fraudlens.models.policy import summarise_policy
from fraudlens.scoring import FraudScorer

logger = get_logger("evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)

    bundle_dir = latest_bundle_dir(cfg)
    scorer = FraudScorer.load(cfg, bundle_dir)
    test = load_split(cfg, "test")
    raw, calibrated = scorer.score_frame(test)
    y = test[cfg.data.target_col].to_numpy()
    amount = test[cfg.data.amount_col].to_numpy(dtype=np.float64)

    metrics = evaluate_scores(y, raw, calibrated, cfg.model.recall_at_precision)
    policy = summarise_policy(calibrated, y, amount, scorer.thresholds, cfg.costs)
    logger.info("Model %s on test block:\n%s", scorer.version, json.dumps(metrics, indent=2))
    logger.info("Policy cost on test block: £%.2f", policy.total_cost)

    recorded = scorer.bundle.metadata.get("metrics_test", {}).get("pr_auc")
    if recorded is not None and not np.isclose(recorded, metrics["pr_auc"], atol=1e-6):
        logger.error("PR-AUC %.6f differs from recorded %.6f", metrics["pr_auc"], recorded)
        return 1
    logger.info("Matches the metrics recorded at training time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
