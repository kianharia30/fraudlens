"""Re-render docs/figures from the latest saved model (no retraining).

Uses the bundle's stored test-block scores and model, plus docs/metrics.json.

Usage:
    python scripts/render_figures.py [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from fraudlens.config import load_config
from fraudlens.logging_utils import configure_logging, get_logger
from fraudlens.models import figures
from fraudlens.models.artifacts import EVAL_SCORES_FILE, latest_bundle_dir, load_bundle
from fraudlens.models.lgbm import feature_importance

logger = get_logger("render_figures")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    bundle_dir = latest_bundle_dir(cfg)
    bundle = load_bundle(bundle_dir)
    ev = pd.read_parquet(bundle_dir / EVAL_SCORES_FILE)
    metrics = json.loads(cfg.paths.metrics_file.read_text())
    paths = figures.render_all(
        cfg.paths.figures_dir,
        ev["y"].to_numpy(),
        ev["lgbm_raw"].to_numpy(),
        ev["lgbm"].to_numpy(),
        ev["logreg"].to_numpy(),
        metrics["test"],
        feature_importance(bundle.booster()),
        bundle.thresholds.review,
        bundle.thresholds.block,
    )
    logger.info("Rendered %d figures to %s", len(paths), cfg.paths.figures_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
