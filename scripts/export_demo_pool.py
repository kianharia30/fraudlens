"""Re-export the demo pool (held-out test rows + scores) for the latest model.

Training already does this; use it after changing app.demo_pool_max_genuine.

Usage:
    python scripts/export_demo_pool.py [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import sys

from fraudlens.config import load_config
from fraudlens.data.split import load_split
from fraudlens.demo_pool import build_demo_pool, save_demo_pool
from fraudlens.logging_utils import configure_logging, get_logger
from fraudlens.models.artifacts import latest_bundle_dir
from fraudlens.scoring import FraudScorer

logger = get_logger("export_demo_pool")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    bundle_dir = latest_bundle_dir(cfg)
    pool = build_demo_pool(load_split(cfg, "test"), FraudScorer.load(cfg, bundle_dir), cfg)
    path = save_demo_pool(pool, bundle_dir)
    logger.info(
        "Wrote %d rows (%d fraud) to %s", len(pool), int(pool[cfg.data.target_col].sum()), path
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
