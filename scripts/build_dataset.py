"""Build processed, time-split parquet files from data/raw.

Usage:
    python scripts/build_dataset.py [--config configs/config.yaml] [--nrows N]
"""

from __future__ import annotations

import argparse
import sys

from fraudlens.config import load_config
from fraudlens.data.build import build_dataset
from fraudlens.data.validation import setup_instructions, validate_raw_data
from fraudlens.logging_utils import configure_logging, get_logger

logger = get_logger("build_dataset")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--nrows", type=int, default=None, help="Debug: only read N transactions")
    args = parser.parse_args(argv)

    configure_logging()
    cfg = load_config(args.config)
    report = validate_raw_data(cfg)
    if not report.ok:
        for problem in report.problems:
            logger.error(problem)
        logger.error("How to fix:\n%s", setup_instructions(cfg))
        return 1
    build_dataset(cfg, nrows=args.nrows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
