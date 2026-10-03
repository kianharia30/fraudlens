"""Validate that the raw IEEE-CIS files are in place.

Usage:
    python scripts/validate_data.py [--config configs/config.yaml]

Exits with status 0 if the data is ready, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys

from fraudlens.config import load_config
from fraudlens.data.validation import setup_instructions, validate_raw_data
from fraudlens.logging_utils import configure_logging, get_logger

logger = get_logger("validate_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args(argv)

    configure_logging()
    cfg = load_config(args.config)
    report = validate_raw_data(cfg)

    if report.ok:
        logger.info("Raw data OK in %s", cfg.paths.raw_dir)
        return 0

    for problem in report.problems:
        logger.error(problem)
    logger.error("How to fix:\n%s", setup_instructions(cfg))
    return 1


if __name__ == "__main__":
    sys.exit(main())
