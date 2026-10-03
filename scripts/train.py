"""Train baseline + LightGBM, calibrate, choose cost thresholds, evaluate, save artefacts.

Usage:
    python scripts/train.py [--config configs/config.yaml]

Requires processed splits (`make data`). Writes a versioned bundle to artifacts/models/,
figures to docs/figures/, and metrics to docs/metrics.json.
"""

from __future__ import annotations

import argparse
import sys

from fraudlens.config import load_config
from fraudlens.logging_utils import configure_logging
from fraudlens.models.train import run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    configure_logging()
    run_training(load_config(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
