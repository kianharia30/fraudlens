"""Write synthetic IEEE-CIS look-alike CSVs for a no-data smoke run.

Writes to the raw_dir of the given config (default: configs/config.synthetic.yaml, which
points at data/synthetic/raw), so the real data/raw is never touched.

Usage:
    python scripts/make_synthetic_data.py [--rows 20000]
"""

from __future__ import annotations

import argparse
import sys

from fraudlens.config import DEFAULT_CONFIG_PATH, load_config
from fraudlens.data.synthetic import make_synthetic
from fraudlens.logging_utils import configure_logging, get_logger

logger = get_logger("make_synthetic_data")
SYNTHETIC_CONFIG = DEFAULT_CONFIG_PATH.with_name("config.synthetic.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SYNTHETIC_CONFIG))
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--cards", type=int, default=1_500)
    args = parser.parse_args(argv)

    configure_logging()
    cfg = load_config(args.config)
    if cfg.paths.raw_dir == load_config(DEFAULT_CONFIG_PATH).paths.raw_dir:
        logger.error("Refusing to write synthetic data into the real raw_dir %s", cfg.paths.raw_dir)
        return 1
    tx, ident = make_synthetic(args.rows, args.cards, seed=cfg.project.seed)
    cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    tx.to_csv(cfg.paths.raw_dir / cfg.data.transaction_file, index=False)
    ident.to_csv(cfg.paths.raw_dir / cfg.data.identity_file, index=False)
    logger.info("Wrote %d synthetic transactions to %s", len(tx), cfg.paths.raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
