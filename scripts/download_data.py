"""Download the IEEE-CIS training files with the Kaggle CLI, then validate them.

Prerequisites (one-off, done by you in a browser):
  1. Sign in at kaggle.com and accept the competition rules:
     https://www.kaggle.com/competitions/ieee-fraud-detection/rules
  2. Create an API token (kaggle.com > Settings > API > Create New Token) and save it
     as ~/.kaggle/kaggle.json (chmod 600), or set KAGGLE_USERNAME / KAGGLE_KEY.
  3. pip install kaggle   (not a project dependency)

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile

from fraudlens.config import load_config
from fraudlens.data.validation import setup_instructions, validate_raw_data
from fraudlens.logging_utils import configure_logging, get_logger

logger = get_logger("download_data")
COMPETITION = "ieee-fraud-detection"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    raw = cfg.paths.raw_dir

    if validate_raw_data(cfg).ok:
        logger.info("Data already present in %s", raw)
        return 0
    if shutil.which("kaggle") is None:
        logger.error("Kaggle CLI not found (pip install kaggle).\n%s", setup_instructions(cfg))
        return 1

    raw.mkdir(parents=True, exist_ok=True)
    for name in (cfg.data.transaction_file, cfg.data.identity_file):
        logger.info("Downloading %s", name)
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-f", name, "-p", str(raw)],
            check=False,
        )
        if result.returncode != 0:
            logger.error("Kaggle download failed. Have you accepted the competition rules?")
            return 1
        archive = raw / f"{name}.zip"
        if archive.is_file():
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(raw)
            archive.unlink()

    report = validate_raw_data(cfg)
    for problem in report.problems:
        logger.error(problem)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
