"""Checks that the raw IEEE-CIS files are present and look correct before any work starts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fraudlens.config import Config

KAGGLE_URL = "https://www.kaggle.com/competitions/ieee-fraud-detection/data"

# A small subset of columns that must be present; enough to catch a wrong/corrupt file.
REQUIRED_TRANSACTION_COLS: frozenset[str] = frozenset(
    {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", "card1"}
)
REQUIRED_IDENTITY_COLS: frozenset[str] = frozenset({"TransactionID", "DeviceType", "DeviceInfo"})


@dataclass
class ValidationReport:
    """Outcome of raw-data validation. ``ok`` is True only if there are no problems."""

    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _check_file(path: Path, required_cols: frozenset[str]) -> list[str]:
    if not path.is_file():
        return [f"Missing file: {path}"]
    if path.stat().st_size == 0:
        return [f"File is empty: {path}"]
    try:
        header = pd.read_csv(path, nrows=0).columns
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        return [f"Could not parse CSV header of {path}: {exc}"]
    missing = sorted(required_cols - set(header))
    if missing:
        return [f"{path.name} is missing expected columns: {', '.join(missing)}"]
    return []


def validate_raw_data(cfg: Config) -> ValidationReport:
    """Validate that both labelled IEEE-CIS training files exist and have the expected schema."""
    raw = cfg.paths.raw_dir
    report = ValidationReport()
    report.problems += _check_file(raw / cfg.data.transaction_file, REQUIRED_TRANSACTION_COLS)
    report.problems += _check_file(raw / cfg.data.identity_file, REQUIRED_IDENTITY_COLS)
    return report


def setup_instructions(cfg: Config) -> str:
    """Human-readable instructions for obtaining the dataset."""
    return (
        "The IEEE-CIS dataset is not bundled with this repo (licence + size).\n"
        f"  1. Accept the competition rules at {KAGGLE_URL}\n"
        "  2. Download it, either manually or with the Kaggle CLI:\n"
        "       kaggle competitions download -c ieee-fraud-detection -p data/raw\n"
        "       unzip data/raw/ieee-fraud-detection.zip -d data/raw\n"
        f"  3. Ensure these two files exist in {cfg.paths.raw_dir}:\n"
        f"       {cfg.data.transaction_file}\n"
        f"       {cfg.data.identity_file}\n"
        "     (The test_*.csv files are unlabelled and are not used.)"
    )
