from __future__ import annotations

from pathlib import Path

from fraudlens.config import Config
from fraudlens.data.validation import (
    REQUIRED_IDENTITY_COLS,
    REQUIRED_TRANSACTION_COLS,
    setup_instructions,
    validate_raw_data,
)
from scripts.validate_data import main as validate_main


def _write_header(path: Path, cols: frozenset[str] | set[str]) -> None:
    path.write_text(",".join(sorted(cols)) + "\n")


def test_missing_files_are_reported(tmp_config: Config) -> None:
    report = validate_raw_data(tmp_config)
    assert not report.ok
    assert len(report.problems) == 2
    assert all("Missing file" in p for p in report.problems)


def test_valid_files_pass(tmp_config: Config) -> None:
    raw = tmp_config.paths.raw_dir
    _write_header(raw / tmp_config.data.transaction_file, REQUIRED_TRANSACTION_COLS)
    _write_header(raw / tmp_config.data.identity_file, REQUIRED_IDENTITY_COLS)
    assert validate_raw_data(tmp_config).ok


def test_wrong_schema_is_reported(tmp_config: Config) -> None:
    raw = tmp_config.paths.raw_dir
    _write_header(raw / tmp_config.data.transaction_file, REQUIRED_TRANSACTION_COLS - {"isFraud"})
    _write_header(raw / tmp_config.data.identity_file, REQUIRED_IDENTITY_COLS)
    report = validate_raw_data(tmp_config)
    assert report.problems == [
        f"{tmp_config.data.transaction_file} is missing expected columns: isFraud"
    ]


def test_empty_file_is_reported(tmp_config: Config) -> None:
    raw = tmp_config.paths.raw_dir
    (raw / tmp_config.data.transaction_file).touch()
    _write_header(raw / tmp_config.data.identity_file, REQUIRED_IDENTITY_COLS)
    assert any("empty" in p for p in validate_raw_data(tmp_config).problems)


def test_instructions_name_both_files(tmp_config: Config) -> None:
    text = setup_instructions(tmp_config)
    assert tmp_config.data.transaction_file in text
    assert tmp_config.data.identity_file in text


def test_cli_exit_codes(tmp_project: Path, tmp_config: Config) -> None:
    config_arg = ["--config", str(tmp_project / "configs" / "config.yaml")]
    assert validate_main(config_arg) == 1
    raw = tmp_config.paths.raw_dir
    _write_header(raw / tmp_config.data.transaction_file, REQUIRED_TRANSACTION_COLS)
    _write_header(raw / tmp_config.data.identity_file, REQUIRED_IDENTITY_COLS)
    assert validate_main(config_arg) == 0
