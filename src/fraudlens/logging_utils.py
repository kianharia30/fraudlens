"""Project-wide logging setup. Use ``get_logger(__name__)`` instead of ``print``."""

from __future__ import annotations

import logging
import os

LOG_LEVEL_ENV_VAR = "FRAUDLENS_LOG_LEVEL"
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | int | None = None) -> None:
    """Configure the root logger once. Safe to call repeatedly.

    Level resolution: explicit argument > ``$FRAUDLENS_LOG_LEVEL`` > INFO.
    """
    resolved = level if level is not None else os.environ.get(LOG_LEVEL_ENV_VAR, "INFO")
    if isinstance(resolved, str):
        resolved = resolved.upper()
    logging.basicConfig(level=resolved, format=_LOG_FORMAT, datefmt=_DATE_FORMAT, force=True)
    # Third-party libraries are noisy at INFO.
    for noisy in ("matplotlib", "urllib3", "httpx", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
