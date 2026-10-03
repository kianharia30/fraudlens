"""End-to-end dataset build: raw CSVs -> time-sorted features -> train/valid/test parquet."""

from __future__ import annotations

import gc
import time
from pathlib import Path

from fraudlens.config import Config
from fraudlens.data.loading import load_merged, memory_mb
from fraudlens.data.split import describe_splits, save_splits, sort_by_time, time_split
from fraudlens.features.behaviour import add_behaviour_features
from fraudlens.logging_utils import get_logger

logger = get_logger(__name__)


def build_dataset(cfg: Config, nrows: int | None = None) -> Path:
    """Build processed splits.

    Behaviour features are computed on the *whole* sorted timeline before splitting. This
    is safe because they only look backwards: a validation row may use the card's training
    period history (as it would in production), but no row ever sees a later row. Anything
    *fitted* (encoders, models, calibrators, thresholds) is fitted after the split.
    """
    start = time.perf_counter()
    df = load_merged(cfg, nrows=nrows)
    df = sort_by_time(df, cfg.data.time_col, cfg.data.id_col)
    df = add_behaviour_features(df, cfg)
    logger.info("Features added: %d cols, %.0f MB", df.shape[1], memory_mb(df))

    splits = time_split(df, cfg)
    del df
    gc.collect()
    for info in describe_splits(splits, cfg):
        logger.info(
            "%-5s rows=%7d fraud=%.2f%% time=[%d, %d]",
            info.name,
            info.n_rows,
            100 * info.fraud_rate,
            info.time_min,
            info.time_max,
        )
    out = save_splits(splits, cfg)
    logger.info("Saved splits to %s in %.0fs", out, time.perf_counter() - start)
    return out
