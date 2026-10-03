"""Load the raw IEEE-CIS tables with memory-efficient dtypes and merge them.

Memory strategy (the full transaction table is ~590k rows x 394 columns):
* dtypes are chosen *at read time* from the header, so pandas never materialises a
  float64 copy of the V/C/D columns: anonymised numeric columns become ``float32``,
  string columns become ``category``;
* ``TransactionAmt`` stays ``float64`` so currency values are exact to the penny;
* the CSV is read in chunks, then remaining numeric columns are downcast.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.config import Config
from fraudlens.logging_utils import get_logger

logger = get_logger(__name__)

_CHUNK_ROWS = 100_000
# Families of columns that are always numeric in IEEE-CIS: read straight into float32.
_FLOAT32_PATTERN = re.compile(r"^(V\d+|C\d+|D\d+|dist\d|card[1235]|addr[12]|id_0\d|id_1[01])$")
_EXACT_COLS: frozenset[str] = frozenset({"TransactionAmt"})


def _dtype_map(columns: pd.Index) -> dict[str, str]:
    """float32 for known-numeric families; everything else is inferred, then compacted."""
    return {col: "float32" for col in columns if _FLOAT32_PATTERN.match(col)}


def _compact_inferred(df: pd.DataFrame) -> pd.DataFrame:
    """Strings -> category, float64 -> float32 (except exact money columns), ints downcast."""
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
        elif dtype == np.float64 and col not in _EXACT_COLS:
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def read_csv_compact(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV with compact dtypes, chunk by chunk to bound peak memory."""
    header = pd.read_csv(path, nrows=0).columns
    reader = pd.read_csv(path, dtype=_dtype_map(header), chunksize=_CHUNK_ROWS, nrows=nrows)
    chunks = [_compact_inferred(chunk) for chunk in reader]
    # Categories differ between chunks; concat would fall back to object, so re-compact.
    return _compact_inferred(pd.concat(chunks, ignore_index=True))


def memory_mb(df: pd.DataFrame) -> float:
    """Deep memory usage of a frame in megabytes."""
    return float(df.memory_usage(deep=True).sum()) / 1e6


def load_merged(cfg: Config, nrows: int | None = None) -> pd.DataFrame:
    """Load transactions and identity, left-joined on the id column.

    Not every transaction has identity information: rows without a match get NaN
    identity columns, which LightGBM treats as "missing" natively.
    """
    raw = cfg.paths.raw_dir
    id_col = cfg.data.id_col

    logger.info("Reading %s", cfg.data.transaction_file)
    tx = read_csv_compact(raw / cfg.data.transaction_file, nrows=nrows)
    logger.info("Reading %s", cfg.data.identity_file)
    ident = read_csv_compact(raw / cfg.data.identity_file)

    if not ident[id_col].is_unique:
        raise ValueError(f"{cfg.data.identity_file} has duplicate {id_col} values")

    df = tx.merge(ident, on=id_col, how="left", validate="one_to_one")
    for col in ident.columns:
        if isinstance(ident[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    logger.info(
        "Merged: %d rows x %d cols, %.0f MB, fraud rate %.2f%%, identity match %.1f%%",
        len(df),
        df.shape[1],
        memory_mb(df),
        100 * float(df[cfg.data.target_col].mean()),
        100 * float(np.isin(tx[id_col], ident[id_col]).mean()),
    )
    return df
