"""Chronological train / validation / test split.

A random split would let the model "see the future" (e.g. later transactions of the same
card, or fraud campaigns that start after the training period), which inflates metrics.
We sort by time and cut contiguous blocks instead. Boundaries never split a timestamp:
if rows share the boundary ``TransactionDT``, they all go to the earlier block, so
``max(train time) < min(valid time) < ... `` holds strictly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.config import Config


@dataclass(frozen=True)
class SplitInfo:
    """Row counts and time ranges of one split block."""

    name: str
    n_rows: int
    n_fraud: int
    time_min: int
    time_max: int

    @property
    def fraud_rate(self) -> float:
        return self.n_fraud / self.n_rows if self.n_rows else 0.0


@dataclass
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        return [("train", self.train), ("valid", self.valid), ("test", self.test)]


def sort_by_time(df: pd.DataFrame, time_col: str, id_col: str) -> pd.DataFrame:
    """Deterministic chronological order (ties broken by transaction id)."""
    return df.sort_values([time_col, id_col], kind="mergesort").reset_index(drop=True)


def _boundary(times: np.ndarray, frac: float) -> int:
    """Index of the first row of the next block, moved forward past timestamp ties."""
    idx = int(len(times) * frac)
    idx = max(1, min(idx, len(times) - 1))
    # searchsorted(side="right") skips every row sharing the boundary timestamp.
    return int(np.searchsorted(times, times[idx - 1], side="right"))


def time_split(df: pd.DataFrame, cfg: Config) -> Splits:
    """Split an already time-sorted frame into contiguous train/valid/test blocks."""
    time_col = cfg.data.time_col
    times = df[time_col].to_numpy()
    if np.any(np.diff(times) < 0):
        raise ValueError("time_split expects a frame sorted by time; call sort_by_time first")

    i_valid = _boundary(times, cfg.split.train_frac)
    i_test = _boundary(times, cfg.split.train_frac + cfg.split.valid_frac)
    if not 0 < i_valid < i_test < len(df):
        raise ValueError("Split produced an empty block; check fractions and data size")
    return Splits(
        train=df.iloc[:i_valid].reset_index(drop=True),
        valid=df.iloc[i_valid:i_test].reset_index(drop=True),
        test=df.iloc[i_test:].reset_index(drop=True),
    )


def describe_splits(splits: Splits, cfg: Config) -> list[SplitInfo]:
    t, y = cfg.data.time_col, cfg.data.target_col
    return [
        SplitInfo(
            name=name,
            n_rows=len(part),
            n_fraud=int(part[y].sum()),
            time_min=int(part[t].min()),
            time_max=int(part[t].max()),
        )
        for name, part in splits.items()
    ]


def save_splits(splits: Splits, cfg: Config) -> Path:
    """Write each block to parquet plus a ``split_meta.json`` describing the cut-offs."""
    out = cfg.paths.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(out / f"{name}.parquet", index=False)
    meta = {
        "blocks": [
            asdict(info) | {"fraud_rate": info.fraud_rate} for info in describe_splits(splits, cfg)
        ],
        "train_frac": cfg.split.train_frac,
        "valid_frac": cfg.split.valid_frac,
        "time_col": cfg.data.time_col,
    }
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def load_split(cfg: Config, name: str) -> pd.DataFrame:
    """Load one processed block (``train``, ``valid`` or ``test``)."""
    path = cfg.paths.processed_dir / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    return pd.read_parquet(path)
