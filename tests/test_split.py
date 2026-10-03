from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import Config
from fraudlens.data.split import sort_by_time, time_split


def _frame(times: list[int]) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {"TransactionID": np.arange(n), "TransactionDT": times, "isFraud": np.zeros(n, dtype=int)}
    )


def test_blocks_are_contiguous_and_do_not_overlap_in_time(tmp_config: Config) -> None:
    rng = np.random.default_rng(0)
    df = sort_by_time(_frame(list(rng.integers(0, 10_000, 1000))), "TransactionDT", "TransactionID")
    s = time_split(df, tmp_config)
    assert s.train["TransactionDT"].max() < s.valid["TransactionDT"].min()
    assert s.valid["TransactionDT"].max() < s.test["TransactionDT"].min()
    assert len(s.train) + len(s.valid) + len(s.test) == len(df)
    assert set(s.train["TransactionID"]).isdisjoint(s.test["TransactionID"])


def test_fractions_are_respected(tmp_config: Config) -> None:
    df = _frame(list(range(10_000)))
    s = time_split(df, tmp_config)
    assert len(s.train) == pytest.approx(7000, abs=1)
    assert len(s.valid) == pytest.approx(1500, abs=1)


def test_timestamp_ties_never_straddle_a_boundary(tmp_config: Config) -> None:
    # 100 rows share every timestamp, so the 70% cut falls inside a tie group.
    df = _frame([t for t in range(20) for _ in range(100)])
    s = time_split(df, tmp_config)
    assert s.train["TransactionDT"].max() < s.valid["TransactionDT"].min()
    assert s.valid["TransactionDT"].max() < s.test["TransactionDT"].min()


def test_unsorted_input_is_rejected(tmp_config: Config) -> None:
    with pytest.raises(ValueError, match="sorted"):
        time_split(_frame([5, 3, 1, 2, 4] * 10), tmp_config)


def test_sort_is_deterministic_with_ties() -> None:
    df = pd.DataFrame({"TransactionID": [3, 1, 2], "TransactionDT": [10, 10, 5]})
    assert sort_by_time(df, "TransactionDT", "TransactionID")["TransactionID"].tolist() == [2, 1, 3]
