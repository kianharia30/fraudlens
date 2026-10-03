"""Behaviour features must depend only on a transaction's past.

The core property: for any cut point k, changing or appending rows *after* k must not
change the features of rows ``<= k``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import Config
from fraudlens.data.split import sort_by_time
from fraudlens.data.synthetic import make_synthetic
from fraudlens.features.behaviour import (
    add_behaviour_features,
    behaviour_feature_names,
    novelty_flag,
    window_counts,
)


@pytest.fixture
def timeline() -> pd.DataFrame:
    tx, ident = make_synthetic(n_rows=1500, n_cards=60, seed=3)
    df = tx.merge(ident, on="TransactionID", how="left")
    return sort_by_time(df, "TransactionDT", "TransactionID")


def _features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return add_behaviour_features(df, cfg)[behaviour_feature_names(cfg)]


@pytest.mark.parametrize("cut", [100, 700, 1200])
def test_mutating_future_rows_does_not_change_past_features(
    timeline: pd.DataFrame, tmp_config: Config, cut: int
) -> None:
    before = _features(timeline, tmp_config)
    future = timeline.copy()
    rng = np.random.default_rng(cut)
    later = future.index > cut
    future.loc[later, "TransactionAmt"] = rng.uniform(1, 5000, later.sum())
    future.loc[later, "P_emaildomain"] = "attacker.example"
    future.loc[later, "DeviceInfo"] = "EvilDevice"
    future.loc[later, "addr1"] = 999.0
    after = _features(future, tmp_config)
    pd.testing.assert_frame_equal(before.iloc[: cut + 1], after.iloc[: cut + 1])


def test_appending_future_rows_does_not_change_past_features(
    timeline: pd.DataFrame, tmp_config: Config
) -> None:
    head = timeline.iloc[:1000].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        _features(head, tmp_config), _features(timeline, tmp_config).iloc[:1000]
    )


def _card(times: list[int], amounts: list[float]) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "TransactionDT": times,
            "TransactionAmt": amounts,
            "card1": 1.0,
            "card2": 2.0,
            "card3": 3.0,
            "card5": 5.0,
            "addr1": 10.0,
            "P_emaildomain": ["a.com", "a.com", "b.com", "a.com"][:n],
        }
    )


def test_expanding_stats_exclude_the_current_row(tmp_config: Config) -> None:
    out = add_behaviour_features(_card([0, 10, 20, 30], [10.0, 20.0, 30.0, 100.0]), tmp_config)
    assert np.isnan(out["amt_to_card_mean_ratio"].iloc[0])  # no history yet
    assert out["amt_to_card_mean_ratio"].iloc[1] == pytest.approx(2.0)  # 20 / mean(10)
    assert out["amt_to_card_mean_ratio"].iloc[3] == pytest.approx(100 / 20)  # mean(10,20,30)
    assert out["amt_to_card_median_ratio"].iloc[3] == pytest.approx(100 / 20)
    assert out["card_txn_count_prior"].tolist() == [0, 1, 2, 3]
    assert out["secs_since_prev_card_txn"].iloc[2] == 10


def test_window_counts_are_strictly_before_now() -> None:
    times = np.array([0, 100, 3_600, 3_700, 3_700])
    codes = np.zeros(5, dtype=np.int64)
    counts = window_counts(times, codes, window=3_600)
    # t=3600 sees [0, 3600): rows at 0 and 100. Simultaneous rows never count each other.
    assert counts.tolist() == [0, 1, 2, 2, 2]


def test_window_counts_do_not_mix_cards() -> None:
    times = np.array([0, 10, 20])
    assert window_counts(times, np.array([0, 1, 0]), 3600).tolist() == [0, 0, 1]


def test_novelty_flag() -> None:
    values = pd.Series(["a", "a", "b", None, "a"])
    codes = np.zeros(5, dtype=np.int64)
    has_prior = np.array([False, True, True, True, True])
    out = novelty_flag(values, codes, has_prior)
    assert np.isnan(out[0])  # first transaction: no history to compare
    assert out[1] == 0 and out[2] == 1 and np.isnan(out[3]) and out[4] == 0
