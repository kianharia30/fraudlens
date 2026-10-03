from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.app.logic import (
    MANUAL_FIELDS,
    apply_form,
    candidates,
    form_value_from_record,
    history_entry,
    sample_transaction,
    tally,
    to_record,
    typical_row,
)
from fraudlens.config import CostsConfig
from fraudlens.models.policy import Decision, Thresholds

THR = Thresholds(review=0.3, block=0.7)
COSTS = CostsConfig(
    currency="GBP",
    missed_fraud_multiplier=1.0,
    false_alarm_cost=5.0,
    review_cost=2.0,
    review_catch_rate=1.0,
)


@pytest.fixture
def pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": range(6),
            "isFraud": [1, 1, 0, 0, 0, 1],
            "fraud_score": [0.9, 0.1, 0.05, 0.5, 0.8, 0.4],
            "TransactionAmt": [100.0, 50.0, 10.0, 20.0, 30.0, 40.0],
            "card1": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
        }
    )


def test_candidates(pool: pd.DataFrame) -> None:
    assert candidates(pool, "borderline", THR)["TransactionID"].tolist() == [3, 5]
    assert set(candidates(pool, "normal", THR)["TransactionID"]) == {1, 2}


def test_random_sampling_is_weighted(pool: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    draws = [sample_transaction(pool, "random", THR, rng, fraud_weight=0.4) for _ in range(2000)]
    share = np.mean([d["isFraud"] for d in draws if d is not None])
    assert share == pytest.approx(0.4, abs=0.04)


def test_empty_kind_returns_none(pool: pd.DataFrame) -> None:
    only_genuine = pool[pool["isFraud"] == 0]
    assert (
        sample_transaction(only_genuine.iloc[:1], "borderline", THR, np.random.default_rng(0), 0.4)
        is None
    )


def test_to_record_drops_labels_and_nans(pool: pd.DataFrame) -> None:
    rec = to_record(pool.iloc[3], {"card1", "TransactionAmt", "isFraud", "fraud_score"})
    assert rec == {"TransactionAmt": 20.0, "card1": None}


def test_typical_row_is_genuine(pool: pd.DataFrame) -> None:
    assert typical_row(pool)["isFraud"] == 0


def test_history_and_tally() -> None:
    caught = history_entry(1, 100.0, 0.9, Decision.BLOCK, True, COSTS)
    false_alarm = history_entry(2, 30.0, 0.8, Decision.BLOCK, False, COSTS)
    assert caught.saved_vs_do_nothing == 100.0 and caught.correct
    assert false_alarm.saved_vs_do_nothing == -5.0 and not false_alarm.correct
    assert tally([caught, false_alarm]) == {"n": 2, "correct": 1, "incorrect": 1, "saved": 95.0}


def test_form_round_trip() -> None:
    template = {
        "TransactionAmt": 75000.0,
        "is_new_DeviceInfo_for_card": 1.0,
        "hour_of_day": 5.0,
        "V1": 3.0,
    }
    fields = {f.name: f for f in MANUAL_FIELDS}
    assert (
        form_value_from_record(fields["TransactionAmt"], template) == 50_000.0
    )  # clamped to widget max
    assert form_value_from_record(fields["is_new_DeviceInfo_for_card"], template) == "yes"
    assert form_value_from_record(fields["hour_of_day"], template) == 5
    record = apply_form(template, {"TransactionAmt": 12.0, "is_new_DeviceInfo_for_card": "no"})
    assert record == {
        "TransactionAmt": 12.0,
        "is_new_DeviceInfo_for_card": 0.0,
        "hour_of_day": 5.0,
        "V1": 3.0,
    }
