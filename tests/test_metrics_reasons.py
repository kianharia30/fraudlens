from __future__ import annotations

import numpy as np
import pytest

from fraudlens.explain.reasons import group_of, humanise_seconds, top_reasons
from fraudlens.models.calibration import ScoreCalibrator
from fraudlens.models.metrics import confusion, recall_at_precision, reliability_table


def test_recall_at_precision() -> None:
    y = np.array([1, 1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    recall, thr = recall_at_precision(y, s, 0.9)
    assert recall == pytest.approx(2 / 3) and thr == pytest.approx(0.8)
    assert recall_at_precision(np.array([0, 1]), np.array([0.9, 0.1]), 0.99)[0] == 0.0


def test_confusion() -> None:
    assert confusion(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])) == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
    }


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibration_fixes_inflated_scores(method: str) -> None:
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0, 0.3, 20_000)
    y = (rng.random(20_000) < p_true).astype(int)
    inflated = np.sqrt(p_true)  # systematically too high, like class-weighted scores
    cal = ScoreCalibrator(method).fit(inflated, y).transform(inflated)
    assert abs(cal.mean() - y.mean()) < 0.01 < abs(inflated.mean() - y.mean())


def test_reliability_table_drops_sparse_bins() -> None:
    t = reliability_table(np.array([0] * 30 + [1]), np.array([0.05] * 30 + [0.95]), min_count=20)
    assert len(t) == 1


def test_grouping_is_honest_about_anonymised_features() -> None:
    assert group_of("V258") == "anon_V" and group_of("C13") == "anon_C"
    assert group_of("id_01") == "anon_id"
    assert group_of("id_31") == "id_31" and group_of("id_31_freq") == "id_31"  # browser
    assert group_of("amt_log") == group_of("TransactionAmt") == "amount"
    assert group_of("P_emaildomain_freq") == "P_emaildomain"


def test_top_reasons_text_and_ranking() -> None:
    shap = {
        "amt_to_card_mean_ratio": 1.5,
        "amt_to_card_median_ratio": 0.5,
        "V1": 0.3,
        "V2": 0.4,
        "card_txn_count_1h": -0.2,
        "TransactionAmt": 0.01,
    }
    values = {
        "amt_to_card_mean_ratio": 27.0,
        "amt_to_card_median_ratio": 30.0,
        "V1": 1.0,
        "V2": 2.0,
        "card_txn_count_1h": 0.0,
        "TransactionAmt": 500.0,
    }
    reasons = top_reasons(shap, values, {"TransactionAmt": 500.0}, top_n=3)
    assert [r.group for r in reasons] == ["amount_vs_card", "anon_V", "card_txn_count_1h"]
    assert (
        reasons[0].text
        == "Amount is 27.0x this card's average and 30.0x its typical (median) amount"
    )
    assert reasons[0].shap_value == pytest.approx(2.0) and reasons[0].direction == "raises risk"
    assert "Anonymised" in reasons[1].text and "2 columns combined" in reasons[1].text
    assert reasons[2].direction == "lowers risk"


def test_missing_history_reason() -> None:
    reasons = top_reasons(
        {"is_new_DeviceInfo_for_card": 0.4}, {"is_new_DeviceInfo_for_card": float("nan")}, {}
    )
    assert reasons[0].text == "No card history to check the device against"


def test_humanise_seconds() -> None:
    assert humanise_seconds(30) == "30 seconds"
    assert humanise_seconds(3 * 3600) == "3 hours"
    assert humanise_seconds(86_400) == "1 day"
