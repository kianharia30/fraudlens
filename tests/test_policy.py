from __future__ import annotations

import itertools

import numpy as np
import pytest

from fraudlens.config import CostsConfig
from fraudlens.models.policy import (
    NEVER,
    Decision,
    Thresholds,
    decide,
    decide_codes,
    do_nothing_thresholds,
    is_correct,
    optimise_thresholds,
    summarise_policy,
    transaction_cost,
)

COSTS = CostsConfig(
    currency="GBP",
    missed_fraud_multiplier=1.0,
    false_alarm_cost=5.0,
    review_cost=2.0,
    review_catch_rate=1.0,
)


def test_decision_boundaries_are_inclusive_at_the_threshold() -> None:
    t = Thresholds(review=0.2, block=0.8)
    assert decide(0.1999, t) is Decision.APPROVE
    assert decide(0.2, t) is Decision.REVIEW
    assert decide(0.7999, t) is Decision.REVIEW
    assert decide(0.8, t) is Decision.BLOCK
    assert decide(1.0, do_nothing_thresholds()) is Decision.APPROVE


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        Thresholds(review=0.9, block=0.1)


@pytest.mark.parametrize(
    ("decision", "fraud", "expected"),
    [
        (Decision.APPROVE, True, 120.0),
        (Decision.APPROVE, False, 0.0),
        (Decision.REVIEW, True, 2.0),
        (Decision.REVIEW, False, 2.0),
        (Decision.BLOCK, True, 0.0),
        (Decision.BLOCK, False, 5.0),
    ],
)
def test_transaction_cost(decision: Decision, fraud: bool, expected: float) -> None:
    assert transaction_cost(decision, fraud, 120.0, COSTS) == expected


def test_review_catch_rate_adds_expected_miss_cost() -> None:
    costs = COSTS.model_copy(update={"review_catch_rate": 0.75})
    assert transaction_cost(Decision.REVIEW, True, 100.0, costs) == pytest.approx(2 + 25)


def test_correctness() -> None:
    assert is_correct(Decision.REVIEW, True) and is_correct(Decision.BLOCK, True)
    assert not is_correct(Decision.APPROVE, True)
    assert is_correct(Decision.REVIEW, False) and not is_correct(Decision.BLOCK, False)


def test_summary_counts_and_cost() -> None:
    scores = np.array([0.1, 0.5, 0.9, 0.95])
    y = np.array([1, 0, 1, 0])
    amount = np.array([50.0, 10.0, 70.0, 30.0])
    s = summarise_policy(scores, y, amount, Thresholds(0.3, 0.9), COSTS)
    assert (s.n_approve, s.n_review, s.n_block) == (1, 1, 2)
    assert s.total_cost == pytest.approx(50 + 2 + 0 + 5)
    assert s.fraud_stopped == 1 and s.genuine_blocked == 1
    assert s.fraud_amount_stopped == 70.0


def test_optimiser_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.1).astype(int)
    scores = np.clip(rng.normal(0.2 + 0.5 * y, 0.2), 0, 1)
    amount = rng.lognormal(4, 1, n)
    thr, cost = optimise_thresholds(scores, y, amount, COSTS, grid_size=40)

    qs = np.quantile(np.sort(scores), np.linspace(0, 1, 40))
    grid = np.unique(np.concatenate([[0.0], qs, [NEVER]]))
    brute = min(
        summarise_policy(scores, y, amount, Thresholds(lo, hi), COSTS).total_cost
        for lo, hi in itertools.combinations_with_replacement(grid, 2)
    )
    assert cost == pytest.approx(brute)
    assert summarise_policy(scores, y, amount, thr, COSTS).total_cost == pytest.approx(cost)


def test_optimiser_never_worse_than_doing_nothing() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.05).astype(int)
    scores, amount = rng.random(300), rng.uniform(1, 100, 300)
    _, cost = optimise_thresholds(scores, y, amount, COSTS, grid_size=30)
    assert cost <= summarise_policy(scores, y, amount, do_nothing_thresholds(), COSTS).total_cost


def test_decide_codes_vectorised() -> None:
    codes = decide_codes(np.array([0.0, 0.5, 1.0]), Thresholds(0.5, 1.0))
    assert codes.tolist() == [0, 1, 2]


def test_optimiser_respects_review_and_block_budgets() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    y = (rng.random(n) < 0.05).astype(int)
    scores = np.clip(rng.normal(0.1 + 0.5 * y, 0.2), 0, 1)
    amount = rng.lognormal(5, 1, n)  # large amounts: unconstrained optimum reviews a lot
    free, _ = optimise_thresholds(scores, y, amount, COSTS, grid_size=60)
    free_review = summarise_policy(scores, y, amount, free, COSTS).n_review / n
    thr, _ = optimise_thresholds(
        scores, y, amount, COSTS, grid_size=60, max_review_rate=0.05, max_block_rate=0.01
    )
    s = summarise_policy(scores, y, amount, thr, COSTS)
    assert free_review > 0.05  # the budget is actually binding in this setup
    assert s.n_review / n <= 0.05 and s.n_block / n <= 0.01
