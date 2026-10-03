"""Cost-based APPROVE / REVIEW / BLOCK policy.

Decision rule for a calibrated fraud score ``s`` and thresholds ``t_review <= t_block``:

    s <  t_review           -> APPROVE
    t_review <= s < t_block -> REVIEW   (sent to a human analyst)
    s >= t_block            -> BLOCK

Per-transaction cost of each outcome (all configurable, in GBP):

    APPROVE fraud    : missed_fraud_multiplier * amount
    APPROVE genuine  : 0
    REVIEW  any      : review_cost  (+ (1 - review_catch_rate) * missed-fraud cost for frauds)
    BLOCK   fraud    : 0            (fraud prevented)
    BLOCK   genuine  : false_alarm_cost

Thresholds are chosen on the validation block by exhaustive search over a quantile grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from fraudlens.config import CostsConfig

NEVER = 1.000001  # threshold above any probability: "never take this action"


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


_CODES = (Decision.APPROVE, Decision.REVIEW, Decision.BLOCK)


@dataclass(frozen=True)
class Thresholds:
    review: float
    block: float

    def __post_init__(self) -> None:
        if self.review > self.block:
            raise ValueError(f"review threshold {self.review} must be <= block {self.block}")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def decide(score: float, thresholds: Thresholds) -> Decision:
    """Decision for a single score."""
    return _CODES[int(decide_codes(np.array([score]), thresholds)[0])]


def decide_codes(scores: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    """Vectorised decisions as codes: 0 = APPROVE, 1 = REVIEW, 2 = BLOCK."""
    s = np.asarray(scores, dtype=np.float64)
    return (s >= thresholds.review).astype(np.int8) + (s >= thresholds.block).astype(np.int8)


def code_to_decision(code: int) -> Decision:
    return _CODES[code]


def _outcome_cost_table(
    y: np.ndarray, amount: np.ndarray, costs: CostsConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row cost of APPROVE, REVIEW and BLOCK."""
    y = np.asarray(y, dtype=np.float64)
    missed = costs.missed_fraud_multiplier * np.asarray(amount, dtype=np.float64)
    approve = y * missed
    review = costs.review_cost + y * (1 - costs.review_catch_rate) * missed
    block = (1 - y) * costs.false_alarm_cost
    return approve, review, block


def transaction_cost(
    decision: Decision, is_fraud: bool, amount: float, costs: CostsConfig
) -> float:
    """Cost (GBP) of one decision given the true label."""
    approve, review, block = _outcome_cost_table(
        np.array([float(is_fraud)]), np.array([amount]), costs
    )
    return float(
        {Decision.APPROVE: approve, Decision.REVIEW: review, Decision.BLOCK: block}[decision][0]
    )


def outcome_label(decision: Decision, is_fraud: bool) -> str:
    """Plain-English outcome used in the app's history table."""
    if is_fraud:
        return {
            Decision.BLOCK: "Fraud blocked",
            Decision.REVIEW: "Fraud sent to review",
            Decision.APPROVE: "Fraud missed",
        }[decision]
    return {
        Decision.APPROVE: "Genuine approved",
        Decision.REVIEW: "Genuine reviewed",
        Decision.BLOCK: "Genuine blocked (false alarm)",
    }[decision]


def is_correct(decision: Decision, is_fraud: bool) -> bool:
    """Fraud must be stopped (REVIEW or BLOCK); genuine must not be blocked."""
    return decision != Decision.APPROVE if is_fraud else decision != Decision.BLOCK


@dataclass(frozen=True)
class PolicySummary:
    """Outcome of applying a policy to a labelled set of transactions."""

    thresholds: dict[str, float]
    total_cost: float
    n: int
    n_approve: int
    n_review: int
    n_block: int
    n_fraud: int
    fraud_stopped: int  # reviewed or blocked
    fraud_blocked: int
    fraud_amount_total: float
    fraud_amount_stopped: float
    genuine_blocked: int
    genuine_reviewed: int

    @property
    def fraud_recall(self) -> float:
        return self.fraud_stopped / self.n_fraud if self.n_fraud else 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"fraud_recall": self.fraud_recall}


def summarise_policy(
    scores: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    thresholds: Thresholds,
    costs: CostsConfig,
) -> PolicySummary:
    codes = decide_codes(scores, thresholds)
    y = np.asarray(y).astype(bool)
    amount = np.asarray(amount, dtype=np.float64)
    approve, review, block = _outcome_cost_table(y, amount, costs)
    per_row = np.choose(codes, [approve, review, block])
    stopped = codes > 0
    return PolicySummary(
        thresholds=thresholds.as_dict(),
        total_cost=float(per_row.sum()),
        n=len(codes),
        n_approve=int((codes == 0).sum()),
        n_review=int((codes == 1).sum()),
        n_block=int((codes == 2).sum()),
        n_fraud=int(y.sum()),
        fraud_stopped=int((stopped & y).sum()),
        fraud_blocked=int(((codes == 2) & y).sum()),
        fraud_amount_total=float(amount[y].sum()),
        fraud_amount_stopped=float(amount[stopped & y].sum()),
        genuine_blocked=int(((codes == 2) & ~y).sum()),
        genuine_reviewed=int(((codes == 1) & ~y).sum()),
    )


def do_nothing_thresholds() -> Thresholds:
    """The 'approve everything' baseline policy."""
    return Thresholds(review=NEVER, block=NEVER)


def optimise_thresholds(
    scores: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    costs: CostsConfig,
    grid_size: int,
    max_review_rate: float = 1.0,
    max_block_rate: float = 1.0,
) -> tuple[Thresholds, float]:
    """Exhaustively search (t_review, t_block) on a score-quantile grid; minimise total cost.

    Only pairs within the operational budgets are allowed: at most ``max_review_rate`` of
    transactions sent to analysts, and at most ``max_block_rate`` declined. Without them,
    cheap per-item review and false-alarm costs push the optimum to reviewing or blocking
    huge volumes, which no fraud team could staff or customers would tolerate.

    Uses prefix sums over rows sorted by score, so each of the ~grid_size^2 / 2 candidate
    pairs is evaluated in O(1). Returns the best thresholds and their total cost.
    """
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    approve, review, block = (
        c[order] for c in _outcome_cost_table(np.asarray(y), np.asarray(amount), costs)
    )
    zero = np.zeros(1)
    cum_a = np.concatenate([zero, np.cumsum(approve)])
    cum_r = np.concatenate([zero, np.cumsum(review)])
    cum_b = np.concatenate([zero, np.cumsum(block)])

    qs = np.quantile(s_sorted, np.linspace(0, 1, grid_size))
    candidates = np.unique(np.concatenate([[0.0], qs, [NEVER]]))
    idx = np.searchsorted(s_sorted, candidates, side="left")  # rows with score < candidate

    i_low, i_high = idx[:, None], idx[None, :]
    total = cum_a[i_low] + (cum_r[i_high] - cum_r[i_low]) + (cum_b[-1] - cum_b[i_high])
    n = len(s_sorted)
    feasible = (
        (i_low <= i_high)
        & ((i_high - i_low) <= max_review_rate * n)  # rows in [t_review, t_block)
        & ((n - i_high) <= max_block_rate * n)  # rows >= t_block
    )
    # Always feasible: review=block=NEVER (approve everything).
    total = np.where(feasible, total, np.inf)
    lo, hi = np.unravel_index(int(np.argmin(total)), total.shape)
    return Thresholds(review=float(candidates[lo]), block=float(candidates[hi])), float(
        total[lo, hi]
    )
