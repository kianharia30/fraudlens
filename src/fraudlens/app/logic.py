"""UI-independent demo logic (sampling, history, form templates). Unit-tested.

Sampling draws only from the held-out demo pool (test block). It is deliberately
weighted towards fraud so the demo is interesting; real prevalence is ~3.5%.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from fraudlens.config import CostsConfig
from fraudlens.models.policy import (
    Decision,
    Thresholds,
    is_correct,
    outcome_label,
    transaction_cost,
)

SampleKind = Literal["random", "suspicious", "normal", "borderline"]
NON_INPUT_COLS = (
    "isFraud",
    "raw_score",
    "fraud_score",
    "decision",
    "TransactionID",
    "TransactionDT",
)


def _pick(rows: pd.DataFrame, rng: np.random.Generator) -> pd.Series | None:
    if rows.empty:
        return None
    return rows.iloc[int(rng.integers(len(rows)))]


def candidates(pool: pd.DataFrame, kind: SampleKind, thresholds: Thresholds) -> pd.DataFrame:
    """Rows eligible for a sample kind (scores are the pool's precomputed calibrated scores)."""
    s = pool["fraud_score"]
    if kind == "suspicious":
        return pool[s >= thresholds.review]
    if kind == "normal":
        return pool[s < thresholds.review]
    if kind == "borderline":
        return pool[(s >= thresholds.review) & (s < thresholds.block)]
    return pool


def sample_transaction(
    pool: pd.DataFrame,
    kind: SampleKind,
    thresholds: Thresholds,
    rng: np.random.Generator,
    fraud_weight: float,
    y_col: str = "isFraud",
) -> pd.Series | None:
    """One demo row, or None if no row matches (e.g. no borderline cases)."""
    if kind == "random":
        want_fraud = rng.random() < fraud_weight
        row = _pick(pool[(pool[y_col] == 1) == want_fraud], rng)
        return row if row is not None else _pick(pool, rng)
    return _pick(candidates(pool, kind, thresholds), rng)


def to_record(row: Mapping[str, Any] | pd.Series, known_fields: Iterable[str]) -> dict[str, Any]:
    """Model-input dict from a pool row (drops labels/scores; NaN -> None)."""
    known = set(known_fields)
    out: dict[str, Any] = {}
    for key, value in dict(row).items():
        if key in NON_INPUT_COLS or key not in known:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            value = None
        out[key] = value
    return out


def typical_row(pool: pd.DataFrame, y_col: str = "isFraud") -> pd.Series:
    """A real genuine transaction whose score is closest to the median genuine score."""
    genuine = pool[pool[y_col] == 0]
    base = genuine if not genuine.empty else pool
    median = base["fraud_score"].median()
    return base.iloc[int((base["fraud_score"] - median).abs().to_numpy().argmin())]


@dataclass(frozen=True)
class HistoryEntry:
    transaction_id: int
    amount: float
    fraud_score: float
    decision: str
    actual: str
    outcome: str
    correct: bool
    saved_vs_do_nothing: float

    def as_row(self) -> dict[str, Any]:
        return {
            "Transaction": self.transaction_id,
            "Amount (£)": round(self.amount, 2),
            "Score": round(self.fraud_score, 3),
            "Decision": self.decision,
            "Actual": self.actual,
            "Outcome": self.outcome,
            "Correct": "✓" if self.correct else "✕",
            "£ vs do-nothing": round(self.saved_vs_do_nothing, 2),
        }


def history_entry(
    transaction_id: int,
    amount: float,
    fraud_score: float,
    decision: Decision,
    is_fraud: bool,
    costs: CostsConfig,
) -> HistoryEntry:
    """Outcome of one call. '£ vs do-nothing' > 0 means the model saved money here."""
    saved = transaction_cost(Decision.APPROVE, is_fraud, amount, costs) - transaction_cost(
        decision, is_fraud, amount, costs
    )
    return HistoryEntry(
        transaction_id=transaction_id,
        amount=amount,
        fraud_score=fraud_score,
        decision=decision.value,
        actual="FRAUD" if is_fraud else "GENUINE",
        outcome=outcome_label(decision, is_fraud),
        correct=is_correct(decision, is_fraud),
        saved_vs_do_nothing=saved,
    )


def tally(entries: list[HistoryEntry]) -> dict[str, float]:
    return {
        "n": len(entries),
        "correct": sum(e.correct for e in entries),
        "incorrect": sum(not e.correct for e in entries),
        "saved": float(sum(e.saved_vs_do_nothing for e in entries)),
    }


@dataclass(frozen=True)
class FormField:
    name: str
    label: str
    kind: Literal["number", "int", "choice"]
    help: str = ""
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    options: tuple[str, ...] = ()


MANUAL_FIELDS: tuple[FormField, ...] = (
    FormField(
        "TransactionAmt", "Amount (£)", "number", min_value=0.01, max_value=50_000.0, step=1.0
    ),
    FormField(
        "hour_of_day",
        "Hour of day (relative)",
        "int",
        "TransactionDT's origin is undisclosed, so hours are relative",
        0,
        23,
        1,
    ),
    FormField("ProductCD", "Product code", "choice", options=("W", "C", "R", "H", "S")),
    FormField(
        "card4",
        "Card network",
        "choice",
        options=("visa", "mastercard", "american express", "discover"),
    ),
    FormField(
        "card6",
        "Card type",
        "choice",
        options=("debit", "credit", "charge card", "debit or credit"),
    ),
    FormField("P_emaildomain", "Purchaser email domain", "choice"),
    FormField(
        "R_emaildomain",
        "Recipient email domain",
        "choice",
        "A different recipient domain sets the mismatch flag",
    ),
    FormField("DeviceType", "Device type", "choice", options=("desktop", "mobile")),
    FormField(
        "dist1",
        "Address distance (dist1)",
        "number",
        "Distance between addresses (anonymised units)",
        0.0,
        10_000.0,
        1.0,
    ),
    FormField(
        "card_txn_count_1h",
        "Card transactions in last hour",
        "int",
        min_value=0,
        max_value=100,
        step=1,
    ),
    FormField(
        "card_txn_count_24h",
        "Card transactions in last 24h",
        "int",
        min_value=0,
        max_value=500,
        step=1,
    ),
    FormField(
        "amt_to_card_mean_ratio",
        "Amount vs card average (x)",
        "number",
        "Leave empty if the card has no history",
        0.0,
        1_000.0,
        0.1,
    ),
    FormField(
        "is_new_DeviceInfo_for_card", "New device for this card", "choice", options=("yes", "no")
    ),
    FormField(
        "is_new_P_emaildomain_for_card",
        "New email domain for this card",
        "choice",
        options=("yes", "no"),
    ),
)
_FLAG_FIELDS = {"is_new_DeviceInfo_for_card", "is_new_P_emaildomain_for_card"}


def form_value_from_record(field: FormField, record: Mapping[str, Any]) -> Any:
    """Widget value for a field, given a model-input record."""
    value = record.get(field.name)
    if value is None:
        return None
    if field.name in _FLAG_FIELDS:
        return "yes" if float(value) >= 0.5 else "no"
    if field.kind in ("int", "number"):
        number = float(value)
        if field.min_value is not None:
            number = max(number, field.min_value)
        if field.max_value is not None:
            number = min(number, field.max_value)  # widgets reject out-of-range values
        return int(round(number)) if field.kind == "int" else number
    return str(value)


def apply_form(template: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Template record overridden by form values (None = missing)."""
    record = dict(template)
    for field in MANUAL_FIELDS:
        if field.name not in values:
            continue
        value = values[field.name]
        if field.name in _FLAG_FIELDS and value is not None:
            value = 1.0 if value == "yes" else 0.0
        record[field.name] = value
    return record


def choice_options(field: FormField, pool: pd.DataFrame, limit: int = 25) -> list[str]:
    """Options for a choice field: fixed list, or the most common values in the pool."""
    if field.options:
        return list(field.options)
    if field.name not in pool.columns:
        return []
    return [str(v) for v in pool[field.name].dropna().astype(str).value_counts().head(limit).index]
