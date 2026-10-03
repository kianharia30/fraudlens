"""Request / response schemas for the scoring API.

The key, human-meaningful fields are typed and validated explicitly. Any other model
input (anonymised V/C/D/M/id columns, precomputed behaviour features) may be sent as an
extra field holding a number, string or null; the API rejects field names the model
does not know. Omitted fields are treated as missing (NaN), which the model handles.

Behaviour features (e.g. ``card_txn_count_1h``) describe the card's *history*. In
production a feature store would supply them; here the caller provides them (the demo
pool rows already carry them).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Transaction(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "TransactionAmt": 117.0,
                "ProductCD": "W",
                "card1": 13926,
                "card2": 321.0,
                "card4": "visa",
                "card6": "debit",
                "addr1": 315.0,
                "P_emaildomain": "gmail.com",
                "hour_of_day": 14,
                "card_txn_count_1h": 0,
                "amt_to_card_mean_ratio": 1.2,
            }
        },
    )

    TransactionAmt: float = Field(gt=0, le=1_000_000, description="Transaction amount (GBP)")
    ProductCD: Literal["W", "H", "C", "S", "R"] | None = None
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: Literal["visa", "mastercard", "american express", "discover"] | None = None
    card5: float | None = None
    card6: Literal["debit", "credit", "charge card", "debit or credit"] | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = Field(default=None, ge=0)
    P_emaildomain: str | None = Field(default=None, max_length=100)
    R_emaildomain: str | None = Field(default=None, max_length=100)
    DeviceType: Literal["desktop", "mobile"] | None = None
    DeviceInfo: str | None = Field(default=None, max_length=200)
    # Behaviour / context features (normally from a feature store).
    hour_of_day: float | None = Field(default=None, ge=0, lt=24)
    card_txn_count_1h: float | None = Field(default=None, ge=0)
    card_txn_count_24h: float | None = Field(default=None, ge=0)
    amt_to_card_mean_ratio: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _extras_are_scalars(self) -> Transaction:
        for key, value in (self.model_extra or {}).items():
            if value is not None and not isinstance(value, int | float | str):
                raise ValueError(f"Field '{key}' must be a number, string or null")
            if isinstance(value, bool):
                raise ValueError(f"Field '{key}' must be a number, string or null, not a boolean")
        return self

    def to_record(self) -> dict[str, Any]:
        return self.model_dump() | dict(self.model_extra or {})


class _Response(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow "model_version" etc.


class ReasonOut(BaseModel):
    group: str
    text: str
    shap_value: float
    direction: Literal["raises risk", "lowers risk"]
    features: list[str]


class ThresholdsOut(BaseModel):
    review: float
    block: float


class ScoreResponse(_Response):
    fraud_score: float = Field(description="Calibrated probability of fraud")
    raw_score: float
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    thresholds: ThresholdsOut
    top_reasons: list[ReasonOut]
    model_version: str
    latency_ms: float
    shap_values: dict[str, float] | None = None
    shap_base_value: float | None = None


class HealthResponse(_Response):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    detail: str | None = None


class ModelInfoResponse(_Response):
    model_version: str
    dataset: str
    trained_at: str | None
    git_hash: str | None
    thresholds: ThresholdsOut
    metrics_test: dict[str, Any]
    costs_test: dict[str, Any]
    n_features: int
