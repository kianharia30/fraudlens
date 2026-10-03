"""The one scoring + explanation code path shared by the API, the app and the scripts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudlens.config import Config
from fraudlens.explain.reasons import Reason, top_reasons
from fraudlens.explain.shap_explain import ShapExplainer
from fraudlens.models.artifacts import ModelBundle, latest_bundle_dir, load_bundle
from fraudlens.models.policy import Decision, Thresholds, code_to_decision, decide_codes


@dataclass
class ScoreResult:
    fraud_score: float  # calibrated probability of fraud
    raw_score: float  # uncalibrated model output
    decision: Decision
    thresholds: Thresholds
    top_reasons: list[Reason]
    shap_values: dict[str, float] = field(repr=False)
    base_value: float = 0.0  # SHAP expected value (log-odds)
    model_version: str = ""

    def as_dict(self, include_shap: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "fraud_score": self.fraud_score,
            "raw_score": self.raw_score,
            "decision": self.decision.value,
            "thresholds": self.thresholds.as_dict(),
            "top_reasons": [r.as_dict() for r in self.top_reasons],
            "model_version": self.model_version,
        }
        if include_shap:
            out["shap_values"] = self.shap_values
            out["shap_base_value"] = self.base_value
        return out


class FraudScorer:
    """Loads a model bundle once and scores / explains transactions."""

    def __init__(self, bundle: ModelBundle, top_n: int = 5) -> None:
        self.bundle = bundle
        self.pipeline = bundle.pipeline
        self.booster = bundle.booster()
        self.calibrator = bundle.calibrator
        self.thresholds = bundle.thresholds
        self.top_n = top_n
        self.explainer = ShapExplainer(self.booster)
        self._known_fields = set(self.pipeline.input_columns)

    @classmethod
    def load(cls, cfg: Config, path: Path | None = None) -> FraudScorer:
        bundle = load_bundle(path or latest_bundle_dir(cfg))
        return cls(bundle, top_n=cfg.service.top_n_reasons)

    @property
    def version(self) -> str:
        return self.bundle.version

    @property
    def known_fields(self) -> set[str]:
        return self._known_fields

    def unknown_fields(self, record: Mapping[str, Any]) -> list[str]:
        return sorted(set(record) - self._known_fields)

    # ------------------------------------------------------------------ batch
    def score_frame(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(raw, calibrated) scores for many rows. No explanations: fast path."""
        X = self.pipeline.transform(df)
        raw = np.asarray(self.booster.predict(X))
        return raw, self.calibrator.transform(raw)

    def decide_many(self, calibrated: np.ndarray) -> list[Decision]:
        return [code_to_decision(int(c)) for c in decide_codes(calibrated, self.thresholds)]

    # ------------------------------------------------------------------ single
    def score_one(self, record: Mapping[str, Any], explain: bool = True) -> ScoreResult:
        """Score one transaction given as a flat dict of fields.

        ``explain=False`` skips SHAP (the dominant cost: exact TreeSHAP over every tree),
        returning the score and decision only.
        """
        clean = {k: _none_if_nan(v) for k, v in record.items()}
        X = self.pipeline.transform_records([clean])
        raw = float(self.booster.predict(X)[0])
        calibrated = float(self.calibrator.transform(np.array([raw]))[0])
        names = list(X.columns)
        shap_map: dict[str, float] = {}
        reasons: list[Reason] = []
        if explain:
            shap_row = self.explainer.explain(X)[0]
            shap_map = {n: float(v) for n, v in zip(names, shap_row, strict=True)}
            values = {n: float(v) for n, v in zip(names, X.iloc[0].to_numpy(), strict=True)}
            reasons = top_reasons(shap_map, values, clean, self.top_n)
        decision = code_to_decision(int(decide_codes(np.array([calibrated]), self.thresholds)[0]))
        return ScoreResult(
            fraud_score=calibrated,
            raw_score=raw,
            decision=decision,
            thresholds=self.thresholds,
            top_reasons=reasons,
            shap_values=shap_map,
            base_value=self.explainer.base_value,
            model_version=self.version,
        )


def _none_if_nan(v: Any) -> Any:
    if isinstance(v, float) and math.isnan(v):
        return None
    if v is pd.NA:
        return None
    return v
