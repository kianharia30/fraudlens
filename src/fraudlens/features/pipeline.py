"""Model-input pipeline: column selection, stateless derived features and encodings.

``FeaturePipeline`` is fitted on the **training block only** and then applied unchanged
to validation, test, API requests and the demo app, so every consumer builds model
inputs in exactly the same way.

Missing values: left as NaN. LightGBM learns a default split direction for missing
values, and in IEEE-CIS missingness is itself informative (e.g. identity columns are
only present for some transaction channels), so imputing would destroy signal. Only the
logistic-regression baseline imputes (medians, inside its own sklearn pipeline).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraudlens.config import Config
from fraudlens.features.behaviour import UID_COL, build_key
from fraudlens.logging_utils import get_logger

logger = get_logger(__name__)

UNSEEN_CODE = -1.0  # ordinal code for a category never seen in training

STATELESS_FEATURES: tuple[str, ...] = ("amt_log", "amt_cents", "email_domain_mismatch")


def add_stateless_features(df: pd.DataFrame, amount_col: str) -> pd.DataFrame:
    """Features computed from the row alone (safe to recompute anywhere)."""
    out = pd.DataFrame(index=df.index)
    amount = pd.to_numeric(df[amount_col], errors="coerce").astype(np.float64)
    out["amt_log"] = np.log1p(amount).astype(np.float32)
    # Cents part of the amount: foreign-currency conversions produce unusual cents values.
    out["amt_cents"] = (np.round(amount * 100) % 100).astype(np.float32)
    if {"P_emaildomain", "R_emaildomain"} <= set(df.columns):
        p = df["P_emaildomain"].astype("string")
        r = df["R_emaildomain"].astype("string")
        both = p.notna() & r.notna()
        out["email_domain_mismatch"] = (p != r).astype("float32").where(both)
    else:
        out["email_domain_mismatch"] = np.float32(np.nan)
    return out


@dataclass
class FeaturePipeline:
    """Fitted transformation from processed rows to a float32 model matrix."""

    amount_col: str
    excluded_cols: list[str]
    max_missing_frac: float
    freq_encode_cols: list[str]
    max_ordinal_cardinality: int
    uid_cols: list[str]
    # --- learned in fit() ---
    numeric_cols: list[str] = field(default_factory=list)
    ordinal_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    freq_maps: dict[str, dict[Any, float]] = field(default_factory=dict)
    freq_numeric: dict[str, bool] = field(default_factory=dict)
    dropped_sparse: dict[str, float] = field(default_factory=dict)
    dropped_constant: list[str] = field(default_factory=list)
    dropped_high_cardinality: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Config) -> FeaturePipeline:
        d = cfg.data
        return cls(
            amount_col=d.amount_col,
            # TransactionDT is excluded: an absolute time index would let the model learn
            # the training period's trend, which does not transfer to the future.
            excluded_cols=[d.id_col, d.target_col, d.time_col, UID_COL],
            max_missing_frac=cfg.features.max_missing_frac,
            freq_encode_cols=list(cfg.features.freq_encode_cols),
            max_ordinal_cardinality=cfg.features.max_ordinal_cardinality,
            uid_cols=list(cfg.features.card_uid_cols),
        )

    def _with_uid(self, df: pd.DataFrame) -> pd.DataFrame:
        """(Re)build the proxy card UID from its parts, identically for training and serving."""
        if not set(self.uid_cols) <= set(df.columns):
            return df
        df = df.copy(deep=False)
        df[UID_COL] = build_key(df, self.uid_cols)[0]
        return df

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> FeaturePipeline:
        """Learn columns and encodings from the training block only."""
        train = self._with_uid(train)
        candidates = [c for c in train.columns if c not in self.excluded_cols]
        missing = train[candidates].isna().mean()
        self.dropped_sparse = {
            c: round(float(f), 4) for c, f in missing.items() if f > self.max_missing_frac
        }
        kept = [c for c in candidates if c not in self.dropped_sparse]
        self.dropped_constant = [c for c in kept if train[c].nunique(dropna=False) <= 1]
        kept = [c for c in kept if c not in self.dropped_constant]

        string_cols = [c for c in kept if _is_string(train[c])]
        self.numeric_cols = [c for c in kept if c not in string_cols]
        self.ordinal_maps, self.dropped_high_cardinality = {}, []
        for col in string_cols:
            counts = train[col].astype("string").value_counts()
            if len(counts) <= self.max_ordinal_cardinality:
                # Most frequent category -> 0, so codes are deterministic given train data.
                self.ordinal_maps[col] = {str(v): i for i, v in enumerate(counts.index)}
            else:
                self.dropped_high_cardinality.append(col)

        self.freq_maps, self.freq_numeric = {}, {}
        for col in self.freq_encode_cols:
            if col in train.columns:
                numeric = not _is_string(train[col])
                freq = _as_key(train[col], numeric).value_counts(normalize=True, dropna=True)
                self.freq_numeric[col] = numeric
                self.freq_maps[col] = {k: float(v) for k, v in freq.items()}

        self.feature_names = (
            self.numeric_cols
            + list(self.ordinal_maps)
            + [f"{c}_freq" for c in self.freq_maps]
            + list(STATELESS_FEATURES)
        )
        logger.info(
            "FeaturePipeline fitted: %d features (%d sparse, %d constant, %d high-cardinality dropped)",
            len(self.feature_names),
            len(self.dropped_sparse),
            len(self.dropped_constant),
            len(self.dropped_high_cardinality),
        )
        return self

    # ------------------------------------------------------------------ transform
    @property
    def input_columns(self) -> list[str]:
        """Raw/processed columns this pipeline reads (others are ignored)."""
        cols = dict.fromkeys(
            [
                *self.numeric_cols,
                *self.ordinal_maps,
                *self.freq_maps,
                self.amount_col,
                "P_emaildomain",
                "R_emaildomain",
                *self.uid_cols,
            ]
        )
        return list(cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the float32 model matrix. Unknown columns are ignored, missing ones are NaN."""
        if not self.feature_names:
            raise RuntimeError("FeaturePipeline must be fitted before transform")
        df = self._with_uid(df.reindex(columns=self.input_columns))
        parts: list[pd.DataFrame] = [_to_float32(df[self.numeric_cols])]

        ordinal: dict[str, pd.Series] = {}
        for col, mapping in self.ordinal_maps.items():
            mapped = df[col].astype("string").map(mapping).astype("float64")
            # Missing stays NaN; present-but-unseen-in-train gets a dedicated code.
            ordinal[col] = mapped.where(mapped.notna() | df[col].isna(), UNSEEN_CODE)
        parts.append(pd.DataFrame(ordinal, index=df.index))

        freq = {
            f"{col}_freq": _as_key(df[col], self.freq_numeric[col])
            .map(mapping)
            .astype("float64")
            .fillna(0.0)
            for col, mapping in self.freq_maps.items()
        }
        parts.append(pd.DataFrame(freq, index=df.index))
        parts.append(add_stateless_features(df, self.amount_col))

        matrix = pd.concat(parts, axis=1).astype(np.float32)
        return matrix[self.feature_names]

    def transform_records(self, records: list[Mapping[str, Any]]) -> pd.DataFrame:
        """Transform plain dicts (API payloads, app form values)."""
        return self.transform(pd.DataFrame.from_records(records))

    def report(self) -> dict[str, Any]:
        """Summary of what was dropped and why (written to docs/ at training time)."""
        return {
            "n_features": len(self.feature_names),
            "max_missing_frac": self.max_missing_frac,
            "dropped_sparse": self.dropped_sparse,
            "dropped_constant": self.dropped_constant,
            "dropped_high_cardinality_raw": self.dropped_high_cardinality,
            "frequency_encoded": list(self.freq_maps),
            "ordinal_encoded": list(self.ordinal_maps),
        }


def _is_string(s: pd.Series) -> bool:
    return (
        isinstance(s.dtype, pd.CategoricalDtype)
        or s.dtype == object
        or pd.api.types.is_string_dtype(s)
    )


def _as_key(s: pd.Series, numeric: bool) -> pd.Series:
    """Lookup key for frequency maps: float values for numeric columns, strings otherwise."""
    if numeric:
        return pd.to_numeric(s, errors="coerce").astype("float64")
    return s.astype("string")


def _to_float32(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.astype(np.float32)
    except (TypeError, ValueError):
        return df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
