"""Leakage-safe behavioural ("context") features.

IEEE-CIS has no customer or card ID. Following common practice on this dataset we build
a **proxy card UID** by concatenating card attributes (``card1, card2, card3, card5``)
with the billing region (``addr1``). This is an *assumption*: two people can share a UID
and one person can appear under several. Features built on it are therefore noisy
behaviour signals, not a true customer history.

Every feature here is a function of the current row and *earlier* rows only (rows are
processed in ``(TransactionDT, TransactionID)`` order). Changing or adding a later row
can never change an earlier row's features; ``tests/test_features_leakage.py`` checks this.
In production these would be served from a feature store keyed on the card; offline we
compute them in one causal pass over the full timeline.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fraudlens.config import Config

UID_COL = "card_uid"
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86_400

BEHAVIOUR_FEATURES: tuple[str, ...] = (
    "card_txn_count_prior",
    "amt_to_card_mean_ratio",
    "amt_to_card_median_ratio",
    "amt_card_zscore",
    "secs_since_prev_card_txn",
    "hour_of_day",
    "day_of_week",
    "hour_dev_from_card_usual",
)


def canonical_text(s: pd.Series) -> pd.Series:
    """String form that is identical whether a value arrived as 13926, 13926.0 or "13926"."""
    if isinstance(s.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(s):
        text = s.astype("string")
        num = pd.to_numeric(text, errors="coerce")
        # Numeric-looking strings (e.g. from JSON) are normalised like numbers.
        return canonical_text(num).where(num.notna(), text) if num.notna().any() else text
    num = pd.to_numeric(s, errors="coerce").astype("float64")
    integral = num.notna() & (num % 1 == 0)
    as_int = num.where(integral).round().astype("Int64").astype("string")
    return as_int.where(integral, num.astype("string"))


def build_key(df: pd.DataFrame, cols: Sequence[str]) -> tuple[pd.Series, np.ndarray]:
    """Return (readable string key, dense integer codes) for a column combination.

    Missing values are kept as the literal ``nan`` so cards with a missing attribute still
    get a (shared) key, rather than being dropped from history features.
    """
    parts = [canonical_text(df[c]).fillna("nan") for c in cols]
    key = parts[0].str.cat(parts[1:], sep="_") if len(parts) > 1 else parts[0]
    codes, _ = pd.factorize(key)
    return key, codes.astype(np.int64)


def _prior_expanding_stats(amount: pd.Series, codes: np.ndarray) -> pd.DataFrame:
    """Mean, std and median of each card's amounts strictly *before* the current row."""
    amt = amount.astype(np.float64)
    grp = amt.groupby(codes, sort=False)
    count_prior = grp.cumcount().astype(np.float64)
    sum_prior = grp.cumsum() - amt
    sq_prior = (amt**2).groupby(codes, sort=False).cumsum() - amt**2

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_prior = (sum_prior / count_prior).where(count_prior > 0)
        var_prior = ((sq_prior - count_prior * mean_prior**2) / (count_prior - 1)).where(
            count_prior > 1
        )
    std_prior = np.sqrt(var_prior.clip(lower=0))

    # Expanding median including the current row, then shifted by one within the card
    # -> median of strictly earlier rows. groupby().expanding() runs in compiled code.
    median_incl = grp.expanding().median().reset_index(level=0, drop=True).sort_index()
    median_prior = median_incl.groupby(codes, sort=False).shift(1)
    return pd.DataFrame(
        {"count": count_prior, "mean": mean_prior, "std": std_prior, "median": median_prior}
    )


def window_counts(times: np.ndarray, codes: np.ndarray, window: int) -> np.ndarray:
    """Number of the same card's transactions in ``[t - window, t)`` for every row.

    Vectorised: encode (card, time) as one sortable int64, then two binary searches per row.
    The interval is open at ``t`` so simultaneous transactions never count each other.
    """
    t = times.astype(np.int64)
    t = t - t.min()
    offset = int(t.max()) + window + 1  # keeps each card's (shifted) range disjoint
    combined = codes * offset + t
    sorted_combined = np.sort(combined)
    upper = np.searchsorted(sorted_combined, combined, side="left")
    lower = np.searchsorted(sorted_combined, combined - window, side="left")
    return (upper - lower).astype(np.float32)


def _hour_features(times: pd.Series, codes: np.ndarray) -> pd.DataFrame:
    """Hour of day, day of week and circular distance from the card's usual hour.

    TransactionDT's origin is undisclosed, so "hour 0" is relative to that origin rather
    than a confirmed local midnight. The *daily* periodicity is still real.
    """
    hour = ((times // SECONDS_PER_HOUR) % 24).astype(np.float64)
    dow = ((times // SECONDS_PER_DAY) % 7).astype(np.float32)
    angle = 2 * np.pi * hour / 24
    sin_prior = np.sin(angle).groupby(codes, sort=False).cumsum() - np.sin(angle)
    cos_prior = np.cos(angle).groupby(codes, sort=False).cumsum() - np.cos(angle)
    has_prior = pd.Series(codes).groupby(codes, sort=False).cumcount().to_numpy() > 0
    usual_hour = (np.arctan2(sin_prior, cos_prior) * 24 / (2 * np.pi)) % 24
    diff = np.abs(hour - usual_hour)
    dev = np.minimum(diff, 24 - diff).where(has_prior)
    return pd.DataFrame(
        {
            "hour_of_day": hour.astype(np.float32),
            "day_of_week": dow,
            "hour_dev_from_card_usual": dev.astype(np.float32),
        }
    )


def novelty_flag(values: pd.Series, codes: np.ndarray, has_prior: np.ndarray) -> np.ndarray:
    """1.0 if this value was never seen earlier for the card, 0.0 if it was.

    NaN when the value is missing or the card has no earlier transaction (no history to
    compare against).
    """
    value_codes, uniques = pd.factorize(values)
    pair = codes * (len(uniques) + 1) + (value_codes + 1)
    first_seen = pd.Series(pair).groupby(pair, sort=False).cumcount().to_numpy() == 0
    out = first_seen.astype(np.float32)
    out[(value_codes < 0) | ~has_prior] = np.nan
    return out


def add_behaviour_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add the card UID and all past-only behaviour features to a time-sorted frame.

    The frame must already be sorted by ``(time_col, id_col)``; see ``sort_by_time``.
    """
    fcfg, dcfg = cfg.features, cfg.data
    times = df[dcfg.time_col]
    if np.any(np.diff(times.to_numpy()) < 0):
        raise ValueError("add_behaviour_features expects a frame sorted by time")

    uid, codes = build_key(df, fcfg.card_uid_cols)
    _, card_codes = build_key(df, fcfg.card_key_cols)
    amount = df[dcfg.amount_col]
    stats = _prior_expanding_stats(amount, codes)
    has_prior = stats["count"].to_numpy() > 0

    feats: dict[str, pd.Series | np.ndarray] = {
        UID_COL: uid.astype("category"),
        "card_txn_count_prior": stats["count"].astype(np.float32),
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["amt_to_card_mean_ratio"] = (amount / stats["mean"]).astype(np.float32)
        feats["amt_to_card_median_ratio"] = (amount / stats["median"]).astype(np.float32)
        feats["amt_card_zscore"] = (
            ((amount - stats["mean"]) / stats["std"]).replace([np.inf, -np.inf], np.nan)
        ).astype(np.float32)
    feats["secs_since_prev_card_txn"] = times.groupby(codes, sort=False).diff().astype(np.float32)
    for label, seconds in fcfg.velocity_windows_seconds.items():
        feats[f"card_txn_count_{label}"] = window_counts(times.to_numpy(), codes, seconds)
    for col in fcfg.novelty_cols:
        if col in df.columns:
            feats[f"is_new_{col}_for_card"] = novelty_flag(df[col], codes, has_prior)
    # addr1 is part of the UID, so "new address" uses the coarser card-only key.
    if "addr1" in df.columns:
        card_has_prior = pd.Series(card_codes).groupby(card_codes).cumcount().to_numpy() > 0
        feats["is_new_addr1_for_card"] = novelty_flag(df["addr1"], card_codes, card_has_prior)

    hour_feats = _hour_features(times, codes)
    out = pd.concat(
        [df, pd.DataFrame(feats, index=df.index), hour_feats.set_index(df.index)], axis=1
    )
    return out


def behaviour_feature_names(cfg: Config) -> list[str]:
    """All behaviour feature names produced by ``add_behaviour_features`` for this config."""
    names = list(BEHAVIOUR_FEATURES)
    names += [f"card_txn_count_{label}" for label in cfg.features.velocity_windows_seconds]
    names += [f"is_new_{col}_for_card" for col in cfg.features.novelty_cols]
    names.append("is_new_addr1_for_card")
    return names
