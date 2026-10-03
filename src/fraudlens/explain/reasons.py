"""Turn per-feature SHAP values into short, honest, human-readable reasons.

Related features are grouped before ranking (e.g. ``TransactionAmt`` + ``amt_log``,
or ``P_emaildomain`` + its frequency encoding), so one real-world fact gives one reason.

Most IEEE-CIS columns are anonymised by the data provider (Vesta): the V, C, D, M and
most id_ columns have no published meaning. We do **not** invent meanings for them;
they are summed into labelled "anonymised signal" groups. Exceptions are id_30, id_31
and id_33, whose values are self-evidently operating system, browser and screen
resolution strings.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

Values = Mapping[str, float]
Raw = Mapping[str, Any]


@dataclass(frozen=True)
class Reason:
    group: str
    text: str
    shap_value: float  # summed log-odds contribution of the group
    direction: str  # "raises risk" | "lowers risk"
    features: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"features": list(self.features)}


_ANONYMISED: tuple[tuple[re.Pattern[str], str, str], ...] = (
    # Labels follow Kaggle's published one-line description of each column family.
    (re.compile(r"^V\d+(_freq)?$"), "anon_V", "Anonymised risk scores from the payment provider"),
    (re.compile(r"^C\d+(_freq)?$"), "anon_C", "Anonymised counts linked to this card"),
    (re.compile(r"^D\d+(_freq)?$"), "anon_D", "Anonymised time-between-events signals"),
    (re.compile(r"^M\d+(_freq)?$"), "anon_M", "Anonymised match checks (e.g. name vs address)"),
    (re.compile(r"^id_\d+(_freq)?$"), "anon_id", "Anonymised device and connection signals"),
)
_NAMED_ID = {"id_30": "Operating system", "id_31": "Browser", "id_33": "Screen resolution"}
_FIXED_GROUPS = {
    "TransactionAmt": "amount",
    "amt_log": "amount",
    **{c: "card_profile" for c in ("card1", "card2", "card3", "card5")},
    **{f"{c}_freq": "card_profile" for c in ("card1", "card2", "card3", "card5")},
    "addr1": "billing_address",
    "addr2": "billing_address",
    "addr1_freq": "billing_address",
    "amt_to_card_mean_ratio": "amount_vs_card",
    "amt_to_card_median_ratio": "amount_vs_card",
    "amt_card_zscore": "amount_vs_card",
}


def group_of(feature: str) -> str:
    """Group key for a feature name."""
    if feature in _FIXED_GROUPS:
        return _FIXED_GROUPS[feature]
    base = feature.removesuffix("_freq")
    if base in _NAMED_ID or base in _TEMPLATES:
        return base
    for pattern, key, _ in _ANONYMISED:
        if pattern.match(feature):
            return key
    return base


# ---------------------------------------------------------------- text helpers
def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _raw_text(raw: Raw, col: str) -> str:
    v = raw.get(col)
    return "missing" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def humanise_seconds(seconds: float) -> str:
    for unit, size in (("day", 86_400), ("hour", 3_600), ("minute", 60)):
        if seconds >= size:
            n = seconds / size
            return f"{n:.0f} {unit}{'s' if round(n) != 1 else ''}"
    return f"{seconds:.0f} seconds"


def _novelty(what: str) -> Callable[[Values, Raw, str], str]:
    def text(values: Values, raw: Raw, feature: str) -> str:
        v = _num(values.get(feature))
        if v is None:
            return f"No card history to check the {what} against"
        return (
            f"{what.capitalize()} {'not seen before' if v >= 0.5 else 'already used'} on this card"
        )

    return text


def _count(window: str) -> Callable[[Values, Raw, str], str]:
    def text(values: Values, raw: Raw, feature: str) -> str:
        v = _num(values.get(feature)) or 0.0
        return (
            f"{v:.0f} earlier transaction{'s' if v != 1 else ''} on this card in the past {window}"
        )

    return text


def _amount_vs_card(values: Values, raw: Raw, feature: str) -> str:
    mean = _num(values.get("amt_to_card_mean_ratio"))
    median = _num(values.get("amt_to_card_median_ratio"))
    if mean is None and median is None:
        return "No earlier transactions on this card to compare the amount with"
    parts = [
        f"{mean:.1f}x this card's average" if mean is not None else None,
        f"{median:.1f}x its typical (median) amount" if median is not None else None,
    ]
    return "Amount is " + " and ".join(p for p in parts if p)


def _amount(values: Values, raw: Raw, feature: str) -> str:
    v = _num(raw.get("TransactionAmt"))
    return "Transaction amount missing" if v is None else f"Transaction amount £{v:,.2f}"


def _cents(values: Values, raw: Raw, feature: str) -> str:
    v = _num(values.get(feature))
    return (
        "Amount cents unknown"
        if v is None
        else f"Amount ends in .{int(v):02d} (unusual cents can indicate currency conversion)"
    )


def _prior_count(values: Values, raw: Raw, feature: str) -> str:
    v = _num(values.get(feature)) or 0.0
    return (
        "First transaction seen for this card"
        if v == 0
        else f"{v:.0f} earlier transactions seen for this card"
    )


def _since_prev(values: Values, raw: Raw, feature: str) -> str:
    v = _num(values.get(feature))
    return (
        "No previous transaction on this card"
        if v is None
        else f"{humanise_seconds(v)} since this card's previous transaction"
    )


def _hour(values: Values, raw: Raw, feature: str) -> str:
    v = _num(values.get(feature))
    return (
        "Time of day unknown"
        if v is None
        else f"Made at hour {v:.0f} of the dataset's (relative) day"
    )


def _hour_dev(values: Values, raw: Raw, feature: str) -> str:
    v = _num(values.get(feature))
    return (
        "No card history for its usual time of day"
        if v is None
        else f"{v:.1f} hours away from this card's usual time of day"
    )


def _mismatch(values: Values, raw: Raw, feature: str) -> str:
    v = _num(values.get(feature))
    if v is None:
        return "Recipient email domain not provided"
    return (
        "Purchaser and recipient email domains differ"
        if v >= 0.5
        else "Purchaser and recipient email domains match"
    )


def _categorical(label: str, col: str) -> Callable[[Values, Raw, str], str]:
    def text(values: Values, raw: Raw, feature: str) -> str:
        return f"{label}: {_raw_text(raw, col)}"

    return text


def _card_uid_freq(values: Values, raw: Raw, feature: str) -> str:
    v = _num(values.get(feature)) or 0.0
    return (
        "This card has never been seen before"
        if v == 0
        else "How often this card has been seen before"
    )


def _distance(values: Values, raw: Raw, feature: str) -> str:
    v = _num(raw.get(feature, values.get(feature)))
    return (
        f"Address distance ({feature}) not available"
        if v is None
        else f"Address distance ({feature}) of {v:.0f}"
    )


def _group_text(label: str) -> Callable[[Values, Raw, str], str]:
    def text(values: Values, raw: Raw, feature: str) -> str:
        return label

    return text


_TEMPLATES: dict[str, Callable[[Values, Raw, str], str]] = {
    "amount": _amount,
    "amt_cents": _cents,
    "amount_vs_card": _amount_vs_card,
    "card_txn_count_prior": _prior_count,
    "card_txn_count_1h": _count("hour"),
    "card_txn_count_24h": _count("24 hours"),
    "secs_since_prev_card_txn": _since_prev,
    "hour_of_day": _hour,
    "day_of_week": _group_text("Day of the week (relative to the dataset start)"),
    "hour_dev_from_card_usual": _hour_dev,
    "is_new_P_emaildomain_for_card": _novelty("purchaser email domain"),
    "is_new_R_emaildomain_for_card": _novelty("recipient email domain"),
    "is_new_DeviceInfo_for_card": _novelty("device"),
    "is_new_addr1_for_card": _novelty("billing region"),
    "email_domain_mismatch": _mismatch,
    "ProductCD": _categorical("Product category", "ProductCD"),
    "card4": _categorical("Card network", "card4"),
    "card6": _categorical("Card type", "card6"),
    "P_emaildomain": _categorical("Purchaser email domain", "P_emaildomain"),
    "R_emaildomain": _categorical("Recipient email domain", "R_emaildomain"),
    "DeviceType": _categorical("Device type", "DeviceType"),
    "DeviceInfo": _categorical("Device", "DeviceInfo"),
    "card_uid": _card_uid_freq,
    "card_profile": _group_text("Card details (anonymised issuer and card codes)"),
    "billing_address": _group_text("Billing region and country (anonymised codes)"),
    "dist1": _distance,
    "dist2": _distance,
    **{k: _categorical(v, k) for k, v in _NAMED_ID.items()},
    **{key: _group_text(label) for _, key, label in _ANONYMISED},
}


def describe_group(group: str, members: list[str], values: Values, raw: Raw) -> str:
    template = _TEMPLATES.get(group)
    if template is None:
        feature = members[0]
        return f"Feature {feature} = {values.get(feature)}"
    # Single-feature templates read the member feature (e.g. "card_uid" reads "card_uid_freq").
    feature = group if group in values else members[0]
    text = template(values, raw, feature)
    if group.startswith("anon_") and len(members) > 1:
        text += f" ({len(members)} columns combined)"
    return text


def top_reasons(
    shap_values: Values, feature_values: Values, raw: Raw, top_n: int = 5
) -> list[Reason]:
    """Rank grouped SHAP contributions by magnitude and describe the top ``top_n``."""
    groups: dict[str, list[str]] = {}
    for feature in shap_values:
        groups.setdefault(group_of(feature), []).append(feature)
    totals = {g: float(sum(shap_values[f] for f in fs)) for g, fs in groups.items()}
    ranked = sorted(totals, key=lambda g: abs(totals[g]), reverse=True)[:top_n]
    return [
        Reason(
            group=g,
            text=describe_group(g, groups[g], feature_values, raw),
            shap_value=totals[g],
            direction="raises risk" if totals[g] > 0 else "lowers risk",
            features=tuple(groups[g]),
        )
        for g in ranked
        if totals[g] != 0.0
    ]
