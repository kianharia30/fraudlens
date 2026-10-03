"""Streamlit rendering helpers. Presentation only: no scoring logic lives here."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from fraudlens import style
from fraudlens.explain.reasons import humanise_seconds
from fraudlens.models.policy import Thresholds
from fraudlens.scoring import ScoreResult

RAISES = "#e34948"  # diverging pole: pushes towards fraud
LOWERS = style.SERIES_1  # diverging pole: pushes towards genuine

CSS = f"""
<style>
.fl-card {{ border: 1px solid {style.GRID}; border-radius: 10px; padding: 14px 16px;
           background: #ffffff; margin-bottom: 10px; }}
.fl-card h4 {{ margin: 0 0 8px 0; font-size: 0.95rem; color: {style.TEXT_SECONDARY};
              text-transform: uppercase; letter-spacing: .04em; }}
.fl-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px 16px; }}
.fl-kv .k {{ font-size: .78rem; color: {style.TEXT_MUTED}; }}
.fl-kv .v {{ font-size: 1.02rem; color: {style.TEXT_PRIMARY}; font-weight: 600; word-break: break-word; }}
.fl-badge {{ display: inline-block; padding: 10px 22px; border-radius: 8px; color: #ffffff;
            font-weight: 700; font-size: 1.35rem; letter-spacing: .05em; }}
.fl-note {{ font-size: .85rem; color: {style.TEXT_SECONDARY}; }}
.fl-reason {{ margin: 4px 0; font-size: .97rem; }}
.fl-row {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(120px, 34%) 92px;
           gap: 14px; align-items: center; padding: 9px 0; border-top: 1px solid {style.GRID}; }}
.fl-row:first-child {{ border-top: none; }}
.fl-row-head {{ font-size: .78rem; color: {style.TEXT_MUTED}; padding-top: 0; }}
.fl-row-text {{ font-size: .97rem; color: {style.TEXT_PRIMARY}; line-height: 1.35; }}
.fl-row-val {{ font-size: .9rem; color: {style.TEXT_SECONDARY}; text-align: right;
              font-variant-numeric: tabular-nums; }}
.fl-track {{ position: relative; height: 14px; }}
.fl-mid {{ position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px; background: {style.AXIS}; }}
.fl-bar {{ position: absolute; top: 0; height: 14px; border-radius: 3px; }}
.fl-track-label {{ display: flex; justify-content: space-between; }}
@media (max-width: 640px) {{ .fl-row {{ grid-template-columns: 1fr 70px 80px; gap: 8px; }} }}
.fl-summary {{ font-size: 1rem; line-height: 1.6; margin: 2px 0 10px 0; color: {style.TEXT_PRIMARY}; }}
.fl-truth {{ border-radius: 10px; padding: 14px 16px; font-size: 1.05rem; font-weight: 600; }}
</style>
"""


def chart(fig: go.Figure, slot: Any = None, key: str | None = None) -> None:
    """Render a Plotly figure without the floating toolbar (it overlaps small charts)."""
    target = slot if slot is not None else st
    target.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _fmt(value: Any, kind: str = "text") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if kind == "money":
        return f"£{float(value):,.2f}"
    if kind == "int":
        return f"{float(value):,.0f}"
    if kind == "ratio":
        return f"{float(value):.1f}x"
    if kind == "flag":
        return "Yes" if float(value) >= 0.5 else "No"
    if kind == "duration":
        return humanise_seconds(float(value))
    if kind == "float":
        return f"{float(value):,.1f}"
    return str(value)


def kv_card(title: str, items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="fl-kv"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>'
        for k, v in items
    )
    st.markdown(
        f'<div class="fl-card"><h4>{html.escape(title)}</h4><div class="fl-grid">{cells}</div></div>',
        unsafe_allow_html=True,
    )


def raw_input_card(row: Mapping[str, Any]) -> None:
    dt = row.get("TransactionDT")
    when = (
        "—" if dt is None else f"Day {int(dt) // 86_400}, {int(dt) // 3600 % 24:02d}:00 (relative)"
    )
    kv_card(
        "1 · Raw transaction",
        [
            ("Amount", _fmt(row.get("TransactionAmt"), "money")),
            ("Product", _fmt(row.get("ProductCD"))),
            ("Card", f"{_fmt(row.get('card4'))} / {_fmt(row.get('card6'))}"),
            ("Purchaser email", _fmt(row.get("P_emaildomain"))),
            ("Recipient email", _fmt(row.get("R_emaildomain"))),
            ("Device", f"{_fmt(row.get('DeviceType'))} · {_fmt(row.get('DeviceInfo'))}"),
            (
                "Billing region / country",
                f"{_fmt(row.get('addr1'), 'int')} / {_fmt(row.get('addr2'), 'int')}",
            ),
            ("Address distance", _fmt(row.get("dist1"), "float")),
            ("Time", when),
        ],
    )


def context_card(row: Mapping[str, Any]) -> None:
    kv_card(
        "2 · Engineered context (from this card's past only)",
        [
            ("Amount vs card average", _fmt(row.get("amt_to_card_mean_ratio"), "ratio")),
            ("Amount vs card median", _fmt(row.get("amt_to_card_median_ratio"), "ratio")),
            ("Card txns, last hour", _fmt(row.get("card_txn_count_1h"), "int")),
            ("Card txns, last 24h", _fmt(row.get("card_txn_count_24h"), "int")),
            ("Earlier txns on card", _fmt(row.get("card_txn_count_prior"), "int")),
            ("Since previous txn", _fmt(row.get("secs_since_prev_card_txn"), "duration")),
            ("New device?", _fmt(row.get("is_new_DeviceInfo_for_card"), "flag")),
            ("New email domain?", _fmt(row.get("is_new_P_emaildomain_for_card"), "flag")),
            ("Hours from usual time", _fmt(row.get("hour_dev_from_card_usual"), "float")),
        ],
    )
    st.caption(
        "“—” means no earlier history for this proxy card, so the feature is missing (the model handles that)."
    )


def gauge(score: float, thresholds: Thresholds, title: str = "Fraud score") -> go.Figure:
    review = min(thresholds.review, 1.0)
    block = min(thresholds.block, 1.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"color": style.TEXT_PRIMARY}},
            title={"text": title, "font": {"size": 15, "color": style.TEXT_SECONDARY}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": style.AXIS, "ticksuffix": "%"},
                "bar": {"color": style.TEXT_PRIMARY, "thickness": 0.28},
                "steps": [
                    {"range": [0, review * 100], "color": "#d9f2d9"},
                    {"range": [review * 100, block * 100], "color": "#fdedc4"},
                    {"range": [block * 100, 100], "color": "#f6d3d3"},
                ],
                "bgcolor": style.SURFACE,
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(
        height=230, margin={"l": 30, "r": 40, "t": 40, "b": 0}, paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def decision_badge(decision: str, score: float, thresholds: Thresholds) -> None:
    color = style.DECISION_COLORS[decision]
    icon = style.DECISION_ICONS[decision]
    text_color = style.TEXT_PRIMARY if decision == "REVIEW" else "#ffffff"
    rule = {
        "APPROVE": f"score {score:.3f} < review threshold {thresholds.review:.3f}",
        "REVIEW": f"review {thresholds.review:.3f} ≤ score {score:.3f} < block {thresholds.block:.3f}",
        "BLOCK": f"score {score:.3f} ≥ block threshold {thresholds.block:.3f}",
    }[decision]
    st.markdown(
        f'<span class="fl-badge" style="background:{color};color:{text_color}">{icon} {decision}</span>'
        f'<p class="fl-note" style="margin-top:8px">{html.escape(rule)}</p>',
        unsafe_allow_html=True,
    )


def impact_label(shap_value: float) -> str:
    """Plain-English strength of one reason (thresholds in log-odds)."""
    size = abs(shap_value)
    return "Strong" if size >= 1.0 else "Moderate" if size >= 0.4 else "Slight"


def _summary(result: ScoreResult) -> str:
    up = [r for r in result.top_reasons if r.shap_value > 0]
    down = [r for r in result.top_reasons if r.shap_value < 0]
    parts = []
    if up:
        parts.append(f"<b style='color:{RAISES}'>Biggest red flag:</b> {html.escape(up[0].text)}")
    if down:
        parts.append(f"<b style='color:{LOWERS}'>Most reassuring:</b> {html.escape(down[0].text)}")
    return "<br>".join(parts)


def reasons_panel(result: ScoreResult) -> None:
    """The top reasons in plain English, each with a bar showing direction and strength.

    Plain HTML rather than a chart, so long reason texts wrap instead of being truncated.
    """
    if not result.top_reasons:
        st.info("No explanation available for this score.")
        return
    span = max(abs(r.shap_value) for r in result.top_reasons) or 1.0
    rows = []
    for r in result.top_reasons:
        up = r.shap_value > 0
        color = RAISES if up else LOWERS
        effect = "Raises risk" if up else "Lowers risk"
        width = 50 * abs(r.shap_value) / span  # % of the track; each half is 50%
        side = "left:50%" if up else f"left:{50 - width:.1f}%"
        rows.append(
            '<div class="fl-row">'
            f'<div class="fl-row-text">{html.escape(r.text)}</div>'
            '<div class="fl-track"><div class="fl-mid"></div>'
            f'<div class="fl-bar" style="{side};width:{width:.1f}%;background:{color}"></div></div>'
            f'<div class="fl-row-val" style="color:{color}" '
            f'title="SHAP {r.shap_value:+.2f} log-odds">{impact_label(r.shap_value)}<br>'
            f'<span class="fl-note">{effect}</span></div>'
            "</div>"
        )
    header = (
        '<div class="fl-row fl-row-head"><div>What the model noticed</div>'
        '<div class="fl-track-label"><span>◀ makes fraud less likely</span>'
        "<span>more likely ▶</span></div><div>Effect</div></div>"
    )
    st.markdown(
        f'<div class="fl-summary">{_summary(result)}</div>'
        f'<div class="fl-card">{header}{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Longer bar = bigger influence on this transaction's score. These are the five "
        "strongest factors; the rest had smaller effects."
    )
    with st.expander("How are these reasons calculated?"):
        st.markdown(
            "- Each reason comes from **SHAP**, a standard method that splits the model's output "
            "into how much each input pushed it towards or away from fraud, for *this* "
            "transaction.\n"
            "- **Strong / Moderate / Slight** = a push of at least 1.0 / 0.4 / under 0.4 on the "
            "model's log-odds scale (hover a label for the exact value). Related inputs, "
            "such as the amount compared with the card's average and median, are combined "
            "into one reason.\n"
            "- **Anonymised** reasons come from dataset columns whose meaning the data provider "
            "kept confidential. We group them under the provider's own short description "
            "rather than guess what they mean.\n"
            "- The score shown on the gauge is the model output after calibration, so it reads "
            "as a fraud probability. Calibration never reverses a reason's direction."
        )


def truth_banner(is_fraud: bool, decision: str) -> None:
    if is_fraud:
        state = "ok" if decision != "APPROVE" else "bad"
        msg = {
            "BLOCK": "Actual label: FRAUD, so the model was correct (blocked).",
            "REVIEW": "Actual label: FRAUD, so the model was correct (sent to an analyst).",
            "APPROVE": "Actual label: FRAUD, so the model missed it.",
        }[decision]
    else:
        state = {"APPROVE": "ok", "REVIEW": "minor", "BLOCK": "bad"}[decision]
        msg = {
            "APPROVE": "Actual label: GENUINE, so the model was correct (approved).",
            "REVIEW": "Actual label: GENUINE: sent to review unnecessarily (small review cost; the customer is not blocked).",
            "BLOCK": "Actual label: GENUINE, so the model raised a false alarm (customer blocked).",
        }[decision]
    bg, fg, icon = {
        "ok": ("#e3f5e3", "#0b5d0b", "✓"),
        "minor": ("#fdf1d2", "#6b4a00", "⚑"),
        "bad": ("#fbe3e3", "#8f1f1f", "✕"),
    }[state]
    st.markdown(
        f'<div class="fl-truth" style="background:{bg};color:{fg}">{icon} {html.escape(msg)}</div>',
        unsafe_allow_html=True,
    )
