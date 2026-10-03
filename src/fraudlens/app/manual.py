"""Manual mode: edit key fields of a real template transaction and watch the score change."""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from fraudlens.app import components as ui
from fraudlens.app.logic import (
    MANUAL_FIELDS,
    FormField,
    apply_form,
    choice_options,
    form_value_from_record,
    sample_transaction,
    to_record,
    typical_row,
)
from fraudlens.app.resources import Resources

MISSING = "(missing)"


def _key(field: FormField) -> str:
    return f"manual_{field.name}"


def _load_template(record: dict[str, Any], source: str) -> None:
    """Set the template and reset every widget to its values (runs before widgets render)."""
    st.session_state.manual_template = record
    st.session_state.manual_source = source
    for field in MANUAL_FIELDS:
        value = form_value_from_record(field, record)
        st.session_state[_key(field)] = (
            MISSING if (field.kind == "choice" and value is None) else value
        )
    st.session_state.manual_prev = None
    st.session_state.manual_last = None


def _random_template(res: Resources) -> None:
    rng = np.random.default_rng()
    row = sample_transaction(
        res.pool, "random", res.scorer.thresholds, rng, res.cfg.app.demo_fraud_weight
    )
    if row is not None:
        _load_template(
            to_record(row, res.scorer.known_fields),
            f"random held-out transaction {int(row['TransactionID'])}",
        )


def _widget(field: FormField, res: Resources) -> Any:
    key = _key(field)
    if field.kind == "choice":
        options = [MISSING, *choice_options(field, res.pool)]
        current = st.session_state.get(key)
        if current not in options:
            options.append(str(current))
        value = st.selectbox(field.label, options, key=key, help=field.help or None)
        return None if value == MISSING else value
    return st.number_input(
        field.label,
        min_value=field.min_value,
        max_value=field.max_value,
        step=field.step,
        key=key,
        help=field.help or None,
        format="%d" if field.kind == "int" else "%.2f",
        placeholder="missing",
    )


def render_manual(res: Resources) -> None:
    if "manual_template" not in st.session_state:
        row = typical_row(res.pool)
        _load_template(
            to_record(row, res.scorer.known_fields), "typical genuine transaction (median score)"
        )

    st.markdown(
        "Edit the key fields below. Every other model input is taken from a **real template "
        f"transaction**: currently a {st.session_state.manual_source}."
    )
    st.button("🎲 Start from a random transaction", on_click=_random_template, args=(res,))

    with st.form("manual_form"):
        cols = st.columns(3)
        values: dict[str, Any] = {}
        for i, field in enumerate(MANUAL_FIELDS):
            with cols[i % 3]:
                values[field.name] = _widget(field, res)
        submitted = st.form_submit_button(
            "Score transaction", type="primary", use_container_width=True
        )

    if submitted:
        record = apply_form(st.session_state.manual_template, values)
        st.session_state.manual_prev = st.session_state.get("manual_last")
        st.session_state.manual_last = res.scorer.score_one(record)

    result = st.session_state.get("manual_last")
    if result is None:
        st.info("Adjust the fields if you like, then press **Score transaction**.")
        return

    prev = st.session_state.get("manual_prev")
    c1, c2, c3 = st.columns([1, 1, 1.3])
    with c1:
        ui.chart(ui.gauge(result.fraud_score, result.thresholds, "New score"))
    with c2:
        if prev is not None:
            ui.chart(ui.gauge(prev.fraud_score, prev.thresholds, "Previous score"))
        else:
            st.markdown("**Previous score**")
            st.caption("Score again after changing a field to compare.")
    with c3:
        delta = (
            None if prev is None else f"{(result.fraud_score - prev.fraud_score) * 100:+.1f} pts"
        )
        st.metric("Fraud score", f"{result.fraud_score:.1%}", delta=delta, delta_color="inverse")
        ui.decision_badge(result.decision.value, result.fraud_score, result.thresholds)
        if prev is not None and prev.decision != result.decision:
            st.caption(f"Decision changed: {prev.decision.value} → {result.decision.value}")

    st.markdown("**Why this score?**")
    ui.reasons_panel(result)
    st.caption(
        "What-if caveat: edited fields are independent inputs. Changing the amount does not "
        "recompute 'amount vs card average'; real card history would."
    )
