"""Simulation mode: sample held-out transactions and walk through the decision step by step."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from fraudlens.app import components as ui
from fraudlens.app.logic import SampleKind, history_entry, sample_transaction, tally, to_record
from fraudlens.app.resources import Resources
from fraudlens.scoring import ScoreResult

BUTTONS: tuple[tuple[SampleKind, str, str], ...] = (
    ("random", "🎲 Random transaction", "~40% fraud / 60% genuine"),
    ("suspicious", "🚩 Random suspicious", "score at or above the review threshold"),
    ("normal", "🟢 Random normal", "score below the review threshold"),
    ("borderline", "⚖️ Borderline case", "between the review and block thresholds"),
)
EMPTY_MESSAGES = {
    "borderline": "No held-out transactions fall between the review and block thresholds.",
}


def _rng() -> np.random.Generator:
    if "sim_rng" not in st.session_state:
        st.session_state.sim_rng = np.random.default_rng()
    rng: np.random.Generator = st.session_state.sim_rng
    return rng


def _pause(seconds: float, animate: bool) -> None:
    if animate:
        time.sleep(seconds)


def _render(row: pd.Series, result: ScoreResult, res: Resources, animate: bool) -> None:
    step = res.cfg.app.animation_step_seconds
    left, right = st.columns([1.15, 1])
    with left:
        ui.raw_input_card(row.to_dict())
        _pause(step, animate)
        ui.context_card(row.to_dict())
        _pause(step, animate)
    with right:
        st.markdown("**3 · Model score**")
        gauge_slot = st.empty()
        if animate:
            for i, frac in enumerate(np.linspace(0.1, 0.9, 8)):
                fig = ui.gauge(result.fraud_score * frac, result.thresholds)
                ui.chart(fig, gauge_slot, key=f"gauge_frame_{i}")
                time.sleep(step / 8)
        ui.chart(ui.gauge(result.fraud_score, result.thresholds), gauge_slot, key="gauge_final")
        st.markdown("**4 · Threshold decision**")
        ui.decision_badge(result.decision.value, result.fraud_score, result.thresholds)
        _pause(step, animate)

    st.markdown("**5 · Why this score?**")
    ui.reasons_panel(result)
    _pause(step, animate)
    st.markdown("**6 · Ground truth**")
    ui.truth_banner(bool(row["isFraud"] == 1), result.decision.value)


def _history_section(res: Resources) -> None:
    entries = st.session_state.get("sim_history", [])
    st.divider()
    st.subheader("Session history")
    if not entries:
        st.info("Generate a transaction to start the tally.")
        return
    t = tally(entries)
    c = st.columns(4)
    c[0].metric("Transactions", f"{t['n']:.0f}")
    c[1].metric("Correct calls", f"{t['correct']:.0f}")
    c[2].metric("Incorrect calls", f"{t['incorrect']:.0f}")
    c[3].metric("£ saved vs do-nothing", f"£{t['saved']:,.2f}")
    st.dataframe(
        pd.DataFrame([e.as_row() for e in reversed(entries)]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Correct = fraud reviewed or blocked; genuine approved or reviewed. "
        "'£ vs do-nothing' compares this decision's cost with approving everything."
    )
    if st.button("Clear history"):
        st.session_state.sim_history = []
        st.session_state.sim_current = None
        st.rerun()


def render_simulation(res: Resources) -> None:
    st.markdown(
        "Sample a real transaction from the **held-out test pool** and watch the model decide."
    )
    skip = st.checkbox("Skip animation", value=False)
    cols = st.columns(len(BUTTONS))
    clicked: SampleKind | None = None
    for col, (kind, label, tip) in zip(cols, BUTTONS, strict=True):
        if col.button(label, help=tip, use_container_width=True):
            clicked = kind

    if clicked is not None:
        row = sample_transaction(
            res.pool, clicked, res.scorer.thresholds, _rng(), res.cfg.app.demo_fraud_weight
        )
        if row is None:
            st.info(EMPTY_MESSAGES.get(clicked, "No matching transaction found."))
        else:
            record = to_record(row, res.scorer.known_fields)
            result = res.scorer.score_one(record)
            entry = history_entry(
                int(row["TransactionID"]),
                float(row["TransactionAmt"]),
                result.fraud_score,
                result.decision,
                bool(row["isFraud"] == 1),
                res.cfg.costs,
            )
            st.session_state.setdefault("sim_history", []).append(entry)
            st.session_state.sim_current = {"row": row, "result": result}
            _render(row, result, res, animate=not skip)
    elif st.session_state.get("sim_current"):
        current: dict[str, Any] = st.session_state.sim_current
        _render(current["row"], current["result"], res, animate=False)
    else:
        st.info("Pick a button above to generate a transaction.")
    _history_section(res)
