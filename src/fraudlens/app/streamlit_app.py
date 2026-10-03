"""FraudLens Streamlit demo. Run with ``make app``.

UI only: scoring and explanations come from ``fraudlens.scoring.FraudScorer``, the same
code path the API uses.
"""

from __future__ import annotations

import streamlit as st

from fraudlens.app import components as ui
from fraudlens.app.manual import render_manual
from fraudlens.app.resources import Resources, load_resources
from fraudlens.app.simulation import render_simulation
from fraudlens.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def _no_model() -> None:
    st.title("🔎 FraudLens")
    st.info("No trained model found yet. Train one first:")
    st.code(
        "make validate-data   # check data/raw\nmake train           # build features, train, calibrate, pick thresholds\nmake app",
        language="bash",
    )
    st.caption(
        "No real data? `make synthetic-demo` runs the whole pipeline on fake data to try the app."
    )


def _sidebar(res: Resources) -> str:
    with st.sidebar:
        st.title("🔎 FraudLens")
        mode = st.radio("Live demo mode", ["Simulation", "Manual"], horizontal=True)
        st.divider()
        thr = res.scorer.thresholds
        st.markdown("**Decision policy**")
        st.markdown(
            f"- ✓ **APPROVE** if score < {thr.review:.3f}\n"
            f"- ⚑ **REVIEW** if {thr.review:.3f} ≤ score < {thr.block:.3f}\n"
            f"- ✕ **BLOCK** if score ≥ {thr.block:.3f}"
        )
        b = res.cfg.thresholds
        st.caption(
            "Thresholds minimise expected £ cost on the validation block, within budgets of "
            f"≤ {b.max_review_rate:.0%} reviewed and ≤ {b.max_block_rate:.0%} blocked."
        )
        st.divider()
        st.caption(
            f"Model `{res.scorer.version}` · dataset: {res.metrics.get('dataset', 'unknown')}"
        )
    return str(mode)


def main() -> None:
    st.set_page_config(page_title="FraudLens", page_icon="🔎", layout="wide")
    configure_logging()
    ui.inject_css()
    try:
        res = load_resources()
    except FileNotFoundError:
        _no_model()
        return

    mode = _sidebar(res)
    if res.metrics.get("dataset") == "synthetic":
        st.error(
            "Running on SYNTHETIC smoke-test data: scores and metrics are meaningless.", icon="🧪"
        )
    try:
        if mode == "Simulation":
            render_simulation(res)
        else:
            render_manual(res)
    except Exception:  # never show a raw traceback to the user
        logger.exception("App error")
        st.error("Something went wrong while rendering this view. Details are in the server log.")


main()
