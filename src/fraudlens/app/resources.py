"""Cached loading of the model bundle and its companion data for the app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from fraudlens.config import Config, get_config
from fraudlens.models.artifacts import DEMO_POOL_FILE, latest_bundle_dir
from fraudlens.scoring import FraudScorer


@dataclass
class Resources:
    cfg: Config
    scorer: FraudScorer
    pool: pd.DataFrame
    metrics: dict[str, Any]


@st.cache_resource(show_spinner="Loading model…")
def load_resources() -> Resources:
    """Raises FileNotFoundError if no model has been trained yet."""
    cfg = get_config()
    bundle_dir = latest_bundle_dir(cfg)
    scorer = FraudScorer.load(cfg, bundle_dir)
    metrics: dict[str, Any] = {}
    if cfg.paths.metrics_file.is_file():
        metrics = json.loads(cfg.paths.metrics_file.read_text())
    return Resources(
        cfg=cfg,
        scorer=scorer,
        pool=pd.read_parquet(bundle_dir / DEMO_POOL_FILE),
        metrics=metrics,
    )
