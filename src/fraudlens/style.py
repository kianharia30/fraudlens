"""Shared colour tokens for figures and the app (validated for colour-vision deficiency).

Series colours follow a fixed categorical order (slot 1 blue, slot 2 orange). Decision
colours are reserved status colours and always appear together with a text label.
"""

from __future__ import annotations

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_1 = "#2a78d6"  # LightGBM / genuine
SERIES_2 = "#eb6834"  # logistic baseline / fraud

MODEL_COLORS = {"LightGBM": SERIES_1, "Logistic regression": SERIES_2}
CLASS_COLORS = {"Genuine": SERIES_1, "Fraud": SERIES_2}
DECISION_COLORS = {"APPROVE": "#0ca30c", "REVIEW": "#fab219", "BLOCK": "#d03b3b"}
DECISION_ICONS = {"APPROVE": "✓", "REVIEW": "⚑", "BLOCK": "✕"}
