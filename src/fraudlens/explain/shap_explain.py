"""SHAP TreeExplainer wrapper.

SHAP values explain the LightGBM model's *raw log-odds* output. The displayed fraud
score additionally passes through a monotone calibrator, so SHAP values tell you which
features pushed the score up or down and by how much in log-odds; they do not add up
to the calibrated probability.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap


class ShapExplainer:
    def __init__(self, booster: lgb.Booster) -> None:
        # A Booster rebuilt from a model string has empty .params; shap reads the objective.
        booster.params.setdefault("objective", "binary")
        self._explainer = shap.TreeExplainer(booster)
        base = np.asarray(self._explainer.expected_value, dtype=np.float64).ravel()
        self.base_value = float(base[-1])

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values, shape ``(n_rows, n_features)``, in log-odds."""
        with warnings.catch_warnings():
            # shap warns that its LightGBM output format changed; handled below.
            warnings.filterwarnings("ignore", message=".*shap values output has changed.*")
            values = self._explainer.shap_values(X)
        if isinstance(values, list):  # older shap: [negative class, positive class]
            values = values[-1]
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 3:  # (n, f, classes)
            arr = arr[..., -1]
        return arr
