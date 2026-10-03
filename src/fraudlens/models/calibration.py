"""Probability calibration fitted on the validation block.

Class weighting (``scale_pos_weight``) deliberately distorts LightGBM's probabilities,
so raw scores cannot be read as fraud probabilities. The calibrator maps raw scores back
to observed fraud frequencies, which the cost-based thresholds rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


@dataclass
class ScoreCalibrator:
    """Isotonic (non-parametric, monotone) or sigmoid / Platt (logistic on the logit)."""

    method: Literal["isotonic", "sigmoid"]
    _model: IsotonicRegression | LogisticRegression | None = field(default=None, repr=False)

    def fit(self, raw_scores: np.ndarray, y: np.ndarray) -> ScoreCalibrator:
        if self.method == "isotonic":
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(np.asarray(raw_scores, dtype=np.float64), y)
        else:
            model = LogisticRegression(C=1e6)  # effectively unregularised Platt scaling
            model.fit(_logit(raw_scores).reshape(-1, 1), y)
        self._model = model
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator is not fitted")
        raw = np.asarray(raw_scores, dtype=np.float64)
        if isinstance(self._model, IsotonicRegression):
            return np.asarray(self._model.predict(raw), dtype=np.float64)
        return np.asarray(self._model.predict_proba(_logit(raw).reshape(-1, 1))[:, 1])
