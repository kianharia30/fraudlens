"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python's and NumPy's global RNGs and return a fresh seeded Generator.

    Libraries with their own RNGs (LightGBM, Optuna, scikit-learn estimators) must also be
    passed ``seed`` explicitly; this function only covers the global state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses, not the current one
    random.seed(seed)
    np.random.seed(seed)  # legacy global state, still used by some libraries
    return np.random.default_rng(seed)
