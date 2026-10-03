"""LightGBM training with a small, time-aware Optuna search.

Time-aware validation: every trial trains on the training block and is scored (PR-AUC)
on the *later* validation block, the same "train on the past, predict the future"
setup as deployment. Early stopping also uses the validation block. The test block
is never seen. Class imbalance is handled with ``scale_pos_weight`` (tuned); no SMOTE,
because synthetic minority samples interpolate between frauds that may be weeks apart,
distort the time structure and produce probabilities that need un-biasing anyway.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from fraudlens.config import Config
from fraudlens.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TuningResult:
    booster: lgb.Booster
    params: dict[str, Any]
    best_iteration: int
    valid_pr_auc: float
    n_trials: int
    seconds: float
    trials: list[dict[str, Any]]


def base_params(cfg: Config) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": cfg.model.learning_rate,
        "seed": cfg.project.seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": cfg.model.num_threads,
        "verbosity": -1,
    }


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 31, 384, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 25.0, log=True),
    }


def make_datasets(
    X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, y_valid: np.ndarray
) -> tuple[lgb.Dataset, lgb.Dataset]:
    """Bin the data once and reuse it across trials (saves time and RAM)."""
    params = {"feature_pre_filter": False}
    dtrain = lgb.Dataset(X_train, y_train, params=params, free_raw_data=True)
    dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain, params=params, free_raw_data=True)
    return dtrain.construct(), dvalid.construct()


def train_once(
    params: dict[str, Any], dtrain: lgb.Dataset, dvalid: lgb.Dataset, cfg: Config
) -> lgb.Booster:
    return lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.model.max_boost_rounds,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(cfg.model.early_stopping_rounds, verbose=False)],
    )


def tune_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    cfg: Config,
) -> TuningResult:
    """Optuna TPE search (seeded); keeps the best trial's booster to avoid retraining."""
    start = time.perf_counter()
    dtrain, dvalid = make_datasets(X_train, y_train, X_valid, y_valid)
    base = base_params(cfg)
    best: dict[str, Any] = {"score": -1.0}

    def objective(trial: optuna.Trial) -> float:
        params = base | suggest_params(trial)
        t0 = time.perf_counter()
        booster = train_once(params, dtrain, dvalid, cfg)
        score = float(
            average_precision_score(
                y_valid, booster.predict(X_valid, num_iteration=booster.best_iteration)
            )
        )
        trial.set_user_attr("best_iteration", booster.best_iteration)
        logger.info(
            "trial %2d: valid PR-AUC=%.4f iters=%d (%.0fs)",
            trial.number,
            score,
            booster.best_iteration,
            time.perf_counter() - t0,
        )
        if score > best["score"]:
            best.update(score=score, booster=booster, params=params)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.project.seed)
    )
    study.optimize(
        objective, n_trials=cfg.model.optuna_trials, timeout=cfg.model.optuna_timeout_seconds
    )
    booster: lgb.Booster = best["booster"]
    return TuningResult(
        booster=booster,
        params=best["params"],
        best_iteration=int(booster.best_iteration),
        valid_pr_auc=float(best["score"]),
        n_trials=len(study.trials),
        seconds=time.perf_counter() - start,
        trials=[
            {"number": t.number, "value": t.value, **t.params, **t.user_attrs} for t in study.trials
        ],
    )


def predict_raw(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Uncalibrated probability from the best iteration."""
    return np.asarray(booster.predict(X, num_iteration=booster.best_iteration))


def feature_importance(booster: lgb.Booster) -> pd.Series:
    """Total gain per feature, descending."""
    return pd.Series(
        booster.feature_importance(importance_type="gain"), index=booster.feature_name()
    ).sort_values(ascending=False)
