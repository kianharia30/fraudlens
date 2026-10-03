"""Scoring, artefacts and API, against a model trained on synthetic data (conftest)."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudlens.app.logic import to_record
from fraudlens.config import Config
from fraudlens.models.artifacts import DEMO_POOL_FILE, latest_bundle_dir
from fraudlens.models.policy import Decision
from fraudlens.scoring import FraudScorer
from fraudlens.service.api import app


def _record(row: pd.Series, scorer: FraudScorer) -> dict[str, Any]:
    return to_record(row, scorer.known_fields)


def test_training_writes_artefacts_and_metrics(trained_config: Config) -> None:
    out = latest_bundle_dir(trained_config)
    for name in ("bundle.joblib", "metadata.json", "demo_pool.parquet", "eval_scores.parquet"):
        assert (out / name).is_file()
    metrics = json.loads(trained_config.paths.metrics_file.read_text())
    assert metrics["dataset"] == "synthetic"
    assert 0 < metrics["test"]["LightGBM"]["pr_auc"] <= 1
    meta = json.loads((out / "metadata.json").read_text())
    assert {"version", "training", "thresholds", "params"} <= meta.keys()
    assert (trained_config.paths.figures_dir / "pr_curve.png").is_file()


def test_single_and_batch_scoring_agree(scorer: FraudScorer, test_block: pd.DataFrame) -> None:
    rows = test_block.iloc[:5]
    _, batch = scorer.score_frame(rows)
    single = [scorer.score_one(_record(r, scorer)).fraud_score for _, r in rows.iterrows()]
    np.testing.assert_allclose(batch, single, rtol=1e-9)


def test_shap_values_sum_to_raw_log_odds(scorer: FraudScorer, test_block: pd.DataFrame) -> None:
    result = scorer.score_one(_record(test_block.iloc[0], scorer))
    log_odds = math.log(result.raw_score / (1 - result.raw_score))
    assert result.base_value + sum(result.shap_values.values()) == pytest.approx(log_odds, abs=1e-4)
    assert 1 <= len(result.top_reasons) <= 5


def test_decision_matches_thresholds(scorer: FraudScorer, test_block: pd.DataFrame) -> None:
    for _, row in test_block.iloc[:20].iterrows():
        r = scorer.score_one(_record(row, scorer))
        expected = (
            Decision.BLOCK
            if r.fraud_score >= r.thresholds.block
            else Decision.REVIEW if r.fraud_score >= r.thresholds.review else Decision.APPROVE
        )
        assert r.decision is expected


def test_demo_pool_is_from_the_test_block_only(
    trained_config: Config, test_block: pd.DataFrame
) -> None:
    pool = pd.read_parquet(latest_bundle_dir(trained_config) / DEMO_POOL_FILE)
    assert set(pool["TransactionID"]) <= set(test_block["TransactionID"])
    assert pool["isFraud"].sum() == test_block["isFraud"].sum()  # every test fraud included


# ---------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client(scorer: FraudScorer) -> Iterator[TestClient]:
    app.state.scorer = scorer
    with TestClient(app) as c:
        yield c
    app.state.scorer = None


def _payload(test_block: pd.DataFrame, scorer: FraudScorer) -> dict[str, Any]:
    return _record(test_block.iloc[0], scorer)


def test_health_and_model_info(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["dataset"] == "synthetic"
    assert info["thresholds"]["review"] <= info["thresholds"]["block"]


def test_score_endpoint(client: TestClient, test_block: pd.DataFrame, scorer: FraudScorer) -> None:
    payload = _payload(test_block, scorer)
    body = client.post("/score", json=payload).json()
    assert 0 <= body["fraud_score"] <= 1
    assert body["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert body["latency_ms"] > 0 and body["top_reasons"]
    assert "shap_values" not in body
    assert body["fraud_score"] == pytest.approx(scorer.score_one(payload).fraud_score)
    with_shap = client.post("/score?include_shap=true", json=payload).json()
    assert len(with_shap["shap_values"]) == len(scorer.pipeline.feature_names)


@pytest.mark.parametrize(
    "bad",
    [
        {"TransactionAmt": -5},
        {"TransactionAmt": 10, "ProductCD": "Z"},
        {"TransactionAmt": 10, "DeviceType": "fridge"},
        {"TransactionAmt": 10, "V1": [1, 2]},
        {"TransactionAmt": 10, "not_a_feature": 1},
        {},
    ],
)
def test_score_rejects_invalid_input(client: TestClient, bad: dict[str, Any]) -> None:
    assert client.post("/score", json=bad).status_code == 422


def test_minimal_payload_is_scored(client: TestClient) -> None:
    assert client.post("/score", json={"TransactionAmt": 42.0}).status_code == 200


def test_service_without_model_is_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    # monkeypatch restores the shared app state, so later tests still see the model.
    monkeypatch.setattr(app.state, "scorer", None, raising=False)
    monkeypatch.setattr(app.state, "load_error", "No trained model found", raising=False)
    c = TestClient(app)  # no lifespan: keep the injected state
    assert c.get("/health").json() == {
        "status": "degraded",
        "model_loaded": False,
        "model_version": None,
        "detail": "No trained model found",
    }
    assert c.post("/score", json={"TransactionAmt": 1.0}).status_code == 503


def test_score_only_mode_skips_explanations(
    client: TestClient, test_block: pd.DataFrame, scorer: FraudScorer
) -> None:
    payload = _payload(test_block, scorer)
    fast = client.post("/score?explain=false", json=payload).json()
    full = client.post("/score", json=payload).json()
    assert fast["top_reasons"] == [] and full["top_reasons"]
    assert fast["fraud_score"] == pytest.approx(full["fraud_score"])
    assert fast["decision"] == full["decision"]
