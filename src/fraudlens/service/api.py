"""FastAPI scoring service.

Run: ``make serve`` (or ``uvicorn fraudlens.service.api:app``). The model bundle is
loaded once at start-up. If no model exists yet the service still starts, reports
``degraded`` on /health and returns 503 from /score.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from fraudlens import __version__
from fraudlens.config import get_config
from fraudlens.logging_utils import configure_logging, get_logger
from fraudlens.scoring import FraudScorer
from fraudlens.service.schemas import (
    HealthResponse,
    ModelInfoResponse,
    ScoreResponse,
    Transaction,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    if getattr(app.state, "scorer", None) is None:  # tests may inject a scorer
        try:
            app.state.scorer = FraudScorer.load(get_config())
            app.state.load_error = None
            logger.info("Loaded model %s", app.state.scorer.version)
        except FileNotFoundError as exc:
            app.state.scorer = None
            app.state.load_error = str(exc)
            logger.warning("Starting without a model: %s", exc)
    yield


app = FastAPI(
    title="FraudLens scoring API",
    version=__version__,
    description="Scores card transactions, applies cost-based thresholds and explains each decision.",
    lifespan=lifespan,
)


def _scorer(request: Request) -> FraudScorer:
    scorer: FraudScorer | None = getattr(request.app.state, "scorer", None)
    if scorer is None:
        detail = getattr(request.app.state, "load_error", None) or "Model not loaded"
        raise HTTPException(status_code=503, detail=detail)
    return scorer


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    # Never leak a traceback to clients; log it server-side.
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error while scoring"})


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    scorer: FraudScorer | None = getattr(request.app.state, "scorer", None)
    if scorer is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            detail=getattr(request.app.state, "load_error", None),
        )
    return HealthResponse(status="ok", model_loaded=True, model_version=scorer.version)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    scorer = _scorer(request)
    meta = scorer.bundle.metadata
    training = meta.get("training", {})
    return ModelInfoResponse(
        model_version=scorer.version,
        dataset=str(meta.get("dataset", "unknown")),
        trained_at=training.get("trained_at"),
        git_hash=training.get("git_hash"),
        thresholds=scorer.thresholds.as_dict(),  # type: ignore[arg-type]
        metrics_test=meta.get("metrics_test", {}),
        costs_test=meta.get("costs_test", {}),
        n_features=len(scorer.pipeline.feature_names),
    )


@app.post("/score", response_model=ScoreResponse, response_model_exclude_none=True)
def score(
    transaction: Transaction,
    request: Request,
    include_shap: Annotated[bool, Query(description="Also return all raw SHAP values")] = False,
    explain: Annotated[
        bool, Query(description="Compute SHAP reasons (~10x slower than scoring alone)")
    ] = True,
) -> ScoreResponse:
    start = time.perf_counter()
    scorer = _scorer(request)
    record = transaction.to_record()
    unknown = scorer.unknown_fields(transaction.model_extra or {})
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown field(s) for this model: {', '.join(unknown[:20])}",
        )
    result = scorer.score_one(record, explain=explain or include_shap)
    payload = result.as_dict(include_shap=include_shap)
    payload["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
    return ScoreResponse(**payload)
