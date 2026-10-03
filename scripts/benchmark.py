"""Latency benchmark for POST /score using real held-out demo-pool transactions.

Usage:
    make serve                                  # in one terminal
    python scripts/benchmark.py                 # in another (default http://localhost:8000)
    python scripts/benchmark.py --in-process    # no server needed (FastAPI TestClient)

Reports p50/p95/p99 of both the server-side ``latency_ms`` (scoring + SHAP explanation)
and the client round trip, and writes them to the config's benchmark_file.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from typing import Any

import httpx
import numpy as np
import pandas as pd

from fraudlens.config import Config, load_config
from fraudlens.logging_utils import configure_logging, get_logger
from fraudlens.models.artifacts import DEMO_POOL_FILE, latest_bundle_dir

logger = get_logger("benchmark")
NON_FEATURE_COLS = (
    "isFraud",
    "raw_score",
    "fraud_score",
    "decision",
    "TransactionID",
    "TransactionDT",
)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_payloads(cfg: Config, n: int, seed: int) -> list[dict[str, Any]]:
    pool = pd.read_parquet(latest_bundle_dir(cfg) / DEMO_POOL_FILE)
    rows = pool.sample(n=n, replace=len(pool) < n, random_state=seed)
    rows = rows.drop(columns=[c for c in NON_FEATURE_COLS if c in rows.columns])
    return [{k: _json_safe(v) for k, v in rec.items()} for rec in rows.to_dict("records")]


def percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values)
    return {f"p{q}": round(float(np.percentile(arr, q)), 2) for q in (50, 95, 99)} | {
        "mean": round(float(arr.mean()), 2)
    }


def run(
    client: Any, payloads: list[dict[str, Any]], warmup: int, explain: bool = True
) -> tuple[list[float], list[float]]:
    server, round_trip = [], []
    path = "/score" if explain else "/score?explain=false"
    for i, payload in enumerate(payloads):
        t0 = time.perf_counter()
        response = client.post(path, json=payload)
        elapsed = (time.perf_counter() - t0) * 1000
        response.raise_for_status()
        if i >= warmup:
            server.append(float(response.json()["latency_ms"]))
            round_trip.append(elapsed)
    return server, round_trip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--in-process", action="store_true")
    args = parser.parse_args(argv)

    configure_logging()
    cfg = load_config(args.config)
    payloads = load_payloads(cfg, args.n + args.warmup, cfg.project.seed)

    if args.in_process:
        from fastapi.testclient import TestClient

        from fraudlens.service.api import app

        with TestClient(app) as client:
            runs = {e: run(client, payloads, args.warmup, e) for e in (True, False)}
        mode = "in-process (TestClient)"
    else:
        try:
            with httpx.Client(base_url=args.url, timeout=10) as client:
                client.get("/health").raise_for_status()
                runs = {e: run(client, payloads, args.warmup, e) for e in (True, False)}
        except httpx.HTTPError as exc:
            logger.error(
                "Could not reach the API at %s (%s). Start it with `make serve`.", args.url, exc
            )
            return 1
        mode = f"HTTP {args.url}"

    result = {
        "n_requests": len(runs[True][0]),
        "mode": mode,
        "server_latency_ms": percentiles(runs[True][0]),
        "round_trip_ms": percentiles(runs[True][1]),
        "server_latency_ms_no_explain": percentiles(runs[False][0]),
        "round_trip_ms_no_explain": percentiles(runs[False][1]),
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "dataset": cfg.project.dataset,
    }
    cfg.paths.benchmark_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.benchmark_file.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Benchmark (%s), with SHAP: %s", mode, json.dumps(result["server_latency_ms"]))
    logger.info("Score only: %s", json.dumps(result["server_latency_ms_no_explain"]))
    logger.info(
        "Round trip: %s. Written to %s",
        json.dumps(result["round_trip_ms"]),
        cfg.paths.benchmark_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
