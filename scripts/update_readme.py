"""Fill README.md and docs/MODEL_CARD.md results from docs/metrics.json (+ benchmark.json).

Only the text between ``<!-- RESULTS:START -->`` / ``<!-- RESULTS:END -->`` and
``<!-- BENCHMARK:START -->`` / ``<!-- BENCHMARK:END -->`` is rewritten. Until a real
training run exists, the placeholder tables stay in place. Synthetic metrics are refused.

Usage:
    python scripts/update_readme.py [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from fraudlens.config import DEFAULT_CONFIG_PATH, load_config
from fraudlens.logging_utils import configure_logging, get_logger

logger = get_logger("update_readme")
README = DEFAULT_CONFIG_PATH.parents[1] / "README.md"
MODEL_CARD = DEFAULT_CONFIG_PATH.parents[1] / "docs" / "MODEL_CARD.md"
LGBM, LOGREG = "LightGBM", "Logistic regression"


def _gbp(v: float) -> str:
    return f"{'-' if v < 0 else ''}£{abs(v):,.0f}"


def results_markdown(m: dict[str, Any], docs_prefix: str = "docs/") -> str:
    t, c = m["test"], m["costs"]
    p_key = next(k for k in t[LGBM] if k.startswith("recall_at_"))
    p_label = p_key.removeprefix("recall_at_").removesuffix("_precision")
    tr = m["training"]
    w = tr["data_window"]
    lines = [
        f"Model `{m['model_version']}` · trained {tr['trained_at']} · git `{tr['git_hash']}` · "
        f"source: [`docs/metrics.json`]({docs_prefix}metrics.json)",
        "",
        f"Held-out **test block**: {w['test']['rows']:,} transactions, "
        f"{w['test']['fraud_rate']:.2%} fraud (never used for training, tuning, calibration or thresholds).",
        "",
        f"| Model | PR-AUC | ROC-AUC | Recall @ {p_label}% precision | Brier (calibrated) |",
        "|---|---|---|---|---|",
    ]
    best = max((LGBM, LOGREG), key=lambda name: t[name]["pr_auc"])
    for name in (LGBM, LOGREG):
        r = t[name]
        pr = f"**{r['pr_auc']:.4f}**" if name == best else f"{r['pr_auc']:.4f}"
        rec = "n/a" if r[p_key] is None else f"{r[p_key]:.1%}"
        lines.append(f"| {name} | {pr} | {r['roc_auc']:.4f} | {rec} | {r['brier']:.4f} |")
    lines += [
        "",
        "| Policy (test block) | Total cost | Fraud stopped | Fraud £ stopped | Genuine blocked | Genuine reviewed |",
        "|---|---|---|---|---|---|",
    ]
    for label, key in (("Do nothing (approve all)", "do_nothing"), (LOGREG, LOGREG), (LGBM, LGBM)):
        s = c[key]["test"]
        lines.append(
            f"| {label} | {_gbp(s['total_cost'])} | {s['fraud_stopped']:,} / {s['n_fraud']:,} "
            f"({s['fraud_recall']:.1%}) | {_gbp(s['fraud_amount_stopped'])} | {s['genuine_blocked']:,} | "
            f"{s['genuine_reviewed']:,} |"
        )
    lg = c[LGBM]["test"]
    budgets = c.get("budgets", {})
    review_rate, block_rate = lg["n_review"] / lg["n"], lg["n_block"] / lg["n"]
    exceeded = [
        name
        for name, rate, cap in (
            ("review", review_rate, budgets.get("max_review_rate", 1)),
            ("block", block_rate, budgets.get("max_block_rate", 1)),
        )
        if rate > cap
    ]
    over = (
        f"on the later test period the {' and '.join(exceeded)} budget is exceeded, a sign of drift "
        "that a live system would need to monitor."
        if exceeded
        else "both budgets also hold on the later test period."
    )
    lines += [
        "",
        f"The LightGBM policy sends **{lg['n_review'] / lg['n']:.1%}** of test transactions to manual "
        f"review and blocks **{lg['n_block'] / lg['n']:.1%}**. Thresholds were tuned to stay within "
        f"budgets of review ≤ {budgets.get('max_review_rate', 1):.0%} and block ≤ "
        f"{budgets.get('max_block_rate', 1):.0%} on *validation*; {over} "
        f"PR-AUC falls from {m['valid'][LGBM]['pr_auc']:.3f} (validation) to {t[LGBM]['pr_auc']:.3f} "
        "(the later test period), consistent with fraud patterns drifting over time.",
    ]
    thr = c[LGBM]["thresholds"]
    lines += [
        "",
        f"LightGBM thresholds (chosen on validation): **REVIEW ≥ {thr['review']:.3f}**, "
        f"**BLOCK ≥ {thr['block']:.3f}**. Savings on the test block: "
        f"**{_gbp(c['savings_vs_do_nothing'][LGBM])}** vs doing nothing, "
        f"**{_gbp(c['savings_vs_logreg'])}** vs the logistic-regression policy.",
        "",
        f"Training time: {tr['seconds_total'] / 60:.1f} min total ({tr['optuna_trials']} Optuna trials, "
        f"{tr['seconds_lgbm_tuning'] / 60:.1f} min) on {tr['machine']}.",
    ]
    return "\n".join(lines)


def benchmark_markdown(b: dict[str, Any], docs_prefix: str = "docs/") -> str:
    def row(label: str, key: str) -> str:
        v = b[key]
        return f"| {label} | {v['p50']} ms | {v['p95']} ms | {v['p99']} ms |"

    lines = [
        f"{b['n_requests']:,} requests per mode ({b['mode']}) on {b['machine']}; "
        f"source: [`docs/benchmark.json`]({docs_prefix}benchmark.json)",
        "",
        "| `POST /score` | p50 | p95 | p99 |",
        "|---|---|---|---|",
        row("Server `latency_ms`, with SHAP reasons (default)", "server_latency_ms"),
        row("Client round trip, with SHAP reasons", "round_trip_ms"),
    ]
    if "server_latency_ms_no_explain" in b:
        lines += [
            row(
                "Server `latency_ms`, score only (`?explain=false`)", "server_latency_ms_no_explain"
            ),
            row("Client round trip, score only", "round_trip_ms_no_explain"),
        ]
    lines += [
        "",
        "Exact TreeSHAP over every tree dominates the explained path; scoring alone is the "
        "feature transform plus one LightGBM prediction.",
    ]
    return "\n".join(lines)


def cv_markdown(m: dict[str, Any], b: dict[str, Any] | None) -> str:
    t, c = m["test"], m["costs"]
    rows = sum(w["rows"] for w in m["training"]["data_window"].values())
    latency = (
        f" served by a FastAPI endpoint (p95 {b['server_latency_ms_no_explain']['p95']:.0f} ms to score, "
        f"{b['server_latency_ms']['p95']:.0f} ms with explanations)"
        if b and "server_latency_ms_no_explain" in b
        else " served by a FastAPI endpoint"
    )
    return (
        "> Built *FraudLens*, an end-to-end card-fraud detection system (Python, LightGBM, SHAP, "
        f"FastAPI, Streamlit, Docker, GitHub Actions) on {rows:,} IEEE-CIS transactions. "
        "It uses leakage-safe time-based validation and past-only behavioural features, reaching "
        f"**{t[LGBM]['pr_auc']:.3f} PR-AUC** on a held-out future period (vs "
        f"{t[LOGREG]['pr_auc']:.3f} for a logistic-regression baseline). Cost-optimised "
        "APPROVE/REVIEW/BLOCK thresholds cut total £ cost (missed fraud + reviews + false alarms) by "
        f"**{c['savings_vs_do_nothing'][LGBM] / (c['savings_vs_do_nothing'][LGBM] + c[LGBM]['test']['total_cost']):.0%}** "
        f"({_gbp(c['savings_vs_do_nothing'][LGBM])}) vs no model on the test period, with "
        f"plain-English SHAP explanations{latency}."
    )


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:START -->\n).*?(\n<!-- {name}:END -->)", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"README is missing the {name} markers")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--readme", default=str(README))
    parser.add_argument("--model-card", default=str(MODEL_CARD))
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    metrics: dict[str, Any] | None = None
    if cfg.paths.metrics_file.is_file():
        loaded: dict[str, Any] = json.loads(cfg.paths.metrics_file.read_text())
        metrics = loaded
        if loaded.get("dataset") == "synthetic":
            logger.error("Refusing to put synthetic metrics in the README")
            return 1
    else:
        logger.warning("%s not found; results placeholders left in place", cfg.paths.metrics_file)
    bench: dict[str, Any] | None = None
    if cfg.paths.benchmark_file.is_file():
        loaded_bench: dict[str, Any] = json.loads(cfg.paths.benchmark_file.read_text())
        bench = None if loaded_bench.get("dataset") == "synthetic" else loaded_bench

    for doc in (Path(args.readme), Path(args.model_card)):
        if not doc.is_file():
            continue
        text = doc.read_text()
        prefix = "" if doc.parent.name == "docs" else "docs/"
        if metrics is not None and "<!-- RESULTS:START -->" in text:
            text = replace_block(text, "RESULTS", results_markdown(metrics, prefix))
        if metrics is not None and "<!-- CV:START -->" in text:
            text = replace_block(text, "CV", cv_markdown(metrics, bench))
        if bench is not None and "<!-- BENCHMARK:START -->" in text:
            text = replace_block(text, "BENCHMARK", benchmark_markdown(bench, prefix))
        doc.write_text(text)
        logger.info("Updated %s", doc.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
