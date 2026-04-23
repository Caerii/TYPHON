from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typhon.baselines.registry import BaselineRegistry
from typhon.benchmarks.registry import BenchmarkRegistry
from typhon.eval.memory_compare import evaluate_memory_strategies


def _safe_name(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def evaluate_memory_suite(
    *,
    baseline_registry: BaselineRegistry,
    benchmark_registry: BenchmarkRegistry,
    backend_id: str,
    models: list[str],
    benchmarks: list[str],
    output_dir: Path,
    dry_run: bool,
    sample_source: str = "auto",
    sample_limit: int | None = None,
    chunk_size_override: int | None = None,
    local_window_tokens_override: int | None = None,
    max_output_tokens: int = 128,
    temperature: float = 0.0,
    think: str | None = None,
    request_timeout_seconds: float = 300.0,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    if not models:
        raise RuntimeError("At least one model is required.")
    if not benchmarks:
        raise RuntimeError("At least one benchmark is required.")

    benchmark_summaries: list[dict[str, Any]] = []
    model_rollups: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []

    for model in models:
        model_results: list[dict[str, Any]] = []
        for benchmark_id in benchmarks:
            summary = evaluate_memory_strategies(
                baseline_registry=baseline_registry,
                benchmark_registry=benchmark_registry,
                backend_id=backend_id,
                model=model,
                benchmark_id=benchmark_id,
                family=None,
                output_dir=output_dir / "_suite_tmp",
                dry_run=True,
                sample_source=sample_source,
                sample_limit=sample_limit,
                chunk_size_override=chunk_size_override,
                local_window_tokens_override=local_window_tokens_override,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                think=think,
                request_timeout_seconds=request_timeout_seconds,
                base_url=base_url,
                api_key=api_key,
            )
            benchmark_summaries.append(summary)
            model_results.append(summary)
            deltas = summary["deltas"]["typhon_v0_vs_attention_baseline"]
            leaderboard_rows.append(
                {
                    "model": model,
                    "benchmark_id": benchmark_id,
                    "sample_count": summary["strategies"]["typhon_v0"]["sample_count"],
                    "mean_token_f1_delta": deltas["mean_token_f1"],
                    "mean_token_recall_delta": deltas["mean_token_recall"],
                    "typhon_mean_token_f1": summary["strategies"]["typhon_v0"]["mean_token_f1"],
                    "baseline_mean_token_f1": summary["strategies"]["attention_baseline"]["mean_token_f1"],
                    "full_context_mean_token_f1": summary["strategies"]["full_context"]["mean_token_f1"],
                }
            )

        f1_deltas = [
            float(item["deltas"]["typhon_v0_vs_attention_baseline"]["mean_token_f1"])
            for item in model_results
            if item["deltas"]["typhon_v0_vs_attention_baseline"]["mean_token_f1"] is not None
        ]
        recall_deltas = [
            float(item["deltas"]["typhon_v0_vs_attention_baseline"]["mean_token_recall"])
            for item in model_results
            if item["deltas"]["typhon_v0_vs_attention_baseline"]["mean_token_recall"] is not None
        ]
        typhon_f1 = [
            float(item["strategies"]["typhon_v0"]["mean_token_f1"])
            for item in model_results
            if item["strategies"]["typhon_v0"]["mean_token_f1"] is not None
        ]
        baseline_f1 = [
            float(item["strategies"]["attention_baseline"]["mean_token_f1"])
            for item in model_results
            if item["strategies"]["attention_baseline"]["mean_token_f1"] is not None
        ]
        full_context_f1 = [
            float(item["strategies"]["full_context"]["mean_token_f1"])
            for item in model_results
            if item["strategies"]["full_context"]["mean_token_f1"] is not None
        ]
        model_rollups.append(
            {
                "model": model,
                "benchmark_count": len(model_results),
                "mean_typhon_vs_baseline_token_f1_delta": _mean(f1_deltas),
                "mean_typhon_vs_baseline_token_recall_delta": _mean(recall_deltas),
                "mean_typhon_token_f1": _mean(typhon_f1),
                "mean_baseline_token_f1": _mean(baseline_f1),
                "mean_full_context_token_f1": _mean(full_context_f1),
            }
        )

    leaderboard_rows.sort(
        key=lambda item: (
            item["mean_token_f1_delta"] is None,
            -(item["mean_token_f1_delta"] or 0.0),
            -(item["mean_token_recall_delta"] or 0.0),
        )
    )
    model_rollups.sort(
        key=lambda item: (
            item["mean_typhon_vs_baseline_token_f1_delta"] is None,
            -(item["mean_typhon_vs_baseline_token_f1_delta"] or 0.0),
        )
    )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "subject": {
            "kind": "memory_strategy_suite",
            "backend": backend_id,
            "models": models,
            "benchmarks": benchmarks,
        },
        "run_config": {
            "sample_source": sample_source,
            "sample_limit": sample_limit,
            "chunk_size_override": chunk_size_override,
            "local_window_tokens_override": local_window_tokens_override,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "think": think,
            "request_timeout_seconds": request_timeout_seconds,
            "base_url": base_url,
        },
        "model_rollups": model_rollups,
        "leaderboard": leaderboard_rows,
        "benchmark_summaries": benchmark_summaries,
    }

    safe_models = "__".join(_safe_name(model) for model in models)
    safe_benchmarks = "__".join(_safe_name(benchmark_id) for benchmark_id in benchmarks)
    artifact_path = output_dir / f"memory_suite__{backend_id}__{safe_models}__{safe_benchmarks}.json"
    summary["artifact_path"] = str(artifact_path)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
