from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typhon.baselines.local_exact import plan_attention_baseline_context
from typhon.baselines.registry import BaselineRegistry
from typhon.benchmarks.registry import BenchmarkRegistry
from typhon.eval.heuristics import score_prediction
from typhon.inference.base import GenerationRequest
from typhon.inference.factory import create_backend
from typhon.inference.prompting import build_selected_context_prompt
from typhon.runtime.detect import detect_runtime
from typhon.runtime.profiles import select_runtime_profile
from typhon.trainers.v0 import plan_typhon_v0_memory


def _aggregate_prediction_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    error_count = sum(1 for result in results if result.get("error"))
    scored = [result for result in results if result["metrics"].get("has_reference")]
    if not scored:
        return {
            "sample_count": len(results),
            "scored_sample_count": 0,
            "error_count": error_count,
            "exact_match_rate": None,
            "mean_token_f1": None,
            "mean_token_recall": None,
        }

    exact_match_rate = sum(1.0 if item["metrics"]["exact_match"] else 0.0 for item in scored) / len(scored)
    mean_token_f1 = sum(float(item["metrics"]["token_f1"]) for item in scored) / len(scored)
    mean_token_recall = sum(float(item["metrics"]["token_recall"]) for item in scored) / len(scored)
    return {
        "sample_count": len(results),
        "scored_sample_count": len(scored),
        "error_count": error_count,
        "exact_match_rate": round(exact_match_rate, 4),
        "mean_token_f1": round(mean_token_f1, 4),
        "mean_token_recall": round(mean_token_recall, 4),
    }


def _metric_delta(lhs: dict[str, Any], rhs: dict[str, Any], key: str) -> float | None:
    left = lhs.get(key)
    right = rhs.get(key)
    if left is None or right is None:
        return None
    return round(float(lhs[key]) - float(rhs[key]), 4)


def _dedupe_segments(segments: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        normalized = " ".join(segment.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(segment)
    return unique


def _strategy_contexts(
    *,
    baseline_plan: dict[str, Any],
    typhon_plan: dict[str, Any],
    full_context: str,
) -> dict[str, list[str]]:
    typhon_segments = [
        entry["content"]
        for entry in sorted(
            typhon_plan["selected_contexts"],
            key=lambda item: (int(item["chunk_id"]), item["layer"]),
        )
    ]
    baseline_segments = [
        entry["content"]
        for entry in sorted(
            baseline_plan["selected_contexts"],
            key=lambda item: int(item["chunk_id"]),
        )
    ]
    return {
        "full_context": [full_context],
        "attention_baseline": _dedupe_segments(baseline_segments),
        "typhon_v0": _dedupe_segments(typhon_segments),
    }


def evaluate_memory_strategies(
    *,
    baseline_registry: BaselineRegistry,
    benchmark_registry: BenchmarkRegistry,
    backend_id: str,
    model: str,
    benchmark_id: str | None,
    family: str | None,
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
    specs = benchmark_registry.list_benchmarks(family=family)
    if benchmark_id:
        specs = [benchmark_registry.get(benchmark_id)]
    if not specs:
        raise RuntimeError("No benchmarks matched the requested filter.")

    baseline = baseline_registry.get("attention_baseline")
    runtime_profile = select_runtime_profile(detect_runtime())
    backend = create_backend(backend_id, base_url=base_url, api_key=api_key)
    status = backend.status()
    if not status.available:
        raise RuntimeError(status.message or f"Backend {backend_id} is not available.")

    strategy_results: dict[str, list[dict[str, Any]]] = {
        "full_context": [],
        "attention_baseline": [],
        "typhon_v0": [],
    }
    sample_rows: list[dict[str, Any]] = []

    for spec in specs:
        samples = benchmark_registry.load_samples(
            spec,
            sample_source=sample_source,
            limit=sample_limit,
        )
        for sample in samples:
            baseline_plan = plan_attention_baseline_context(
                baseline=baseline,
                benchmark=spec,
                sample=sample,
                runtime_profile=runtime_profile,
                chunk_size_override=chunk_size_override,
                local_window_tokens_override=local_window_tokens_override,
            )
            typhon_plan = plan_typhon_v0_memory(
                spec=spec,
                sample=sample,
                runtime_profile=runtime_profile,
                chunk_size_override=chunk_size_override,
                local_window_tokens_override=local_window_tokens_override,
            )
            strategy_contexts = _strategy_contexts(
                baseline_plan=baseline_plan,
                typhon_plan=typhon_plan,
                full_context=sample.context,
            )

            sample_result: dict[str, Any] = {
                "benchmark_id": spec.id,
                "sample_id": sample.sample_id,
                "source": sample.source,
                "question": sample.question,
                "reference_answer": sample.reference_answer,
                "strategies": {},
            }

            for strategy_id, context_segments in strategy_contexts.items():
                system_prompt, user_prompt = build_selected_context_prompt(
                    spec,
                    sample,
                    strategy_id=strategy_id,
                    context_segments=context_segments,
                )
                error: str | None = None
                try:
                    generation = backend.generate(
                        GenerationRequest(
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            benchmark_id=spec.id,
                            question=sample.question,
                            context="\n\n".join(context_segments),
                            expected_answer_type=sample.expected_answer_type,
                            max_output_tokens=max_output_tokens,
                            temperature=temperature,
                            think=think,
                            request_timeout_seconds=request_timeout_seconds,
                        )
                    )
                    predicted_answer = generation.content
                    usage = generation.usage
                except Exception as exc:
                    predicted_answer = ""
                    usage = {}
                    error = f"{type(exc).__name__}: {exc}"

                metrics = score_prediction(
                    predicted_answer,
                    sample.reference_answer,
                    reference_answers=sample.reference_answers,
                )
                result_row = {
                    "benchmark_id": spec.id,
                    "sample_id": sample.sample_id,
                    "strategy_id": strategy_id,
                    "predicted_answer": predicted_answer,
                    "reference_answer": sample.reference_answer,
                    "reference_answers": list(sample.reference_answers),
                    "metrics": metrics,
                    "context_excerpt_count": len(context_segments),
                    "context_token_estimate": sum(len(segment.split()) for segment in context_segments),
                    "context_previews": [segment[:180] for segment in context_segments[:4]],
                    "usage": usage,
                    "error": error,
                }
                strategy_results[strategy_id].append(result_row)
                sample_result["strategies"][strategy_id] = result_row

            sample_rows.append(sample_result)

    strategy_summaries = {
        strategy_id: _aggregate_prediction_metrics(rows)
        for strategy_id, rows in strategy_results.items()
    }
    comparison = {
        "generated_at": datetime.now(UTC).isoformat(),
        "subject": {
            "kind": "memory_strategy_compare",
            "backend": backend_id,
            "model": model,
            "scope": benchmark_id or family or "all",
        },
        "backend_status": status.to_dict(),
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
        "strategies": strategy_summaries,
        "deltas": {
            "typhon_v0_vs_attention_baseline": {
                "exact_match_rate": _metric_delta(
                    strategy_summaries["typhon_v0"],
                    strategy_summaries["attention_baseline"],
                    "exact_match_rate",
                ),
                "mean_token_f1": _metric_delta(
                    strategy_summaries["typhon_v0"],
                    strategy_summaries["attention_baseline"],
                    "mean_token_f1",
                ),
                "mean_token_recall": _metric_delta(
                    strategy_summaries["typhon_v0"],
                    strategy_summaries["attention_baseline"],
                    "mean_token_recall",
                ),
            },
            "typhon_v0_vs_full_context": {
                "exact_match_rate": _metric_delta(
                    strategy_summaries["typhon_v0"],
                    strategy_summaries["full_context"],
                    "exact_match_rate",
                ),
                "mean_token_f1": _metric_delta(
                    strategy_summaries["typhon_v0"],
                    strategy_summaries["full_context"],
                    "mean_token_f1",
                ),
                "mean_token_recall": _metric_delta(
                    strategy_summaries["typhon_v0"],
                    strategy_summaries["full_context"],
                    "mean_token_recall",
                ),
            },
        },
        "samples": sample_rows,
    }
    filename_scope = benchmark_id or family or "all"
    safe_model = model.replace(":", "_").replace("/", "_")
    artifact_path = output_dir / f"memory_compare__{backend_id}__{safe_model}__{filename_scope}.json"
    comparison["artifact_path"] = str(artifact_path)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison
