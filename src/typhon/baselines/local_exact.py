from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typhon.baselines.base import BaselineSpec
from typhon.baselines.gated_deltanet_fla import run_gated_deltanet_fla_baseline
from typhon.baselines.registry import BaselineRegistry
from typhon.benchmarks.base import BenchmarkSample, BenchmarkSpec
from typhon.benchmarks.registry import BenchmarkRegistry
from typhon.eval.heuristics import build_prediction_block
from typhon.experiments.budget import BudgetLedger
from typhon.memory.interfaces import MemoryWriteRequest
from typhon.memory.store import SimpleMemoryStore
from typhon.runtime.base import RuntimeProfile
from typhon.runtime.detect import detect_runtime
from typhon.runtime.profiles import select_runtime_profile
from typhon.trainers.common import (
    chunk_context,
    effective_local_window_tokens,
    estimate_chunk_features,
    proxy_token_ops,
    question_term_set,
    runtime_aware_chunk_size,
)


def _local_window_chunk_budget(
    chunk_size: int,
    preferred_local_window_tokens: int,
    chunk_count: int,
) -> int:
    budget = max(1, preferred_local_window_tokens // max(1, chunk_size))
    return min(chunk_count, budget)


def _attention_score(signal: dict[str, Any], recent_bias: float) -> float:
    return round(
        0.7 * float(signal["normalized_overlap"])
        + 0.15 * float(signal["predicted_utility"])
        + 0.1 * float(signal["has_numeric_signal"])
        + 0.05 * recent_bias,
        4,
    )


def build_attention_baseline_artifact(
    baseline: BaselineSpec,
    benchmark: BenchmarkSpec,
    sample: BenchmarkSample,
    runtime_profile: RuntimeProfile,
    *,
    chunk_size_override: int | None = None,
    local_window_tokens_override: int | None = None,
) -> dict[str, Any]:
    plan = plan_attention_baseline_context(
        baseline=baseline,
        benchmark=benchmark,
        sample=sample,
        runtime_profile=runtime_profile,
        chunk_size_override=chunk_size_override,
        local_window_tokens_override=local_window_tokens_override,
    )
    prediction = build_prediction_block(
        question=sample.question,
        retrieval_texts=plan["retrieval_texts"],
        expected_answer_type=sample.expected_answer_type,
        reference_answer=sample.reference_answer,
        reference_answers=sample.reference_answers,
    )
    budget_ledger = BudgetLedger(
        proxy_token_ops=proxy_token_ops(
            token_count=len(sample.context.split()),
            runtime_profile=runtime_profile,
            chunk_size=plan["context"]["chunk_size"],
            local_window_tokens_override=local_window_tokens_override,
            spec=benchmark,
        ),
        active_memory_units=plan["memory_state"]["local_exact"]["record_count"],
        notes=[
            f"Runtime profile: {runtime_profile.profile_id}",
            "Baseline uses only local exact recall within the accessible window.",
        ],
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": {
            "id": baseline.id,
            "name": baseline.name,
            "type": baseline.type,
            "retrieval_strategy": baseline.retrieval_strategy,
        },
        "benchmark": {
            "id": benchmark.id,
            "name": benchmark.name,
            "family": benchmark.family,
        },
        "runtime_profile": runtime_profile.to_dict(),
        "fixture": {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "task_type": sample.task_type,
            "question": sample.question,
            "expected_answer_type": sample.expected_answer_type,
            "reference_answers": list(sample.reference_answers),
            "metadata": sample.metadata,
        },
        "context": plan["context"],
        "chunk_plan": plan["chunk_plan"],
        "memory_state": plan["memory_state"],
        "retrieval_preview": plan["retrieval_preview"],
        "selected_contexts": plan["selected_contexts"],
        "prediction": prediction,
        "limitations": [
            "No fast-weight adaptation.",
            "No episodic persistence.",
            "No cross-episode consolidation.",
        ],
        "budget_ledger": budget_ledger.to_dict(),
    }


def plan_attention_baseline_context(
    baseline: BaselineSpec,
    benchmark: BenchmarkSpec,
    sample: BenchmarkSample,
    runtime_profile: RuntimeProfile,
    *,
    chunk_size_override: int | None = None,
    local_window_tokens_override: int | None = None,
) -> dict[str, Any]:
    chunk_size = runtime_aware_chunk_size(
        benchmark,
        runtime_profile,
        chunk_size_override=chunk_size_override,
    )
    chunks = chunk_context(sample.context, chunk_size)
    question_terms = question_term_set(sample.question)

    local_store = SimpleMemoryStore(
        layer_name="local_exact",
        capacity=max(1, len(chunks)),
    )
    accessible_window = _local_window_chunk_budget(
        chunk_size=chunk_size,
        preferred_local_window_tokens=effective_local_window_tokens(
            spec=benchmark,
            runtime_profile=runtime_profile,
            local_window_tokens_override=local_window_tokens_override,
        ),
        chunk_count=len(chunks),
    )
    accessible_start = max(0, len(chunks) - accessible_window)

    chunk_artifacts: list[dict[str, Any]] = []
    accessible_candidates: list[dict[str, Any]] = []
    for chunk_id, chunk_words in chunks:
        signal = estimate_chunk_features(
            chunk_id=chunk_id,
            chunk_words=chunk_words,
            question_terms=question_terms,
            family=benchmark.family,
            fixture=sample,
        )
        is_accessible = chunk_id >= accessible_start
        recent_bias = 1.0 if is_accessible else 0.0
        attention_score = _attention_score(signal=signal, recent_bias=recent_bias)
        if is_accessible:
            local_store.write(
                MemoryWriteRequest(
                    layer="local_exact",
                    content=str(signal["text"]),
                    utility_score=attention_score,
                    metadata={
                        "chunk_id": chunk_id,
                        "attention_score": attention_score,
                        "question_overlap_terms": signal["question_overlap_terms"],
                    },
                )
            )
            accessible_candidates.append(
                {
                    "chunk_id": chunk_id,
                    "attention_score": attention_score,
                    "text": signal["text"],
                    "question_overlap_terms": signal["question_overlap_terms"],
                }
            )

        chunk_artifacts.append(
            {
                "chunk_id": chunk_id,
                "text_preview": str(signal["text"])[:220],
                "is_accessible_in_local_window": is_accessible,
                "attention_score": attention_score,
                "signal": {
                    "normalized_overlap": signal["normalized_overlap"],
                    "predicted_utility": signal["predicted_utility"],
                    "question_overlap_terms": signal["question_overlap_terms"],
                    "has_numeric_signal": signal["has_numeric_signal"],
                },
            }
        )

    accessible_candidates.sort(key=lambda item: item["attention_score"], reverse=True)
    retrieval_preview = [
        item["text"] for item in accessible_candidates[: baseline.max_chunks_to_retrieve]
    ]
    selected_contexts = [
        {
            "layer": "local_exact",
            "chunk_id": item["chunk_id"],
            "utility_score": item["attention_score"],
            "content": item["text"],
        }
        for item in sorted(
            accessible_candidates[: baseline.max_chunks_to_retrieve],
            key=lambda entry: entry["chunk_id"],
        )
    ]

    return {
        "context": {
            "token_count_estimate": len(sample.context.split()),
            "chunk_size": chunk_size,
            "chunk_count": len(chunks),
            "local_window_tokens": effective_local_window_tokens(
                spec=benchmark,
                runtime_profile=runtime_profile,
                local_window_tokens_override=local_window_tokens_override,
            ),
            "accessible_window_chunks": accessible_window,
        },
        "chunk_plan": chunk_artifacts,
        "memory_state": {
            "local_exact": {
                "capacity": local_store.capacity,
                "record_count": local_store.record_count,
                "top_records": [record.to_dict() for record in local_store.top_records()],
            }
        },
        "retrieval_preview": {"local_exact": retrieval_preview},
        "retrieval_texts": retrieval_preview,
        "selected_contexts": selected_contexts,
    }


def run_baseline(
    baseline_registry: BaselineRegistry,
    benchmark_registry: BenchmarkRegistry,
    baseline_id: str,
    benchmark_id: str | None,
    family: str | None,
    output_dir: Path,
    dry_run: bool,
    sample_source: str = "auto",
    sample_limit: int | None = None,
    chunk_size_override: int | None = None,
    local_window_tokens_override: int | None = None,
) -> list[dict[str, Any]]:
    baseline = baseline_registry.get(baseline_id)
    specs = benchmark_registry.list_benchmarks(family=family)
    if benchmark_id:
        specs = [benchmark_registry.get(benchmark_id)]
    if not specs:
        return []

    runtime_profile = select_runtime_profile(detect_runtime())
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    for spec in specs:
        if baseline.supports_families and spec.family not in baseline.supports_families:
            continue
        samples = benchmark_registry.load_samples(
            spec,
            sample_source=sample_source,
            limit=sample_limit,
        )
        if baseline.id == "attention_baseline":
            for sample in samples:
                artifact = build_attention_baseline_artifact(
                    baseline=baseline,
                    benchmark=spec,
                    sample=sample,
                    runtime_profile=runtime_profile,
                    chunk_size_override=chunk_size_override,
                    local_window_tokens_override=local_window_tokens_override,
                )
                suffix = (
                    f"__{sample.sample_id}" if len(samples) > 1 or sample.source == "local" else ""
                )
                artifact_path = output_dir / f"{baseline.id}__{spec.id}{suffix}.json"
                artifact["artifact_path"] = str(artifact_path)
                if not dry_run:
                    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
                artifacts.append(artifact)
            continue
        if baseline.id == "gated_deltanet_fla":
            artifacts.extend(
                run_gated_deltanet_fla_baseline(
                    baseline=baseline,
                    benchmark=spec,
                    samples=samples,
                    runtime_profile=runtime_profile,
                    output_dir=output_dir,
                    dry_run=dry_run,
                )
            )
            continue
        raise NotImplementedError(f"Baseline runner not implemented for {baseline.id}")
    return artifacts
