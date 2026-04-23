from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typhon.benchmarks.base import BenchmarkSample, BenchmarkSpec
from typhon.benchmarks.registry import BenchmarkRegistry
from typhon.experiments.budget import BudgetLedger
from typhon.utils.text import significant_terms


def _chunk_terms(text: str, chunk_size: int) -> list[tuple[int, list[str]]]:
    words = text.split()
    chunks: list[tuple[int, list[str]]] = []
    for start in range(0, len(words), chunk_size):
        chunk_words = words[start : start + chunk_size]
        chunks.append((start // chunk_size, chunk_words))
    return chunks


def _extract_rare_terms(text: str) -> list[str]:
    raw_terms = significant_terms(text)
    counts: dict[str, int] = {}
    for term in raw_terms:
        counts[term] = counts.get(term, 0) + 1
    return sorted(term for term, count in counts.items() if count == 1)


def _chunk_analysis(question: str, chunk_words: list[str]) -> dict[str, Any]:
    chunk_text = " ".join(chunk_words)
    question_terms = set(significant_terms(question))
    chunk_terms = set(significant_terms(chunk_text))
    overlap = sorted(question_terms.intersection(chunk_terms))
    rare_terms = _extract_rare_terms(chunk_text)[:5]
    return {
        "text_preview": chunk_text[:220],
        "token_count_estimate": len(chunk_words),
        "question_overlap_terms": overlap,
        "overlap_score": len(overlap),
        "has_numeric_signal": bool(re.search(r"\d", chunk_text)),
        "rare_terms": rare_terms,
    }


def build_smoke_artifact(
    spec: BenchmarkSpec,
    sample: BenchmarkSample,
    chunk_size: int | None = None,
    local_window_tokens_override: int | None = None,
) -> dict[str, Any]:
    actual_chunk_size = chunk_size or spec.default_chunk_size
    chunks = _chunk_terms(sample.context, actual_chunk_size)
    chunk_records: list[dict[str, Any]] = []
    for chunk_id, chunk_words in chunks:
        analysis = _chunk_analysis(sample.question, chunk_words)
        chunk_records.append({"chunk_id": chunk_id, **analysis})

    ranked_chunks = sorted(
        chunk_records,
        key=lambda item: (
            item["overlap_score"],
            item["has_numeric_signal"],
            item["token_count_estimate"],
        ),
        reverse=True,
    )
    top_chunks = ranked_chunks[: min(3, len(ranked_chunks))]
    recent_chunk_ids = [item["chunk_id"] for item in chunk_records[-2:]]
    episodic_candidates = [
        item["chunk_id"]
        for item in ranked_chunks
        if item["has_numeric_signal"] or item["rare_terms"]
    ][:3]
    cross_episode_candidates = []
    if spec.family in {"conversational_memory", "streaming_agentic_memory", "continual_learning"}:
        cross_episode_candidates = [
            item["chunk_id"] for item in ranked_chunks if item["overlap_score"] > 0
        ][:2]
        if not cross_episode_candidates:
            cross_episode_candidates = episodic_candidates[:2] or [
                item["chunk_id"] for item in top_chunks
            ][:2]

    local_window_tokens = local_window_tokens_override or spec.default_local_window
    proxy_token_ops = len(sample.context.split()) * max(local_window_tokens, actual_chunk_size)
    ledger = BudgetLedger(
        proxy_token_ops=proxy_token_ops,
        active_memory_units=len(top_chunks) + len(episodic_candidates) + len(cross_episode_candidates),
        notes=[
            "Smoke test uses heuristic allocation only.",
            "Real FLOP accounting must replace proxy_token_ops once model runners exist.",
        ],
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": {
            "id": spec.id,
            "name": spec.name,
            "family": spec.family,
            "paper_url": spec.paper_url,
            "dataset_ref": spec.dataset_ref,
            "tasks": list(spec.tasks),
        },
        "fixture": {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "task_type": sample.task_type,
            "question": sample.question,
            "expected_answer_type": sample.expected_answer_type,
            "metadata": sample.metadata,
        },
        "context": {
            "token_count_estimate": len(sample.context.split()),
            "chunk_size": actual_chunk_size,
            "chunk_count": len(chunk_records),
            "local_window_tokens": local_window_tokens,
        },
        "retrieval_preview": top_chunks,
        "memory_plan": {
            "local_exact_recall_chunk_ids": recent_chunk_ids,
            "fast_weight_chunk_ids": [item["chunk_id"] for item in top_chunks],
            "episodic_chunk_ids": episodic_candidates,
            "cross_episode_chunk_ids": cross_episode_candidates,
            "write_policy_rationale": [
                "Fast-weight candidates are selected from highest question-overlap chunks.",
                "Episodic candidates favor numeric or rare-term signals.",
                "Cross-episode candidates are enabled only for conversational, streaming, or continual-learning families.",
                "When lexical overlap is weak, cross-episode fallback uses episodic or top-ranked chunks to preserve latent constraints.",
            ],
        },
        "budget_ledger": ledger.to_dict(),
    }


def _select_specs(
    registry: BenchmarkRegistry,
    benchmark_id: str | None,
    family: str | None,
) -> list[BenchmarkSpec]:
    if benchmark_id:
        return [registry.get(benchmark_id)]
    return registry.list_benchmarks(family=family)


def run_smoke_tests(
    registry: BenchmarkRegistry,
    benchmark_id: str | None,
    family: str | None,
    chunk_size: int | None,
    output_dir: Path,
    dry_run: bool,
    sample_source: str = "fixture",
    sample_limit: int | None = None,
    local_window_tokens_override: int | None = None,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    specs = _select_specs(registry=registry, benchmark_id=benchmark_id, family=family)
    if not specs:
        return artifacts

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        samples = registry.load_samples(spec, sample_source=sample_source, limit=sample_limit)
        for sample in samples:
            artifact = build_smoke_artifact(
                spec=spec,
                sample=sample,
                chunk_size=chunk_size,
                local_window_tokens_override=local_window_tokens_override,
            )
            suffix = f"__{sample.sample_id}" if len(samples) > 1 or sample.source == "local" else ""
            artifact_path = output_dir / f"{spec.id}{suffix}.json"
            artifact["artifact_path"] = str(artifact_path)
            if not dry_run:
                artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            artifacts.append(artifact)
    return artifacts
