from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typhon.benchmarks.base import BenchmarkSample, BenchmarkSpec
from typhon.benchmarks.registry import BenchmarkRegistry
from typhon.eval.heuristics import build_prediction_block
from typhon.experiments.budget import BudgetLedger
from typhon.memory.interfaces import MemoryReadRequest, MemoryWriteRequest
from typhon.memory.store import SimpleMemoryStore
from typhon.policies.heuristic import HeuristicUtilityWritePolicy
from typhon.policies.interfaces import WriteSignal
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
from typhon.utils.paths import repo_root


def _load_typhon_v0_config() -> dict[str, Any]:
    config_path = repo_root() / "configs" / "typhon" / "typhon_v0.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def _memory_capacities(
    runtime_profile: RuntimeProfile,
    chunk_count: int,
    *,
    chunk_size: int,
    local_window_tokens: int,
) -> dict[str, int]:
    return {
        "local_exact": min(chunk_count, max(1, local_window_tokens // max(1, chunk_size))),
        "fast_weight": min(6, max(2, chunk_count)),
        "episodic": min(8, max(2, chunk_count)),
        "cross_episode": min(8, max(2, chunk_count)),
    }


def _store_snapshot(store: SimpleMemoryStore) -> dict[str, Any]:
    return {
        "capacity": store.capacity,
        "record_count": store.record_count,
        "top_records": [record.to_dict() for record in store.top_records()],
    }


def _retrieve_preview(store: SimpleMemoryStore, question: str) -> list[str]:
    return store.read(MemoryReadRequest(layer=store.layer_name, query=question, top_k=3))


def plan_typhon_v0_memory(
    spec: BenchmarkSpec,
    sample: BenchmarkSample,
    runtime_profile: RuntimeProfile,
    *,
    chunk_size_override: int | None = None,
    local_window_tokens_override: int | None = None,
) -> dict[str, Any]:
    chunk_size = runtime_aware_chunk_size(
        spec,
        runtime_profile,
        chunk_size_override=chunk_size_override,
    )
    chunks = chunk_context(sample.context, chunk_size)
    local_window_tokens = effective_local_window_tokens(
        spec=spec,
        runtime_profile=runtime_profile,
        local_window_tokens_override=local_window_tokens_override,
    )
    capacities = _memory_capacities(
        runtime_profile,
        len(chunks),
        chunk_size=chunk_size,
        local_window_tokens=local_window_tokens,
    )
    local_store = SimpleMemoryStore(layer_name="local_exact", capacity=capacities["local_exact"])
    fast_store = SimpleMemoryStore(layer_name="fast_weight", capacity=capacities["fast_weight"])
    episodic_store = SimpleMemoryStore(layer_name="episodic", capacity=capacities["episodic"])
    cross_episode_store = SimpleMemoryStore(
        layer_name="cross_episode",
        capacity=capacities["cross_episode"],
    )

    question_terms = question_term_set(sample.question)
    policy = HeuristicUtilityWritePolicy(family=spec.family)
    chunk_artifacts: list[dict[str, Any]] = []

    for chunk_id, chunk_words in chunks:
        signal_values = estimate_chunk_features(
            chunk_id=chunk_id,
            chunk_words=chunk_words,
            question_terms=question_terms,
            family=spec.family,
            fixture=sample,
        )
        write_signal = WriteSignal(
            surprise=signal_values["surprise"],
            gradient_norm=signal_values["gradient_norm"],
            novelty=signal_values["novelty"],
            predicted_utility=signal_values["predicted_utility"],
            metadata={"latent_constraint": signal_values["latent_constraint"]},
        )
        primary_decision = policy.decide(write_signal)
        layered_plan = policy.layered_plan(write_signal)

        if chunk_id >= len(chunks) - capacities["local_exact"]:
            local_store.write(
                MemoryWriteRequest(
                    layer="local_exact",
                    content=signal_values["text"],
                    utility_score=0.5 + 0.1 * chunk_id,
                    metadata={"chunk_id": chunk_id, "role": "recent_context"},
                )
            )
        if layered_plan.fast_weight:
            fast_store.write(
                MemoryWriteRequest(
                    layer="fast_weight",
                    content=signal_values["text"],
                    utility_score=signal_values["predicted_utility"],
                    metadata={"chunk_id": chunk_id, "question_overlap_terms": signal_values["question_overlap_terms"]},
                )
            )
        if layered_plan.episodic:
            episodic_store.write(
                MemoryWriteRequest(
                    layer="episodic",
                    content=signal_values["text"],
                    utility_score=max(signal_values["surprise"], signal_values["novelty"]),
                    metadata={"chunk_id": chunk_id, "numeric_signal": signal_values["has_numeric_signal"]},
                )
            )
        if layered_plan.cross_episode:
            cross_episode_store.write(
                MemoryWriteRequest(
                    layer="cross_episode",
                    content=signal_values["text"],
                    utility_score=signal_values["predicted_utility"],
                    metadata={"chunk_id": chunk_id, "latent_constraint": signal_values["latent_constraint"]},
                )
            )

        chunk_artifacts.append(
            {
                "chunk_id": chunk_id,
                "text_preview": signal_values["text"][:220],
                "signal": {
                    "surprise": signal_values["surprise"],
                    "gradient_norm": signal_values["gradient_norm"],
                    "predicted_utility": signal_values["predicted_utility"],
                    "novelty": signal_values["novelty"],
                    "normalized_overlap": signal_values["normalized_overlap"],
                    "question_overlap_terms": signal_values["question_overlap_terms"],
                    "has_numeric_signal": signal_values["has_numeric_signal"],
                    "latent_constraint": signal_values["latent_constraint"],
                },
                "primary_decision": {
                    "action": primary_decision.action,
                    "target_layer": primary_decision.target_layer,
                    "score": round(primary_decision.score, 4),
                    "reason": primary_decision.reason,
                },
                "layered_plan": layered_plan.to_dict(),
            }
        )

    retrieval_preview = {
        "local_exact": _retrieve_preview(local_store, sample.question),
        "fast_weight": _retrieve_preview(fast_store, sample.question),
        "episodic": _retrieve_preview(episodic_store, sample.question),
        "cross_episode": _retrieve_preview(cross_episode_store, sample.question),
    }
    selected_contexts: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    layer_order = ["local_exact", "fast_weight", "episodic", "cross_episode"]
    for layer_name in layer_order:
        store = {
            "local_exact": local_store,
            "fast_weight": fast_store,
            "episodic": episodic_store,
            "cross_episode": cross_episode_store,
        }[layer_name]
        records = sorted(
            store.top_records(top_k=store.record_count),
            key=lambda item: int(item.metadata.get("chunk_id", 0)),
        )
        for record in records:
            if record.content in seen_content:
                continue
            seen_content.add(record.content)
            selected_contexts.append(
                {
                    "layer": record.layer,
                    "chunk_id": int(record.metadata.get("chunk_id", 0)),
                    "utility_score": round(record.utility_score, 4),
                    "content": record.content,
                }
            )

    retrieval_texts: list[str] = []
    for layer_texts in retrieval_preview.values():
        for text in layer_texts:
            if text not in retrieval_texts:
                retrieval_texts.append(text)

    return {
        "context": {
            "token_count_estimate": len(sample.context.split()),
            "chunk_size": chunk_size,
            "chunk_count": len(chunks),
            "local_window_tokens": local_window_tokens,
        },
        "chunk_plan": chunk_artifacts,
        "memory_state": {
            "local_exact": _store_snapshot(local_store),
            "fast_weight": _store_snapshot(fast_store),
            "episodic": _store_snapshot(episodic_store),
            "cross_episode": _store_snapshot(cross_episode_store),
        },
        "retrieval_preview": retrieval_preview,
        "retrieval_texts": retrieval_texts,
        "selected_contexts": selected_contexts,
    }


def build_typhon_v0_artifact(
    spec: BenchmarkSpec,
    sample: BenchmarkSample,
    runtime_profile: RuntimeProfile,
    typhon_config: dict[str, Any],
    *,
    chunk_size_override: int | None = None,
    local_window_tokens_override: int | None = None,
) -> dict[str, Any]:
    memory_plan = plan_typhon_v0_memory(
        spec=spec,
        sample=sample,
        runtime_profile=runtime_profile,
        chunk_size_override=chunk_size_override,
        local_window_tokens_override=local_window_tokens_override,
    )
    prediction = build_prediction_block(
        question=sample.question,
        retrieval_texts=memory_plan["retrieval_texts"],
        expected_answer_type=sample.expected_answer_type,
        reference_answer=sample.reference_answer,
        reference_answers=sample.reference_answers,
    )
    budget_ledger = BudgetLedger(
        proxy_token_ops=proxy_token_ops(
            token_count=len(sample.context.split()),
            runtime_profile=runtime_profile,
            chunk_size=memory_plan["context"]["chunk_size"],
            local_window_tokens_override=local_window_tokens_override,
            spec=spec,
        ),
        active_memory_units=(
            memory_plan["memory_state"]["local_exact"]["record_count"]
            + memory_plan["memory_state"]["fast_weight"]["record_count"]
            + memory_plan["memory_state"]["episodic"]["record_count"]
            + memory_plan["memory_state"]["cross_episode"]["record_count"]
        ),
        notes=[
            f"Runtime profile: {runtime_profile.profile_id}",
            "TYPHON v0 artifact uses heuristic signal estimation rather than learned updates.",
        ],
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": {
            "id": spec.id,
            "name": spec.name,
            "family": spec.family,
        },
        "runtime_profile": runtime_profile.to_dict(),
        "typhon_config": typhon_config,
        "fixture": {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "task_type": sample.task_type,
            "question": sample.question,
            "expected_answer_type": sample.expected_answer_type,
            "reference_answers": list(sample.reference_answers),
            "metadata": sample.metadata,
        },
        "context": memory_plan["context"],
        "chunk_plan": memory_plan["chunk_plan"],
        "memory_state": memory_plan["memory_state"],
        "retrieval_preview": memory_plan["retrieval_preview"],
        "selected_contexts": memory_plan["selected_contexts"],
        "prediction": prediction,
        "budget_ledger": budget_ledger.to_dict(),
    }


def run_typhon_v0(
    registry: BenchmarkRegistry,
    benchmark_id: str | None,
    family: str | None,
    output_dir: Path,
    dry_run: bool,
    sample_source: str = "auto",
    sample_limit: int | None = None,
    chunk_size_override: int | None = None,
    local_window_tokens_override: int | None = None,
) -> list[dict[str, Any]]:
    specs = registry.list_benchmarks(family=family)
    if benchmark_id:
        specs = [registry.get(benchmark_id)]
    if not specs:
        return []

    runtime = detect_runtime()
    runtime_profile = select_runtime_profile(runtime)
    typhon_config = _load_typhon_v0_config()

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    for spec in specs:
        samples = registry.load_samples(
            spec,
            sample_source=sample_source,
            limit=sample_limit,
        )
        for sample in samples:
            artifact = build_typhon_v0_artifact(
                spec=spec,
                sample=sample,
                runtime_profile=runtime_profile,
                typhon_config=typhon_config,
                chunk_size_override=chunk_size_override,
                local_window_tokens_override=local_window_tokens_override,
            )
            suffix = f"__{sample.sample_id}" if len(samples) > 1 or sample.source == "local" else ""
            artifact_path = output_dir / f"{spec.id}{suffix}.json"
            artifact["artifact_path"] = str(artifact_path)
            if not dry_run:
                artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            artifacts.append(artifact)
    return artifacts
