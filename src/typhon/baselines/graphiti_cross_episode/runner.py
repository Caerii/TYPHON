"""Runner: orchestrate ingest+search across samples and assemble baseline artifacts.

This is the public entry point (``run_graphiti_cross_episode_baseline``), dispatched from
``local_exact.run_baseline``. It groups samples for graph reuse, resolves facts per sample,
and assembles one artifact per sample in the standard baseline shape — emitting a
``not_executed`` artifact (rather than failing) when graphiti or a backend is unavailable.
The prediction path is identical to ``attention_baseline`` (``build_prediction_block``), so
a head-to-head isolates retrieval: a temporal graph vs. a local window.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typhon.baselines.base import BaselineSpec
from typhon.benchmarks.base import BenchmarkSample, BenchmarkSpec
from typhon.eval.heuristics import build_prediction_block
from typhon.experiments.budget import BudgetLedger
from typhon.runtime.base import RuntimeProfile

from .facts import _order_facts
from .graph import _availability, _group_samples, _run_one_graph
from .sessions import _episodes


def _cost_block(
    group_id: str,
    settings: dict[str, Any],
    usage: dict[str, int],
    episode_count: int,
    price_per_1m: float,
) -> dict[str, Any]:
    """Summarize a group's extraction-LLM usage + a $ estimate (the cost behind the recall).

    Scope is ``per_group_ingestion``: in shared-graph mode the same block appears on every
    sample of a conversation (one ingestion amortized across its QA), so aggregate cost by
    distinct ``group_id`` — not by summing samples — to avoid double counting.
    """
    total = int(usage.get("total_tokens", 0))
    return {
        "scope": "per_group_ingestion",
        "group_id": group_id,
        "llm_model": settings.get("llm_model"),
        "llm_calls": int(usage.get("llm_calls", 0)),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": total,
        "tokens_per_episode": round(total / episode_count) if total and episode_count else 0,
        "usd_estimate": round(total / 1_000_000 * price_per_1m, 6),
        "price_per_1m_tokens": price_per_1m,
    }


def run_graphiti_cross_episode_baseline(
    *,
    baseline: BaselineSpec,
    benchmark: BenchmarkSpec,
    samples: list[BenchmarkSample],
    runtime_profile: RuntimeProfile,
    output_dir: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    settings = baseline.settings
    num_results = int(settings.get("num_results", baseline.max_chunks_to_retrieve or 8))
    # Blended $/1M tokens for the extraction LLM (Together Llama-3.3-70B-Turbo default);
    # override per backend via settings to price the cost behind the recall lift.
    price_per_1m = float(settings.get("llm_price_per_1m_tokens", 0.88))
    available, unavailable_reason = _availability(settings)
    can_run = available and not dry_run

    artifacts: list[dict[str, Any]] = []
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve facts per sample, reusing one graph per group. ``shared_graph_key`` (env
    # GRAPHITI_SHARED_GRAPH_KEY or settings) groups a conversation's QA so the conversation
    # is ingested once; the default (no key) is one graph per sample (original behavior).
    shared_key = os.environ.get("GRAPHITI_SHARED_GRAPH_KEY") or settings.get("shared_graph_key")
    facts_by_sample: dict[str, list[dict[str, Any]]] = {}
    errors_by_sample: dict[str, str] = {}
    group_by_sample: dict[str, str] = {}
    usage_by_group: dict[str, dict[str, int]] = {}
    for group_key, group_samples in _group_samples(samples, str(shared_key) if shared_key else None):
        neo4j_group = f"{benchmark.id}__{group_key}"
        for grouped in group_samples:
            group_by_sample[grouped.sample_id] = neo4j_group
        if not can_run:
            continue
        try:
            group_facts, group_usage = asyncio.run(
                _run_one_graph(group_samples, settings, neo4j_group, num_results)
            )
            facts_by_sample.update(group_facts)
            usage_by_group[neo4j_group] = group_usage
        except Exception as exc:  # noqa: BLE001 - record the failure per sample in the artifact
            for grouped in group_samples:
                errors_by_sample[grouped.sample_id] = f"{type(exc).__name__}: {exc}"

    for sample in samples:
        group_id = group_by_sample.get(sample.sample_id, f"{benchmark.id}__{sample.sample_id}")
        if not can_run:
            facts = []
            status = "not_executed"
            error = "dry-run" if dry_run else unavailable_reason
        elif sample.sample_id in errors_by_sample:
            facts = []
            status = "error"
            error = errors_by_sample[sample.sample_id]
        else:
            facts = facts_by_sample.get(sample.sample_id, [])
            status = "ok"
            error = None

        ordered_facts = _order_facts(facts, settings)
        retrieval_texts = [str(item["fact"]) for item in ordered_facts]
        prediction = build_prediction_block(
            question=sample.question,
            retrieval_texts=retrieval_texts,
            expected_answer_type=sample.expected_answer_type,
            reference_answer=sample.reference_answer,
            reference_answers=sample.reference_answers,
        )

        episodes = _episodes(sample, settings)
        cost = _cost_block(
            group_id, settings, usage_by_group.get(group_id, {}), len(episodes), price_per_1m
        )
        artifact: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "status": status,
            "error": error,
            "baseline": {
                "id": baseline.id,
                "name": baseline.name,
                "type": baseline.type,
                "retrieval_strategy": baseline.retrieval_strategy,
                "settings": baseline.settings,
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
            "memory_state": {
                "cross_episode": {
                    "group_id": group_id,
                    "episode_count": len(episodes),
                    "retrieved_fact_count": len(facts),
                    "current_fact_count": sum(1 for item in facts if item.get("current")),
                    "superseded_fact_count": sum(1 for item in facts if not item.get("current")),
                    "facts": facts,
                }
            },
            "retrieval_preview": {"cross_episode": retrieval_texts},
            "prediction": prediction,
            "cost": cost,
            "limitations": [
                "Graphs are keyed per group (shared_graph_key reuses one graph across a "
                "conversation's QA; default is one graph per sample).",
                "Episodes are stamped with synthetic increasing valid-times, not the "
                "benchmark's real session timestamps.",
                "Extraction quality and cost depend on the configured LLM; long multi-turn "
                "sessions can exceed the extraction token budget.",
            ],
            "budget_ledger": BudgetLedger(
                proxy_token_ops=cost["total_tokens"] or None,
                active_memory_units=len(facts),
                notes=[
                    f"Runtime profile: {runtime_profile.profile_id}",
                    "Cross-episode symbolic memory via Graphiti (bi-temporal knowledge graph).",
                    f"Extraction LLM: {cost['llm_calls']} calls, {cost['total_tokens']} tokens "
                    f"(~${cost['usd_estimate']}, {cost['tokens_per_episode']} tok/episode) "
                    "[per-group ingestion]",
                    f"Status: {status}" + (f" ({error})" if error else ""),
                ],
            ).to_dict(),
        }

        suffix = f"__{sample.sample_id}" if len(samples) > 1 or sample.source == "local" else ""
        artifact_path = output_dir / f"{baseline.id}__{benchmark.id}{suffix}.json"
        artifact["artifact_path"] = str(artifact_path)
        if not dry_run:
            artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        artifacts.append(artifact)

    return artifacts
