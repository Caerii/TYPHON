from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typhon.baselines.local_exact import run_baseline
from typhon.baselines.registry import BaselineRegistry
from typhon.benchmarks.registry import BenchmarkRegistry
from typhon.trainers.v0 import run_typhon_v0


def _non_empty_layers(mapping: dict[str, Any]) -> list[str]:
    return [key for key, value in mapping.items() if value]


def _memory_counts(memory_state: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer, payload in memory_state.items():
        counts[layer] = int(payload.get("record_count", 0))
    return counts


def _build_observations(
    baseline_artifact: dict[str, Any],
    typhon_artifact: dict[str, Any],
) -> list[str]:
    observations: list[str] = []
    baseline_layers = _non_empty_layers(baseline_artifact["retrieval_preview"])
    typhon_layers = _non_empty_layers(typhon_artifact["retrieval_preview"])
    if len(typhon_layers) > len(baseline_layers):
        observations.append(
            "TYPHON exposes more active retrieval layers than the local baseline on this sample."
        )

    typhon_memory = _memory_counts(typhon_artifact["memory_state"])
    if typhon_memory.get("cross_episode", 0) > 0:
        observations.append("TYPHON allocated cross-episode memory for this sample.")
    if typhon_memory.get("episodic", 0) > 0:
        observations.append("TYPHON allocated episodic memory for this sample.")
    if typhon_memory.get("fast_weight", 0) > 0:
        observations.append("TYPHON allocated fast-weight memory for this sample.")
    if typhon_artifact["budget_ledger"]["active_memory_units"] > baseline_artifact["budget_ledger"]["active_memory_units"]:
        observations.append("TYPHON used more active memory units than the baseline on this sample.")
    if not observations:
        observations.append("No clear structural difference was detected on this sample.")
    return observations


def build_comparison_artifact(
    baseline_artifact: dict[str, Any],
    typhon_artifact: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": typhon_artifact["benchmark"],
        "sample": {
            "sample_id": typhon_artifact["fixture"]["sample_id"],
            "source": typhon_artifact["fixture"]["source"],
        },
        "runtime_profile": typhon_artifact["runtime_profile"],
        "baseline": {
            "id": baseline_artifact["baseline"]["id"],
            "source_mode": "generated_in_memory",
            "active_layers": _non_empty_layers(baseline_artifact["retrieval_preview"]),
            "active_memory_units": baseline_artifact["budget_ledger"]["active_memory_units"],
            "proxy_token_ops": baseline_artifact["budget_ledger"]["proxy_token_ops"],
        },
        "typhon_v0": {
            "source_mode": "generated_in_memory",
            "active_layers": _non_empty_layers(typhon_artifact["retrieval_preview"]),
            "active_memory_units": typhon_artifact["budget_ledger"]["active_memory_units"],
            "proxy_token_ops": typhon_artifact["budget_ledger"]["proxy_token_ops"],
            "memory_counts": _memory_counts(typhon_artifact["memory_state"]),
        },
        "observations": _build_observations(
            baseline_artifact=baseline_artifact,
            typhon_artifact=typhon_artifact,
        ),
    }


def compare_baseline_to_typhon_v0(
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
    baseline_artifacts = run_baseline(
        baseline_registry=baseline_registry,
        benchmark_registry=benchmark_registry,
        baseline_id=baseline_id,
        benchmark_id=benchmark_id,
        family=family,
        output_dir=output_dir / "_baseline_tmp",
        dry_run=True,
        sample_source=sample_source,
        sample_limit=sample_limit,
        chunk_size_override=chunk_size_override,
        local_window_tokens_override=local_window_tokens_override,
    )
    typhon_artifacts = run_typhon_v0(
        registry=benchmark_registry,
        benchmark_id=benchmark_id,
        family=family,
        output_dir=output_dir / "_typhon_tmp",
        dry_run=True,
        sample_source=sample_source,
        sample_limit=sample_limit,
        chunk_size_override=chunk_size_override,
        local_window_tokens_override=local_window_tokens_override,
    )

    typhon_by_benchmark = {
        (artifact["benchmark"]["id"], artifact["fixture"]["sample_id"]): artifact
        for artifact in typhon_artifacts
    }
    comparisons: list[dict[str, Any]] = []
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for baseline_artifact in baseline_artifacts:
        benchmark_key = baseline_artifact["benchmark"]["id"]
        sample_key = baseline_artifact["fixture"]["sample_id"]
        typhon_artifact = typhon_by_benchmark.get((benchmark_key, sample_key))
        if typhon_artifact is None:
            continue
        comparison = build_comparison_artifact(
            baseline_artifact=baseline_artifact,
            typhon_artifact=typhon_artifact,
        )
        suffix = (
            f"__{sample_key}"
            if baseline_artifact["fixture"]["source"] == "local"
            or typhon_artifact["fixture"]["source"] == "local"
            else ""
        )
        artifact_path = output_dir / f"{baseline_id}__vs__typhon_v0__{benchmark_key}{suffix}.json"
        comparison["artifact_path"] = str(artifact_path)
        if not dry_run:
            artifact_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        comparisons.append(comparison)

    return comparisons
