from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typhon.baselines.local_exact import run_baseline
from typhon.baselines.registry import BaselineRegistry
from typhon.benchmarks.registry import BenchmarkRegistry
from typhon.inference.runner import run_inference_backend
from typhon.trainers.v0 import run_typhon_v0


def _aggregate_prediction_metrics(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        artifact for artifact in artifacts if artifact.get("prediction", {}).get("metrics", {}).get("has_reference")
    ]
    if not scored:
        return {
            "sample_count": len(artifacts),
            "scored_sample_count": 0,
            "exact_match_rate": None,
            "mean_token_f1": None,
            "mean_token_recall": None,
        }

    exact_matches = [
        1.0 if artifact["prediction"]["metrics"]["exact_match"] else 0.0 for artifact in scored
    ]
    token_f1 = [float(artifact["prediction"]["metrics"]["token_f1"]) for artifact in scored]
    token_recall = [float(artifact["prediction"]["metrics"]["token_recall"]) for artifact in scored]
    return {
        "sample_count": len(artifacts),
        "scored_sample_count": len(scored),
        "exact_match_rate": round(sum(exact_matches) / len(scored), 4),
        "mean_token_f1": round(sum(token_f1) / len(scored), 4),
        "mean_token_recall": round(sum(token_recall) / len(scored), 4),
    }


def summarize_artifacts(
    *,
    subject: dict[str, Any],
    artifacts: list[dict[str, Any]],
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "subject": subject,
        "run_config": run_config or {},
        "aggregate": _aggregate_prediction_metrics(artifacts),
        "samples": [
            {
                "benchmark_id": artifact["benchmark"]["id"],
                "sample_id": artifact["fixture"]["sample_id"],
                "source": artifact["fixture"]["source"],
                "predicted_answer": artifact.get("prediction", {}).get("predicted_answer"),
                "reference_answer": artifact.get("prediction", {}).get("reference_answer"),
                "reference_answers": artifact.get("prediction", {}).get("reference_answers", []),
                "metrics": artifact.get("prediction", {}).get("metrics", {}),
                "artifact_mode": "generated_in_memory",
            }
            for artifact in artifacts
        ],
    }


def _write_summary(
    *,
    summary: dict[str, Any],
    output_dir: Path,
    filename: str,
    dry_run: bool,
) -> dict[str, Any]:
    artifact_path = output_dir / filename
    summary["artifact_path"] = str(artifact_path)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _config_suffix(
    *,
    chunk_size_override: int | None,
    local_window_tokens_override: int | None,
) -> str:
    parts: list[str] = []
    if chunk_size_override is not None:
        parts.append(f"chunk{chunk_size_override}")
    if local_window_tokens_override is not None:
        parts.append(f"window{local_window_tokens_override}")
    if not parts:
        return ""
    return "__" + "__".join(parts)


def evaluate_baseline(
    *,
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
) -> dict[str, Any]:
    artifacts = run_baseline(
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
    filename_scope = benchmark_id or family or "all"
    summary = summarize_artifacts(
        subject={"kind": "baseline", "id": baseline_id, "scope": filename_scope},
        artifacts=artifacts,
        run_config={
            "sample_source": sample_source,
            "sample_limit": sample_limit,
            "chunk_size_override": chunk_size_override,
            "local_window_tokens_override": local_window_tokens_override,
        },
    )
    return _write_summary(
        summary=summary,
        output_dir=output_dir,
        filename=f"baseline__{baseline_id}__{filename_scope}{_config_suffix(chunk_size_override=chunk_size_override, local_window_tokens_override=local_window_tokens_override)}.json",
        dry_run=dry_run,
    )


def evaluate_typhon_v0(
    *,
    benchmark_registry: BenchmarkRegistry,
    benchmark_id: str | None,
    family: str | None,
    output_dir: Path,
    dry_run: bool,
    sample_source: str = "auto",
    sample_limit: int | None = None,
    chunk_size_override: int | None = None,
    local_window_tokens_override: int | None = None,
) -> dict[str, Any]:
    artifacts = run_typhon_v0(
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
    filename_scope = benchmark_id or family or "all"
    summary = summarize_artifacts(
        subject={"kind": "typhon_v0", "id": "typhon_v0", "scope": filename_scope},
        artifacts=artifacts,
        run_config={
            "sample_source": sample_source,
            "sample_limit": sample_limit,
            "chunk_size_override": chunk_size_override,
            "local_window_tokens_override": local_window_tokens_override,
        },
    )
    return _write_summary(
        summary=summary,
        output_dir=output_dir,
        filename=f"typhon_v0__{filename_scope}{_config_suffix(chunk_size_override=chunk_size_override, local_window_tokens_override=local_window_tokens_override)}.json",
        dry_run=dry_run,
    )


def evaluate_compare(
    *,
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
) -> dict[str, Any]:
    baseline_summary = evaluate_baseline(
        baseline_registry=baseline_registry,
        benchmark_registry=benchmark_registry,
        baseline_id=baseline_id,
        benchmark_id=benchmark_id,
        family=family,
        output_dir=output_dir / "_baseline_eval_tmp",
        dry_run=True,
        sample_source=sample_source,
        sample_limit=sample_limit,
        chunk_size_override=chunk_size_override,
        local_window_tokens_override=local_window_tokens_override,
    )
    typhon_summary = evaluate_typhon_v0(
        benchmark_registry=benchmark_registry,
        benchmark_id=benchmark_id,
        family=family,
        output_dir=output_dir / "_typhon_eval_tmp",
        dry_run=True,
        sample_source=sample_source,
        sample_limit=sample_limit,
        chunk_size_override=chunk_size_override,
        local_window_tokens_override=local_window_tokens_override,
    )
    baseline_agg = baseline_summary["aggregate"]
    typhon_agg = typhon_summary["aggregate"]
    comparison = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": benchmark_id or family or "all",
        "run_config": {
            "sample_source": sample_source,
            "sample_limit": sample_limit,
            "chunk_size_override": chunk_size_override,
            "local_window_tokens_override": local_window_tokens_override,
        },
        "baseline": {
            "id": baseline_id,
            "aggregate": baseline_agg,
        },
        "typhon_v0": {
            "aggregate": typhon_agg,
        },
        "deltas": {
            "exact_match_rate": None
            if baseline_agg["exact_match_rate"] is None or typhon_agg["exact_match_rate"] is None
            else round(float(typhon_agg["exact_match_rate"]) - float(baseline_agg["exact_match_rate"]), 4),
            "mean_token_f1": None
            if baseline_agg["mean_token_f1"] is None or typhon_agg["mean_token_f1"] is None
            else round(float(typhon_agg["mean_token_f1"]) - float(baseline_agg["mean_token_f1"]), 4),
            "mean_token_recall": None
            if baseline_agg["mean_token_recall"] is None or typhon_agg["mean_token_recall"] is None
            else round(float(typhon_agg["mean_token_recall"]) - float(baseline_agg["mean_token_recall"]), 4),
        },
    }
    filename_scope = benchmark_id or family or "all"
    return _write_summary(
        summary=comparison,
        output_dir=output_dir,
        filename=f"compare__{baseline_id}__vs__typhon_v0__{filename_scope}{_config_suffix(chunk_size_override=chunk_size_override, local_window_tokens_override=local_window_tokens_override)}.json",
        dry_run=dry_run,
    )


def evaluate_inference_backend(
    *,
    benchmark_registry: BenchmarkRegistry,
    backend_id: str,
    model: str,
    benchmark_id: str | None,
    family: str | None,
    output_dir: Path,
    dry_run: bool,
    sample_source: str = "auto",
    sample_limit: int | None = None,
    max_output_tokens: int = 128,
    temperature: float = 0.0,
    think: str | None = None,
    request_timeout_seconds: float = 300.0,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    artifacts = run_inference_backend(
        benchmark_registry=benchmark_registry,
        backend_id=backend_id,
        model=model,
        benchmark_id=benchmark_id,
        family=family,
        output_dir=output_dir / "_inference_tmp",
        dry_run=True,
        sample_source=sample_source,
        sample_limit=sample_limit,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        think=think,
        request_timeout_seconds=request_timeout_seconds,
        base_url=base_url,
        api_key=api_key,
    )
    filename_scope = benchmark_id or family or "all"
    safe_model = model.replace(":", "_").replace("/", "_")
    summary = summarize_artifacts(
        subject={
            "kind": "inference_backend",
            "id": backend_id,
            "model": model,
            "scope": filename_scope,
        },
        artifacts=artifacts,
        run_config={
            "sample_source": sample_source,
            "sample_limit": sample_limit,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "think": think,
            "request_timeout_seconds": request_timeout_seconds,
            "base_url": base_url,
        },
    )
    return _write_summary(
        summary=summary,
        output_dir=output_dir,
        filename=f"inference__{backend_id}__{safe_model}__{filename_scope}.json",
        dry_run=dry_run,
    )
