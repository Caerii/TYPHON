from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typhon.benchmarks.registry import BenchmarkRegistry
from typhon.eval.heuristics import score_prediction
from typhon.experiments.budget import BudgetLedger
from typhon.inference.base import GenerationRequest
from typhon.inference.factory import create_backend
from typhon.inference.prompting import build_benchmark_prompt
from typhon.runtime.detect import detect_runtime
from typhon.runtime.profiles import select_runtime_profile


def run_inference_backend(
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
) -> list[dict[str, Any]]:
    specs = benchmark_registry.list_benchmarks(family=family)
    if benchmark_id:
        specs = [benchmark_registry.get(benchmark_id)]
    if not specs:
        return []

    backend = create_backend(backend_id, base_url=base_url, api_key=api_key)
    status = backend.status()
    if not status.available:
        raise RuntimeError(status.message or f"Backend {backend_id} is not available.")

    runtime_profile = select_runtime_profile(detect_runtime())
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    for spec in specs:
        samples = benchmark_registry.load_samples(
            spec,
            sample_source=sample_source,
            limit=sample_limit,
        )
        for sample in samples:
            system_prompt, user_prompt = build_benchmark_prompt(spec, sample)
            generation = backend.generate(
                GenerationRequest(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    benchmark_id=spec.id,
                    question=sample.question,
                    context=sample.context,
                    expected_answer_type=sample.expected_answer_type,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    think=think,
                    request_timeout_seconds=request_timeout_seconds,
                )
            )
            metrics = score_prediction(
                generation.content,
                sample.reference_answer,
                reference_answers=sample.reference_answers,
            )
            artifact = {
                "generated_at": datetime.now(UTC).isoformat(),
                "backend": {
                    "id": backend_id,
                    "model": model,
                    "base_url": base_url,
                    "status": status.to_dict(),
                },
                "benchmark": {
                    "id": spec.id,
                    "name": spec.name,
                    "family": spec.family,
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
                "prompt": {
                    "system": system_prompt,
                    "user": user_prompt,
                },
                "prediction": {
                    "predicted_answer": generation.content,
                    "reference_answer": sample.reference_answer,
                    "reference_answers": list(sample.reference_answers),
                    "metrics": metrics,
                },
                "generation": {
                    "usage": generation.usage,
                    "raw_response": generation.raw_response,
                },
                "budget_ledger": BudgetLedger(
                    inference_flops=None,
                    active_memory_units=None,
                    notes=[
                        f"Backend: {backend_id}",
                        f"Model: {model}",
                    ],
                ).to_dict(),
            }
            suffix = f"__{sample.sample_id}" if len(samples) > 1 or sample.source == "local" else ""
            safe_model = model.replace(":", "_").replace("/", "_")
            artifact_path = output_dir / f"{backend_id}__{safe_model}__{spec.id}{suffix}.json"
            artifact["artifact_path"] = str(artifact_path)
            if not dry_run:
                artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            artifacts.append(artifact)
    return artifacts
