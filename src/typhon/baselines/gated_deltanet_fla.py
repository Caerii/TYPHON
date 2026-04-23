from __future__ import annotations

import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from typhon.baselines.base import BaselineSpec
from typhon.benchmarks.base import BenchmarkSample, BenchmarkSpec
from typhon.eval.heuristics import score_prediction
from typhon.experiments.budget import BudgetLedger
from typhon.inference.prompting import build_benchmark_prompt
from typhon.runtime.base import RuntimeProfile
from typhon.utils.paths import repo_root
from typhon.utils.wsl import windows_path_to_wsl


def _int_setting(settings: dict[str, Any], key: str, default: int) -> int:
    return int(settings.get(key, default))


def _float_setting(settings: dict[str, Any], key: str, default: float) -> float:
    return float(settings.get(key, default))


def _bridge_dir() -> Path:
    path = repo_root() / "results" / "_wsl_bridge"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _invoke_wsl_generation(
    *,
    baseline: BaselineSpec,
    benchmark: BenchmarkSpec,
    samples: list[BenchmarkSample],
) -> dict[str, Any]:
    settings = baseline.settings
    request_samples: list[dict[str, Any]] = []
    for sample in samples:
        system_prompt, user_prompt = build_benchmark_prompt(benchmark, sample)
        request_samples.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "expected_answer_type": sample.expected_answer_type,
                "reference_answer": sample.reference_answer,
                "reference_answers": list(sample.reference_answers),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

    request_payload = {
        "baseline_id": baseline.id,
        "benchmark_id": benchmark.id,
        "model": {
            "checkpoint_id": settings.get("checkpoint_id"),
            "tokenizer_id": settings.get("tokenizer_id"),
            "dtype": settings.get("dtype", "bfloat16"),
            "device": settings.get("device", "cuda"),
            "max_input_tokens": _int_setting(settings, "max_input_tokens", 2048),
            "max_output_tokens": _int_setting(settings, "max_output_tokens", 64),
            "temperature": _float_setting(settings, "temperature", 0.0),
            "trust_remote_code": bool(settings.get("trust_remote_code", True)),
            "checkpoint_conversion": dict(settings.get("checkpoint_conversion", {})),
        },
        "samples": request_samples,
    }

    bridge_dir = _bridge_dir()
    request_path = bridge_dir / f"{baseline.id}__{benchmark.id}__{uuid4().hex}__request.json"
    response_path = bridge_dir / f"{baseline.id}__{benchmark.id}__{uuid4().hex}__response.json"
    request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")

    repo_wsl = windows_path_to_wsl(repo_root())
    request_wsl = windows_path_to_wsl(request_path)
    response_wsl = windows_path_to_wsl(response_path)
    wsl_python = str(settings.get("wsl_python", ".venv-wsl-vllm/bin/python"))
    script_path = "scripts/wsl/run_gated_deltanet_fla.py"
    timeout_seconds = _int_setting(settings, "timeout_seconds", 1800)
    command = (
        f"cd {shlex.quote(repo_wsl)} && "
        f"{shlex.quote(wsl_python)} {shlex.quote(script_path)} "
        f"--request-file {shlex.quote(request_wsl)} "
        f"--response-file {shlex.quote(response_wsl)}"
    )

    try:
        completed = subprocess.run(
            ["wsl.exe", "bash", "-lc", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Gated DeltaNet FLA WSL runner timed out after {timeout_seconds}s."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "Gated DeltaNet FLA WSL runner failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not response_path.exists():
        raise RuntimeError(
            "Gated DeltaNet FLA WSL runner completed without writing a response file."
        )

    try:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
    return payload


def run_gated_deltanet_fla_baseline(
    *,
    baseline: BaselineSpec,
    benchmark: BenchmarkSpec,
    samples: list[BenchmarkSample],
    runtime_profile: RuntimeProfile,
    output_dir: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    response = _invoke_wsl_generation(
        baseline=baseline,
        benchmark=benchmark,
        samples=samples,
    )
    if response.get("status") != "ok":
        raise RuntimeError(
            "Gated DeltaNet FLA WSL runner returned an error response. "
            f"{response.get('error', 'No error message provided.')}"
        )
    sample_results = {
        item["sample_id"]: item for item in response.get("samples", [])
    }
    runtime_details = dict(response.get("runtime", {}))

    artifacts: list[dict[str, Any]] = []
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        result = sample_results.get(
            sample.sample_id,
            {
                "predicted_answer": "",
                "usage": {},
                "truncated_prompt": False,
                "input_token_count": None,
                "retained_input_tokens": None,
                "error": "Missing sample result from WSL runner.",
            },
        )
        predicted_answer = str(result.get("predicted_answer", "")).strip()
        metrics = score_prediction(
            predicted_answer,
            sample.reference_answer,
            reference_answers=sample.reference_answers,
        )
        system_prompt, user_prompt = build_benchmark_prompt(benchmark, sample)
        artifact = {
            "generated_at": datetime.now(UTC).isoformat(),
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
            "execution_runtime": runtime_details,
            "prompt": {
                "system": system_prompt,
                "user": user_prompt,
            },
            "prediction": {
                "predicted_answer": predicted_answer,
                "reference_answer": sample.reference_answer,
                "reference_answers": list(sample.reference_answers),
                "metrics": metrics,
            },
            "generation": {
                "usage": result.get("usage", {}),
                "truncated_prompt": bool(result.get("truncated_prompt", False)),
                "input_token_count": result.get("input_token_count"),
                "retained_input_tokens": result.get("retained_input_tokens"),
                "error": result.get("error"),
            },
            "limitations": [
                "Runs inside WSL rather than the Windows uv environment.",
                "Uses a finite model context window and truncates prompts when the sample exceeds that limit.",
                "Default checkpoint is a third-party community model because NVIDIA does not release pretrained Gated DeltaNet weights.",
            ],
            "budget_ledger": BudgetLedger(
                proxy_token_ops=None,
                active_memory_units=None,
                notes=[
                    f"Checkpoint: {baseline.settings.get('checkpoint_id')}",
                    f"Runtime target: {baseline.settings.get('runtime_target', 'wsl')}",
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
