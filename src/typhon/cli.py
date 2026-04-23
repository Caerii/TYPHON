from __future__ import annotations

import argparse
import json
from pathlib import Path

from typhon.baselines.registry import BaselineRegistry
from typhon.baselines.local_exact import run_baseline
from typhon.benchmarks.importer import import_benchmark_pack
from typhon.benchmarks.longbench import import_longbench_pack, load_longbench_import_config
from typhon.benchmarks.registry import BenchmarkRegistry
from typhon.benchmarks.smoke import run_smoke_tests
from typhon.eval.compare import compare_baseline_to_typhon_v0
from typhon.eval.memory_compare import evaluate_memory_strategies
from typhon.eval.memory_suite import evaluate_memory_suite
from typhon.eval.summary import (
    evaluate_baseline,
    evaluate_compare,
    evaluate_inference_backend,
    evaluate_typhon_v0,
)
from typhon.inference.factory import available_backend_ids, create_backend
from typhon.inference.runner import run_inference_backend
from typhon.runtime.detect import detect_runtime
from typhon.runtime.profiles import select_runtime_profile
from typhon.trainers.v0 import run_typhon_v0
from typhon.utils.paths import repo_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TYPHON research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-benchmarks", help="List registered benchmarks.")
    list_parser.add_argument("--family", help="Optional family filter.")

    list_baselines_parser = subparsers.add_parser(
        "list-baselines", help="List implemented or configured baselines."
    )

    list_inference_parser = subparsers.add_parser(
        "list-inference-backends",
        help="List implemented inference backends.",
    )

    inspect_inference_parser = subparsers.add_parser(
        "inspect-inference-backend",
        help="Inspect local availability and status of one inference backend or all backends.",
    )
    inspect_inference_parser.add_argument(
        "--backend",
        help="Optional backend id. If omitted, inspect all backends.",
    )
    inspect_inference_parser.add_argument(
        "--base-url",
        help="Optional base URL override for HTTP-backed inference backends.",
    )
    inspect_inference_parser.add_argument(
        "--api-key",
        help="Optional API key for authenticated OpenAI-compatible endpoints.",
    )

    list_backend_models_parser = subparsers.add_parser(
        "list-backend-models",
        help="List model ids exposed by one inference backend.",
    )
    list_backend_models_parser.add_argument("--backend", required=True, help="Inference backend id to inspect.")
    list_backend_models_parser.add_argument(
        "--base-url",
        help="Optional base URL override for HTTP-backed inference backends.",
    )
    list_backend_models_parser.add_argument(
        "--api-key",
        help="Optional API key for authenticated OpenAI-compatible endpoints.",
    )

    data_parser = subparsers.add_parser(
        "inspect-benchmark-data",
        help="Inspect whether repo-local benchmark samples exist or whether the benchmark will fall back to fixtures.",
    )
    data_parser.add_argument("--benchmark", help="Benchmark id to inspect.")
    data_parser.add_argument("--family", help="Inspect all benchmarks in one family.")

    validate_data_parser = subparsers.add_parser(
        "validate-benchmark-pack",
        help="Validate local benchmark pack manifests or legacy local sample files.",
    )
    validate_data_parser.add_argument("--benchmark", help="Benchmark id to validate.")
    validate_data_parser.add_argument("--family", help="Validate all benchmarks in one family.")

    import_pack_parser = subparsers.add_parser(
        "import-benchmark-pack",
        help="Import an external JSON or JSONL file into a normalized local benchmark pack.",
    )
    import_pack_parser.add_argument("--benchmark", required=True, help="Benchmark id to import into.")
    import_pack_parser.add_argument("--input", required=True, help="Path to the source JSON or JSONL file.")
    import_pack_parser.add_argument("--pack-id", required=True, help="Stable local pack identifier.")
    import_pack_parser.add_argument("--description", default="", help="Short description for the imported pack.")
    import_pack_parser.add_argument(
        "--default-split",
        default="local",
        help="Default split to assign when the source records do not include one.",
    )
    import_pack_parser.add_argument(
        "--default-task-type",
        default="imported_task",
        help="Fallback task type when the source records do not include one.",
    )
    import_pack_parser.add_argument(
        "--default-expected-answer-type",
        default="short_text",
        help="Fallback expected answer type when the source records do not include one.",
    )
    import_pack_parser.add_argument("--sample-id-field", help="Optional source field for sample ids.")
    import_pack_parser.add_argument("--split-field", help="Optional source field for split names.")
    import_pack_parser.add_argument("--task-type-field", help="Optional source field for task type.")
    import_pack_parser.add_argument(
        "--question-field",
        default="question",
        help="Source field containing the question text.",
    )
    import_pack_parser.add_argument(
        "--context-field",
        default="context",
        help="Source field containing the context text.",
    )
    import_pack_parser.add_argument(
        "--expected-answer-type-field",
        help="Optional source field for expected answer type.",
    )
    import_pack_parser.add_argument(
        "--reference-answer-field",
        help="Optional source field for reference answers.",
    )
    import_pack_parser.add_argument(
        "--metadata-field",
        help="Optional source field containing a metadata object.",
    )
    import_pack_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing pack with the same id.",
    )
    import_pack_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the import configuration without writing files.",
    )

    import_longbench_parser = subparsers.add_parser(
        "import-longbench",
        help="Import LongBench or LongBench-E samples from the official Hugging Face dataset using a JSON adapter config.",
    )
    import_longbench_parser.add_argument(
        "--config",
        required=True,
        help="Path to the LongBench adapter config JSON.",
    )
    import_longbench_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing imported LongBench pack with the same id.",
    )
    import_longbench_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the parsed LongBench adapter config instead of importing data.",
    )

    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Run deterministic smoke tests that validate benchmark configs and emit allocation plans.",
    )
    smoke_parser.add_argument("--benchmark", help="Benchmark id to run.")
    smoke_parser.add_argument("--family", help="Run all benchmarks in one family.")
    smoke_parser.add_argument("--chunk-size", type=int, help="Override chunk size in tokens.")
    smoke_parser.add_argument(
        "--local-window-tokens",
        type=int,
        help="Optional local-window override in tokens.",
    )
    smoke_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "smoke"),
        help="Directory for JSON smoke artifacts.",
    )
    smoke_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print artifacts to stdout instead of writing files.",
    )
    smoke_parser.add_argument(
        "--sample-source",
        default="fixture",
        choices=["fixture", "local", "auto"],
        help="Select fixture-only, local-only, or auto sample loading.",
    )
    smoke_parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional per-benchmark sample limit.",
    )

    runtime_parser = subparsers.add_parser(
        "profile-runtime",
        help="Detect local runtime characteristics and select the matching workstation profile.",
    )
    runtime_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "runtime"),
        help="Directory for runtime profile artifacts.",
    )
    runtime_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the runtime profile instead of writing it to disk.",
    )

    run_v0_parser = subparsers.add_parser(
        "run-v0",
        help="Run the first executable TYPHON v0 heuristic memory pipeline and emit artifacts.",
    )
    run_v0_parser.add_argument("--benchmark", help="Benchmark id to run.")
    run_v0_parser.add_argument("--family", help="Run all benchmarks in one family.")
    run_v0_parser.add_argument("--chunk-size", type=int, help="Override chunk size in tokens.")
    run_v0_parser.add_argument(
        "--local-window-tokens",
        type=int,
        help="Optional local-window override in tokens.",
    )
    run_v0_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "typhon"),
        help="Directory for TYPHON v0 artifacts.",
    )
    run_v0_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print artifacts to stdout instead of writing files.",
    )
    run_v0_parser.add_argument(
        "--sample-source",
        default="auto",
        choices=["fixture", "local", "auto"],
        help="Select fixture-only, local-only, or auto sample loading.",
    )
    run_v0_parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional per-benchmark sample limit.",
    )

    baseline_parser = subparsers.add_parser(
        "run-baseline",
        help="Run a baseline artifact pipeline against one benchmark or family.",
    )
    baseline_parser.add_argument("--baseline", required=True, help="Baseline id to run.")
    baseline_parser.add_argument("--benchmark", help="Benchmark id to run.")
    baseline_parser.add_argument("--family", help="Run all benchmarks in one family.")
    baseline_parser.add_argument("--chunk-size", type=int, help="Override chunk size in tokens.")
    baseline_parser.add_argument(
        "--local-window-tokens",
        type=int,
        help="Optional local-window override in tokens.",
    )
    baseline_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "baselines"),
        help="Directory for baseline artifacts.",
    )
    baseline_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print artifacts to stdout instead of writing files.",
    )
    baseline_parser.add_argument(
        "--sample-source",
        default="auto",
        choices=["fixture", "local", "auto"],
        help="Select fixture-only, local-only, or auto sample loading.",
    )
    baseline_parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional per-benchmark sample limit.",
    )

    compare_parser = subparsers.add_parser(
        "compare-v0",
        help="Compare a baseline artifact path against TYPHON v0 on the same benchmarks.",
    )
    compare_parser.add_argument("--baseline", required=True, help="Baseline id to compare.")
    compare_parser.add_argument("--benchmark", help="Benchmark id to run.")
    compare_parser.add_argument("--family", help="Run all benchmarks in one family.")
    compare_parser.add_argument("--chunk-size", type=int, help="Override chunk size in tokens.")
    compare_parser.add_argument(
        "--local-window-tokens",
        type=int,
        help="Optional local-window override in tokens.",
    )
    compare_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "comparisons"),
        help="Directory for comparison artifacts.",
    )
    compare_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print artifacts to stdout instead of writing files.",
    )
    compare_parser.add_argument(
        "--sample-source",
        default="auto",
        choices=["fixture", "local", "auto"],
        help="Select fixture-only, local-only, or auto sample loading.",
    )
    compare_parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional per-benchmark sample limit.",
    )

    evaluate_baseline_parser = subparsers.add_parser(
        "evaluate-baseline",
        help="Run baseline evaluation and write an aggregate summary over scored samples.",
    )
    evaluate_baseline_parser.add_argument("--baseline", required=True, help="Baseline id to evaluate.")
    evaluate_baseline_parser.add_argument("--benchmark", help="Benchmark id to run.")
    evaluate_baseline_parser.add_argument("--family", help="Run all benchmarks in one family.")
    evaluate_baseline_parser.add_argument("--chunk-size", type=int, help="Override chunk size in tokens.")
    evaluate_baseline_parser.add_argument(
        "--local-window-tokens",
        type=int,
        help="Optional local-window override in tokens.",
    )
    evaluate_baseline_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "evaluations"),
        help="Directory for evaluation summaries.",
    )
    evaluate_baseline_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation summary to stdout instead of writing files.",
    )
    evaluate_baseline_parser.add_argument(
        "--sample-source",
        default="auto",
        choices=["fixture", "local", "auto"],
        help="Select fixture-only, local-only, or auto sample loading.",
    )
    evaluate_baseline_parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional per-benchmark sample limit.",
    )

    evaluate_v0_parser = subparsers.add_parser(
        "evaluate-v0",
        help="Run TYPHON v0 evaluation and write an aggregate summary over scored samples.",
    )
    evaluate_v0_parser.add_argument("--benchmark", help="Benchmark id to run.")
    evaluate_v0_parser.add_argument("--family", help="Run all benchmarks in one family.")
    evaluate_v0_parser.add_argument("--chunk-size", type=int, help="Override chunk size in tokens.")
    evaluate_v0_parser.add_argument(
        "--local-window-tokens",
        type=int,
        help="Optional local-window override in tokens.",
    )
    evaluate_v0_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "evaluations"),
        help="Directory for evaluation summaries.",
    )
    evaluate_v0_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation summary to stdout instead of writing files.",
    )
    evaluate_v0_parser.add_argument(
        "--sample-source",
        default="auto",
        choices=["fixture", "local", "auto"],
        help="Select fixture-only, local-only, or auto sample loading.",
    )
    evaluate_v0_parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional per-benchmark sample limit.",
    )

    evaluate_compare_parser = subparsers.add_parser(
        "evaluate-compare",
        help="Run aggregate evaluation comparison between a baseline and TYPHON v0.",
    )
    evaluate_compare_parser.add_argument("--baseline", required=True, help="Baseline id to compare.")
    evaluate_compare_parser.add_argument("--benchmark", help="Benchmark id to run.")
    evaluate_compare_parser.add_argument("--family", help="Run all benchmarks in one family.")
    evaluate_compare_parser.add_argument("--chunk-size", type=int, help="Override chunk size in tokens.")
    evaluate_compare_parser.add_argument(
        "--local-window-tokens",
        type=int,
        help="Optional local-window override in tokens.",
    )
    evaluate_compare_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "evaluations"),
        help="Directory for evaluation summaries.",
    )
    evaluate_compare_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation summary to stdout instead of writing files.",
    )
    evaluate_compare_parser.add_argument(
        "--sample-source",
        default="auto",
        choices=["fixture", "local", "auto"],
        help="Select fixture-only, local-only, or auto sample loading.",
    )
    evaluate_compare_parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional per-benchmark sample limit.",
    )

    run_inference_parser = subparsers.add_parser(
        "run-inference-backend",
        help="Run a model-backed inference backend against one benchmark or family.",
    )
    run_inference_parser.add_argument("--backend", required=True, help="Inference backend id to run.")
    run_inference_parser.add_argument("--model", required=True, help="Model id for the selected backend.")
    run_inference_parser.add_argument("--benchmark", help="Benchmark id to run.")
    run_inference_parser.add_argument("--family", help="Run all benchmarks in one family.")
    run_inference_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "inference"),
        help="Directory for inference artifacts.",
    )
    run_inference_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print artifacts to stdout instead of writing files.",
    )
    run_inference_parser.add_argument(
        "--sample-source",
        default="auto",
        choices=["fixture", "local", "auto"],
        help="Select fixture-only, local-only, or auto sample loading.",
    )
    run_inference_parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional per-benchmark sample limit.",
    )
    run_inference_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=128,
        help="Maximum output tokens to request from the backend.",
    )
    run_inference_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    run_inference_parser.add_argument(
        "--think",
        help='Optional backend thinking mode, for example "low" or "medium" where supported.',
    )
    run_inference_parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=300.0,
        help="Per-request timeout for backend generations.",
    )
    run_inference_parser.add_argument(
        "--base-url",
        help="Optional base URL override for HTTP-backed inference backends.",
    )
    run_inference_parser.add_argument(
        "--api-key",
        help="Optional API key for authenticated OpenAI-compatible endpoints.",
    )

    evaluate_inference_parser = subparsers.add_parser(
        "evaluate-inference-backend",
        help="Run aggregate evaluation for a model-backed inference backend.",
    )
    evaluate_inference_parser.add_argument("--backend", required=True, help="Inference backend id to run.")
    evaluate_inference_parser.add_argument("--model", required=True, help="Model id for the selected backend.")
    evaluate_inference_parser.add_argument("--benchmark", help="Benchmark id to run.")
    evaluate_inference_parser.add_argument("--family", help="Run all benchmarks in one family.")
    evaluate_inference_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "evaluations"),
        help="Directory for evaluation summaries.",
    )
    evaluate_inference_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation summary to stdout instead of writing files.",
    )
    evaluate_inference_parser.add_argument(
        "--sample-source",
        default="auto",
        choices=["fixture", "local", "auto"],
        help="Select fixture-only, local-only, or auto sample loading.",
    )
    evaluate_inference_parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional per-benchmark sample limit.",
    )
    evaluate_inference_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=128,
        help="Maximum output tokens to request from the backend.",
    )
    evaluate_inference_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    evaluate_inference_parser.add_argument(
        "--think",
        help='Optional backend thinking mode, for example "low" or "medium" where supported.',
    )
    evaluate_inference_parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=300.0,
        help="Per-request timeout for backend generations.",
    )
    evaluate_inference_parser.add_argument(
        "--base-url",
        help="Optional base URL override for HTTP-backed inference backends.",
    )
    evaluate_inference_parser.add_argument(
        "--api-key",
        help="Optional API key for authenticated OpenAI-compatible endpoints.",
    )

    evaluate_memory_parser = subparsers.add_parser(
        "evaluate-memory-strategies",
        help="Evaluate full-context, local baseline, and TYPHON-selected context strategies on one live backend.",
    )
    evaluate_memory_parser.add_argument("--backend", required=True, help="Inference backend id to run.")
    evaluate_memory_parser.add_argument("--model", required=True, help="Model id for the selected backend.")
    evaluate_memory_parser.add_argument("--benchmark", help="Benchmark id to run.")
    evaluate_memory_parser.add_argument("--family", help="Run all benchmarks in one family.")
    evaluate_memory_parser.add_argument("--chunk-size", type=int, help="Override chunk size in tokens.")
    evaluate_memory_parser.add_argument(
        "--local-window-tokens",
        type=int,
        help="Optional local-window override in tokens.",
    )
    evaluate_memory_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "evaluations"),
        help="Directory for evaluation summaries.",
    )
    evaluate_memory_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation summary to stdout instead of writing files.",
    )
    evaluate_memory_parser.add_argument(
        "--sample-source",
        default="auto",
        choices=["fixture", "local", "auto"],
        help="Select fixture-only, local-only, or auto sample loading.",
    )
    evaluate_memory_parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional per-benchmark sample limit.",
    )
    evaluate_memory_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=128,
        help="Maximum output tokens to request from the backend.",
    )
    evaluate_memory_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    evaluate_memory_parser.add_argument(
        "--think",
        help='Optional backend thinking mode, for example "low" or "medium" where supported.',
    )
    evaluate_memory_parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=300.0,
        help="Per-request timeout for backend generations.",
    )
    evaluate_memory_parser.add_argument(
        "--base-url",
        help="Optional base URL override for HTTP-backed inference backends.",
    )
    evaluate_memory_parser.add_argument(
        "--api-key",
        help="Optional API key for authenticated OpenAI-compatible endpoints.",
    )

    evaluate_memory_suite_parser = subparsers.add_parser(
        "evaluate-memory-suite",
        help="Run memory-strategy evaluations across multiple models and benchmarks and write one suite summary.",
    )
    evaluate_memory_suite_parser.add_argument("--backend", required=True, help="Inference backend id to run.")
    evaluate_memory_suite_parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model id to include. Repeat for multiple models.",
    )
    evaluate_memory_suite_parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        help="Benchmark id to include. Repeat for multiple benchmarks.",
    )
    evaluate_memory_suite_parser.add_argument("--chunk-size", type=int, help="Override chunk size in tokens.")
    evaluate_memory_suite_parser.add_argument(
        "--local-window-tokens",
        type=int,
        help="Optional local-window override in tokens.",
    )
    evaluate_memory_suite_parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "results" / "evaluations"),
        help="Directory for evaluation summaries.",
    )
    evaluate_memory_suite_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation summary to stdout instead of writing files.",
    )
    evaluate_memory_suite_parser.add_argument(
        "--sample-source",
        default="auto",
        choices=["fixture", "local", "auto"],
        help="Select fixture-only, local-only, or auto sample loading.",
    )
    evaluate_memory_suite_parser.add_argument(
        "--sample-limit",
        type=int,
        help="Optional per-benchmark sample limit.",
    )
    evaluate_memory_suite_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=128,
        help="Maximum output tokens to request from the backend.",
    )
    evaluate_memory_suite_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    evaluate_memory_suite_parser.add_argument(
        "--think",
        help='Optional backend thinking mode, for example "low" or "medium" where supported.',
    )
    evaluate_memory_suite_parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=300.0,
        help="Per-request timeout for backend generations.",
    )
    evaluate_memory_suite_parser.add_argument(
        "--base-url",
        help="Optional base URL override for HTTP-backed inference backends.",
    )
    evaluate_memory_suite_parser.add_argument(
        "--api-key",
        help="Optional API key for authenticated OpenAI-compatible endpoints.",
    )
    return parser


def _handle_list_benchmarks(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry.load()
    specs = registry.list_benchmarks(family=args.family)
    if not specs:
        print("No benchmarks matched the requested filter.")
        return 1

    print("id | family | chunk_size | fixture")
    print("--- | --- | --- | ---")
    for spec in specs:
        print(
            f"{spec.id} | {spec.family} | {spec.default_chunk_size} | {spec.smoke_fixture}"
        )
    return 0


def _handle_list_baselines(_: argparse.Namespace) -> int:
    registry = BaselineRegistry.load()
    specs = registry.list_baselines()
    if not specs:
        print("No baselines are configured.")
        return 1

    print("id | type | retrieval_strategy")
    print("--- | --- | ---")
    for spec in specs:
        print(f"{spec.id} | {spec.type} | {spec.retrieval_strategy}")
    return 0


def _handle_list_inference_backends(_: argparse.Namespace) -> int:
    backend_ids = available_backend_ids()
    if not backend_ids:
        print("No inference backends are implemented.")
        return 1

    print("id")
    print("---")
    for backend_id in backend_ids:
        print(backend_id)
    return 0


def _handle_inspect_inference_backend(args: argparse.Namespace) -> int:
    backend_ids = [args.backend] if args.backend else available_backend_ids()
    statuses: list[dict[str, object]] = []
    for backend_id in backend_ids:
        backend = create_backend(backend_id, base_url=args.base_url, api_key=args.api_key)
        statuses.append(backend.status().to_dict())

    if len(statuses) == 1:
        print(json.dumps(statuses[0], indent=2))
        return 0

    print("backend_id | available | message")
    print("--- | --- | ---")
    for status in statuses:
        print(f"{status['backend_id']} | {status['available']} | {status['message']}")
    return 0


def _handle_list_backend_models(args: argparse.Namespace) -> int:
    backend = create_backend(args.backend, base_url=args.base_url, api_key=args.api_key)
    status = backend.status().to_dict()
    if not status["available"]:
        print(status["message"])
        return 1

    models = status.get("details", {}).get("models", [])
    if not models:
        print("No models reported by the backend.")
        return 0

    print("id | owned_by")
    print("--- | ---")
    for model in models:
        print(f"{model.get('id')} | {model.get('owned_by')}")
    return 0


def _handle_inspect_benchmark_data(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry.load()
    specs = registry.list_benchmarks(family=args.family)
    if args.benchmark:
        specs = [registry.get(args.benchmark)]
    if not specs:
        print("No benchmarks matched the requested filter.")
        return 1

    print("id | source_mode | has_local_data | sample_count")
    print("--- | --- | --- | ---")
    for spec in specs:
        status = registry.get_dataset_status(spec)
        print(
            f"{spec.id} | {status['source_mode']} | {status['has_local_data']} | {status['sample_count']}"
        )
        for path in status["paths"]:
            print(f"path: {path}")
    return 0


def _handle_validate_benchmark_pack(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry.load()
    specs = registry.list_benchmarks(family=args.family)
    if args.benchmark:
        specs = [registry.get(args.benchmark)]
    if not specs:
        print("No benchmarks matched the requested filter.")
        return 1

    print("id | source_mode | is_valid | sample_count | error_count")
    print("--- | --- | --- | --- | ---")
    exit_code = 0
    for spec in specs:
        result = registry.validate_local_data(spec)
        error_count = len(result["errors"])
        print(
            f"{spec.id} | {result['source_mode']} | {result['is_valid']} | {result['sample_count']} | {error_count}"
        )
        for path in result["paths"]:
            print(f"path: {path}")
        for warning in result["warnings"]:
            print(f"warning: {warning}")
        for error in result["errors"]:
            print(f"error: {error}")
        if error_count:
            exit_code = 1
    return exit_code


def _handle_import_benchmark_pack(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry.load()
    spec = registry.get(args.benchmark)
    if args.dry_run:
        preview = {
            "benchmark_id": spec.id,
            "input": args.input,
            "pack_id": args.pack_id,
            "question_field": args.question_field,
            "context_field": args.context_field,
            "reference_answer_field": args.reference_answer_field,
            "default_split": args.default_split,
            "default_task_type": args.default_task_type,
            "default_expected_answer_type": args.default_expected_answer_type,
            "replace": args.replace,
        }
        print(json.dumps(preview, indent=2))
        return 0

    artifact = import_benchmark_pack(
        spec=spec,
        input_path=Path(args.input),
        pack_id=args.pack_id,
        description=args.description,
        default_split=args.default_split,
        default_task_type=args.default_task_type,
        default_expected_answer_type=args.default_expected_answer_type,
        sample_id_field=args.sample_id_field,
        split_field=args.split_field,
        task_type_field=args.task_type_field,
        question_field=args.question_field,
        context_field=args.context_field,
        expected_answer_type_field=args.expected_answer_type_field,
        reference_answer_field=args.reference_answer_field,
        metadata_field=args.metadata_field,
        replace=args.replace,
    )
    print(json.dumps(artifact, indent=2))
    return 0


def _handle_import_longbench(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry.load()
    config = load_longbench_import_config(Path(args.config))
    spec = registry.get(config.benchmark_id)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "benchmark_id": config.benchmark_id,
                    "dataset_id": config.dataset_id,
                    "split": config.split,
                    "pack_id": config.pack_id,
                    "description": config.description,
                    "language": config.language,
                    "max_samples_per_task": config.max_samples_per_task,
                    "tasks": [task.source_name for task in config.tasks if task.include],
                    "length_buckets": [
                        {
                            "name": bucket.name,
                            "min_length": bucket.min_length,
                            "max_length": bucket.max_length,
                        }
                        for bucket in config.length_buckets
                    ],
                    "replace": args.replace,
                },
                indent=2,
            )
        )
        return 0

    artifact = import_longbench_pack(
        spec=spec,
        config=config,
        replace=args.replace,
    )
    print(json.dumps(artifact, indent=2))
    return 0


def _handle_smoke_test(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry.load()
    artifacts = run_smoke_tests(
        registry=registry,
        benchmark_id=args.benchmark,
        family=args.family,
        chunk_size=args.chunk_size,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        sample_source=args.sample_source,
        sample_limit=args.sample_limit,
        local_window_tokens_override=args.local_window_tokens,
    )
    if not artifacts:
        print("No smoke tests ran. Check benchmark id or family filter.")
        return 1

    if args.dry_run:
        print(json.dumps(artifacts, indent=2))
        return 0

    print(f"Wrote {len(artifacts)} smoke artifact(s) to {args.output_dir}")
    for artifact in artifacts:
        print(f"- {artifact['benchmark']['id']}: {artifact['artifact_path']}")
    return 0


def _handle_profile_runtime(args: argparse.Namespace) -> int:
    runtime = detect_runtime()
    profile = select_runtime_profile(runtime)
    payload = profile.to_dict()
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "runtime_profile.json"
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote runtime profile to {artifact_path}")
    return 0


def _handle_run_v0(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry.load()
    artifacts = run_typhon_v0(
        registry=registry,
        benchmark_id=args.benchmark,
        family=args.family,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        sample_source=args.sample_source,
        sample_limit=args.sample_limit,
        chunk_size_override=args.chunk_size,
        local_window_tokens_override=args.local_window_tokens,
    )
    if not artifacts:
        print("No TYPHON v0 runs executed. Check benchmark id or family filter.")
        return 1

    if args.dry_run:
        print(json.dumps(artifacts, indent=2))
        return 0

    print(f"Wrote {len(artifacts)} TYPHON v0 artifact(s) to {args.output_dir}")
    for artifact in artifacts:
        print(f"- {artifact['benchmark']['id']}: {artifact['artifact_path']}")
    return 0


def _handle_run_baseline(args: argparse.Namespace) -> int:
    baseline_registry = BaselineRegistry.load()
    benchmark_registry = BenchmarkRegistry.load()
    artifacts = run_baseline(
        baseline_registry=baseline_registry,
        benchmark_registry=benchmark_registry,
        baseline_id=args.baseline,
        benchmark_id=args.benchmark,
        family=args.family,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        sample_source=args.sample_source,
        sample_limit=args.sample_limit,
        chunk_size_override=args.chunk_size,
        local_window_tokens_override=args.local_window_tokens,
    )
    if not artifacts:
        print("No baseline runs executed. Check baseline id, benchmark id, or family filter.")
        return 1

    if args.dry_run:
        print(json.dumps(artifacts, indent=2))
        return 0

    print(f"Wrote {len(artifacts)} baseline artifact(s) to {args.output_dir}")
    for artifact in artifacts:
        print(
            f"- {artifact['baseline']['id']} on {artifact['benchmark']['id']}: {artifact['artifact_path']}"
        )
    return 0


def _handle_compare_v0(args: argparse.Namespace) -> int:
    baseline_registry = BaselineRegistry.load()
    benchmark_registry = BenchmarkRegistry.load()
    artifacts = compare_baseline_to_typhon_v0(
        baseline_registry=baseline_registry,
        benchmark_registry=benchmark_registry,
        baseline_id=args.baseline,
        benchmark_id=args.benchmark,
        family=args.family,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        sample_source=args.sample_source,
        sample_limit=args.sample_limit,
        chunk_size_override=args.chunk_size,
        local_window_tokens_override=args.local_window_tokens,
    )
    if not artifacts:
        print("No comparison artifacts were generated. Check baseline id, benchmark id, or family filter.")
        return 1

    if args.dry_run:
        print(json.dumps(artifacts, indent=2))
        return 0

    print(f"Wrote {len(artifacts)} comparison artifact(s) to {args.output_dir}")
    for artifact in artifacts:
        print(f"- {artifact['benchmark']['id']}: {artifact['artifact_path']}")
    return 0


def _handle_evaluate_baseline(args: argparse.Namespace) -> int:
    baseline_registry = BaselineRegistry.load()
    benchmark_registry = BenchmarkRegistry.load()
    summary = evaluate_baseline(
        baseline_registry=baseline_registry,
        benchmark_registry=benchmark_registry,
        baseline_id=args.baseline,
        benchmark_id=args.benchmark,
        family=args.family,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        sample_source=args.sample_source,
        sample_limit=args.sample_limit,
        chunk_size_override=args.chunk_size,
        local_window_tokens_override=args.local_window_tokens,
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"Wrote baseline evaluation summary to {summary['artifact_path']}")
    return 0


def _handle_evaluate_v0(args: argparse.Namespace) -> int:
    benchmark_registry = BenchmarkRegistry.load()
    summary = evaluate_typhon_v0(
        benchmark_registry=benchmark_registry,
        benchmark_id=args.benchmark,
        family=args.family,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        sample_source=args.sample_source,
        sample_limit=args.sample_limit,
        chunk_size_override=args.chunk_size,
        local_window_tokens_override=args.local_window_tokens,
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"Wrote TYPHON v0 evaluation summary to {summary['artifact_path']}")
    return 0


def _handle_evaluate_compare(args: argparse.Namespace) -> int:
    baseline_registry = BaselineRegistry.load()
    benchmark_registry = BenchmarkRegistry.load()
    summary = evaluate_compare(
        baseline_registry=baseline_registry,
        benchmark_registry=benchmark_registry,
        baseline_id=args.baseline,
        benchmark_id=args.benchmark,
        family=args.family,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        sample_source=args.sample_source,
        sample_limit=args.sample_limit,
        chunk_size_override=args.chunk_size,
        local_window_tokens_override=args.local_window_tokens,
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"Wrote evaluation comparison summary to {summary['artifact_path']}")
    return 0


def _handle_run_inference_backend(args: argparse.Namespace) -> int:
    benchmark_registry = BenchmarkRegistry.load()
    try:
        artifacts = run_inference_backend(
            benchmark_registry=benchmark_registry,
            backend_id=args.backend,
            model=args.model,
            benchmark_id=args.benchmark,
            family=args.family,
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            sample_source=args.sample_source,
            sample_limit=args.sample_limit,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            think=args.think,
            request_timeout_seconds=args.request_timeout_seconds,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1

    if not artifacts:
        print("No inference runs executed. Check backend id, model, benchmark id, or family filter.")
        return 1

    if args.dry_run:
        print(json.dumps(artifacts, indent=2))
        return 0

    print(f"Wrote {len(artifacts)} inference artifact(s) to {args.output_dir}")
    for artifact in artifacts:
        print(
            f"- {artifact['backend']['id']}:{artifact['backend']['model']} on {artifact['benchmark']['id']}: {artifact['artifact_path']}"
        )
    return 0


def _handle_evaluate_inference_backend(args: argparse.Namespace) -> int:
    benchmark_registry = BenchmarkRegistry.load()
    try:
        summary = evaluate_inference_backend(
            benchmark_registry=benchmark_registry,
            backend_id=args.backend,
            model=args.model,
            benchmark_id=args.benchmark,
            family=args.family,
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            sample_source=args.sample_source,
            sample_limit=args.sample_limit,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            think=args.think,
            request_timeout_seconds=args.request_timeout_seconds,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Wrote inference evaluation summary to {summary['artifact_path']}")
    return 0


def _handle_evaluate_memory_strategies(args: argparse.Namespace) -> int:
    baseline_registry = BaselineRegistry.load()
    benchmark_registry = BenchmarkRegistry.load()
    try:
        summary = evaluate_memory_strategies(
            baseline_registry=baseline_registry,
            benchmark_registry=benchmark_registry,
            backend_id=args.backend,
            model=args.model,
            benchmark_id=args.benchmark,
            family=args.family,
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            sample_source=args.sample_source,
            sample_limit=args.sample_limit,
            chunk_size_override=args.chunk_size,
            local_window_tokens_override=args.local_window_tokens,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            think=args.think,
            request_timeout_seconds=args.request_timeout_seconds,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Wrote memory strategy evaluation summary to {summary['artifact_path']}")
    return 0


def _handle_evaluate_memory_suite(args: argparse.Namespace) -> int:
    baseline_registry = BaselineRegistry.load()
    benchmark_registry = BenchmarkRegistry.load()
    try:
        summary = evaluate_memory_suite(
            baseline_registry=baseline_registry,
            benchmark_registry=benchmark_registry,
            backend_id=args.backend,
            models=args.model,
            benchmarks=args.benchmark,
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            sample_source=args.sample_source,
            sample_limit=args.sample_limit,
            chunk_size_override=args.chunk_size,
            local_window_tokens_override=args.local_window_tokens,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            think=args.think,
            request_timeout_seconds=args.request_timeout_seconds,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Wrote memory suite summary to {summary['artifact_path']}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "list-benchmarks":
        return _handle_list_benchmarks(args)
    if args.command == "list-baselines":
        return _handle_list_baselines(args)
    if args.command == "list-inference-backends":
        return _handle_list_inference_backends(args)
    if args.command == "inspect-inference-backend":
        return _handle_inspect_inference_backend(args)
    if args.command == "list-backend-models":
        return _handle_list_backend_models(args)
    if args.command == "inspect-benchmark-data":
        return _handle_inspect_benchmark_data(args)
    if args.command == "validate-benchmark-pack":
        return _handle_validate_benchmark_pack(args)
    if args.command == "import-benchmark-pack":
        return _handle_import_benchmark_pack(args)
    if args.command == "import-longbench":
        return _handle_import_longbench(args)
    if args.command == "smoke-test":
        return _handle_smoke_test(args)
    if args.command == "profile-runtime":
        return _handle_profile_runtime(args)
    if args.command == "run-v0":
        return _handle_run_v0(args)
    if args.command == "run-baseline":
        return _handle_run_baseline(args)
    if args.command == "compare-v0":
        return _handle_compare_v0(args)
    if args.command == "evaluate-baseline":
        return _handle_evaluate_baseline(args)
    if args.command == "evaluate-v0":
        return _handle_evaluate_v0(args)
    if args.command == "evaluate-compare":
        return _handle_evaluate_compare(args)
    if args.command == "run-inference-backend":
        return _handle_run_inference_backend(args)
    if args.command == "evaluate-inference-backend":
        return _handle_evaluate_inference_backend(args)
    if args.command == "evaluate-memory-strategies":
        return _handle_evaluate_memory_strategies(args)
    if args.command == "evaluate-memory-suite":
        return _handle_evaluate_memory_suite(args)
    parser.error(f"Unknown command: {args.command}")
    return 2
