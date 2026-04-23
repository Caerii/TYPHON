# System Overview

Last updated: 2026-04-22

## Scope

TYPHON is a research harness for hierarchical memory experiments. The current implementation is useful and testable, but it is still a selection-and-evaluation system rather than a learned memory architecture.

Today the repo can:

- normalize and load benchmark data from fixtures, local slices, and manifest-backed packs
- run a heuristic `typhon_v0` memory-selection pipeline
- run a local exact baseline and a first external Gated DeltaNet baseline
- compare memory strategies on a live hosted model through LM Studio or another OpenAI-compatible backend

It does not yet provide:

- learned fast-weight updates
- real neural episodic memory
- persistent cross-session consolidation
- broad upstream benchmark replication
- broad baseline reproduction coverage

## System Layers

### 1. Control Plane

Primary files:

- `pyproject.toml`
- `src/typhon/cli.py`

Responsibilities:

- package entrypoint
- command dispatch
- orchestration of loaders, runners, and evaluators

Convention:

- all commands should run through `uv`

### 2. Benchmark and Data Plane

Primary files:

- `src/typhon/benchmarks/base.py`
- `src/typhon/benchmarks/registry.py`
- `src/typhon/benchmarks/datasets.py`
- `src/typhon/benchmarks/importer.py`
- `src/typhon/benchmarks/longbench.py`
- `src/typhon/benchmarks/packs.py`
- `configs/benchmarks/`
- `data/benchmarks/`

Responsibilities:

- benchmark registration
- fixture loading
- local sample discovery
- manifest-backed benchmark packs
- upstream LongBench normalization

Preferred local data shape:

- `data/benchmarks/<benchmark_id>/pack.json`
- `data/benchmarks/<benchmark_id>/packs/<pack_id>/samples.jsonl`

### 3. Memory Planning Plane

Primary files:

- `src/typhon/trainers/common.py`
- `src/typhon/trainers/v0.py`
- `src/typhon/policies/heuristic.py`
- `src/typhon/memory/store.py`
- `src/typhon/baselines/local_exact.py`

Responsibilities:

- chunk context into manageable segments
- estimate per-chunk utility signals
- route chunks into layered memory stores
- reconstruct selected context for downstream inference

Current memory layers in code:

- `local_exact`
- `fast_weight`
- `episodic`
- `cross_episode`

Important limitation:

- these are ranked stores and planners, not learned memory modules

### 4. Inference and Evaluation Plane

Primary files:

- `src/typhon/inference/`
- `src/typhon/eval/`
- `src/typhon/baselines/gated_deltanet_fla.py`
- `scripts/wsl/run_gated_deltanet_fla.py`

Responsibilities:

- prompt construction
- local and remote backend invocation
- live evaluation against the same model under multiple memory strategies
- aggregate metric reporting

Current inference backends:

- `extractive_heuristic`
- `lmstudio_local`
- `openai_compatible_http`
- `ollama_local`

Current primary live model path:

- `lmstudio_local` on `qwen3.5-9b-vlm`

## End-to-End Flow

1. The CLI resolves a benchmark, sample source, backend, and runtime parameters.
2. The benchmark layer loads fixtures or local benchmark-pack samples.
3. `typhon_v0` scores chunks using heuristic utility features.
4. The write policy places strong chunks into layered ranked stores.
5. The baseline and TYPHON each build their own selected-context view.
6. The inference layer sends `full_context`, baseline, and TYPHON prompts to the same backend when requested.
7. The evaluation layer computes token-level metrics against available reference answers.
8. Generated artifacts are written under `results/`.

## Top-Level Folder Intent

- [`../adr`](../adr/README.md): stable decisions
- [`../project`](../project/technical-pm-brief.md): current state and execution tracking
- [`../runbooks`](../runbooks/benchmark-packs.md): how to operate the system
- [`../research`](../research/sota-model-backends.md): source-backed implementation guidance
- [`../../configs`](../../configs/README.md): runtime and experiment declarations
- [`../../data`](../../data/README.md): normalized sample assets and imports
- [`../../results`](../../results/README.md): generated outputs only

## Current Next-Step Priority

The repo is past the substrate phase. The next credible progress path is:

1. expand the real LongBench adapter path beyond the smoke slice
2. add the LoCoMo upstream QA adapter
3. improve external baseline coverage
4. only then move heuristic TYPHON memory toward a learned adapter path
