# TYPHON

Hierarchical memory research harness for long-context and cross-episode LLM evaluation.

TYPHON is currently a structured research substrate, not yet a learned state-of-the-art memory model. The repo already supports:

- benchmark registration and local benchmark-pack ingestion
- a heuristic `typhon_v0` memory-selection pipeline
- a local exact baseline plus a WSL-backed Gated DeltaNet baseline
- model-backed evaluation through LM Studio and OpenAI-compatible servers
- reproducible configs, runbooks, and experiment tracking

## Start Here

- [Documentation Index](docs/README.md)
- [System Overview](docs/architecture/system-overview.md)
- [Technical PM Brief](docs/project/technical-pm-brief.md)
- [Experiment Matrix](docs/project/experiment-matrix.md)
- [Benchmark Packs Runbook](docs/runbooks/benchmark-packs.md)
- [WSL Runtime Runbook](docs/runbooks/wsl-runtime.md)

## Repo Layout

- [`src/typhon`](src/typhon): package code, CLI, benchmarks, baselines, memory logic, inference, evaluation
- [`configs`](configs/README.md): benchmark, runtime, baseline, and live-eval configuration
- [`data`](data/README.md): benchmark packs, imports, and normalized local sample assets
- [`results`](results/README.md): generated artifacts and evaluation outputs
- [`scripts`](scripts/README.md): wrapper scripts, mostly evaluation and WSL runtime helpers
- [`docs`](docs/README.md): architecture docs, ADRs, project state, runbooks, research notes, archive
- [`third_party`](third_party/README.md): isolated external code or cloned repos
- [`notebooks`](notebooks/README.md): exploratory analysis only

## Common Commands

```powershell
uv run typhon list-benchmarks
uv run typhon list-baselines
uv run typhon validate-benchmark-pack --benchmark longbench
uv run typhon evaluate-memory-suite --backend lmstudio_local --model qwen3.5-9b-vlm --benchmark longbench_v2 --benchmark locomo --benchmark locomo_plus --benchmark memorybench --benchmark evo_memory --sample-source local --chunk-size 24 --local-window-tokens 24 --request-timeout-seconds 600
uv run typhon run-baseline --baseline gated_deltanet_fla --benchmark longbench --sample-source local --sample-limit 1
```

## Working Conventions

- Use `uv run ...` for repo commands and `uv lock` after dependency changes.
- Keep major architectural or process decisions in [ADRs](docs/adr/README.md), not in ad hoc notes.
- Put new docs into the structured docs tree. Do not add new flat markdown files at the repo root or `docs/` root unless they are indexes.
- Treat `results/` as generated output, not hand-maintained source material.

## Current Status

The repo has one upstream benchmark adapter live today:

- `LongBench` English-task import via Hugging Face into manifest-backed local packs

The current implemented baselines are:

- `attention_baseline`
- `gated_deltanet_fla`

The current primary live model path is:

- `lmstudio_local` with `qwen3.5-9b-vlm`

The canonical project state trackers are:

- [Experiment Matrix](docs/project/experiment-matrix.md)
- [Technical PM Brief](docs/project/technical-pm-brief.md)
- [Cursor Handoff](docs/project/cursor-handoff.md)
