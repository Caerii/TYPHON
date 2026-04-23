# Cursor Handoff

Last verified: 2026-04-22
Audience: the next Codex or Cursor session working locally in this repo

## Read In This Order

1. [Documentation Index](../README.md)
2. [System Overview](../architecture/system-overview.md)
3. [ADR Index](../adr/README.md)
4. [Experiment Matrix](experiment-matrix.md)
5. [Technical PM Brief](technical-pm-brief.md)

## Current State

TYPHON is beyond the empty-scaffold phase. The repo already contains:

- a working CLI and Python package
- benchmark registry and manifest-backed pack infrastructure
- a dedicated LongBench English adapter
- a heuristic `typhon_v0` memory-selection path
- a local exact baseline
- a WSL-backed Gated DeltaNet baseline via FLA
- LM Studio-backed live evaluation on `qwen3.5-9b-vlm`

What it still does not contain:

- a learned TYPHON memory module
- a LoCoMo upstream adapter
- broad baseline coverage
- credible official-benchmark reproduction claims

## Current Priority Order

1. Expand the LongBench adapter beyond the smoke slice.
2. Add the LoCoMo QA upstream adapter.
3. Improve external baseline coverage.
4. Move TYPHON from heuristic stores toward a minimal learned adapter path.

## Ground Rules

- Use `uv` for repo commands.
- Keep architectural and workflow decisions in ADRs.
- Update the experiment matrix when a meaningful benchmark or baseline milestone lands.
- Keep docs in the structured tree under `docs/`; do not add new flat docs at the top level.
- Keep `results/` as generated output only.

## Canonical Commands

Benchmark inspection:

```powershell
uv run typhon list-benchmarks
uv run typhon inspect-benchmark-data --benchmark longbench
uv run typhon validate-benchmark-pack --benchmark longbench
```

LongBench adapter path:

```powershell
uv run --extra adapter_hf typhon import-longbench --config configs/benchmarks/adapters/longbench_english_smoke.json --replace
uv run typhon smoke-test --benchmark longbench --sample-source local --sample-limit 10
```

Baseline and live evaluation:

```powershell
uv run typhon run-baseline --baseline gated_deltanet_fla --benchmark longbench --sample-source local --sample-limit 1
uv run typhon evaluate-memory-suite --backend lmstudio_local --model qwen3.5-9b-vlm --benchmark longbench_v2 --benchmark locomo --benchmark locomo_plus --benchmark memorybench --benchmark evo_memory --sample-source local --chunk-size 24 --local-window-tokens 24 --request-timeout-seconds 600
```

## Immediate Questions To Resolve

- How large should the next committed LongBench slice be?
- Should the next baseline effort stay with stronger Gated DeltaNet checkpoints or move to a different published baseline first?
- What is the minimum learned adapter experiment that is meaningful on a 10 GB GPU?
