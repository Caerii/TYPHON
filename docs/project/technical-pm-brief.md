# Technical PM Brief

This is the current-state briefing for a technical project manager reviewing TYPHON as of 2026-04-22.

## 1. What TYPHON Is Right Now

TYPHON is currently a research harness and evaluation substrate for hierarchical memory experiments. It is not yet a learned state-of-the-art memory model.

What exists today:

- a Python package and CLI named `typhon`
- a benchmark registry and local data layer
- manifest-backed benchmark packs
- one upstream benchmark adapter for LongBench English tasks
- a heuristic `typhon_v0` memory planner
- a local exact baseline
- a WSL-backed Gated DeltaNet baseline via FLA
- model-backed evaluation against LM Studio on `qwen3.5-9b-vlm`
- local benchmark slices across five benchmark families
- aggregate evaluation artifacts and leaderboards

What does not exist yet:

- learned fast-weight updates
- real neural episodic memory
- persistent cross-session memory
- upstream adapters for the other benchmark families
- broad external baseline coverage
- matched-budget SOTA reproduction claims

## 2. Project Goal

The intended research direction is a hierarchical memory system for long-context and cross-episode tasks:

- local exact recall
- fast adaptive memory
- sparse episodic memory
- slow cross-episode memory

The current repo approximates that idea with heuristic chunk scoring and layered context selection so we can test whether the retrieval policy itself is useful before implementing heavier model-side memory mechanisms.

## 3. Current System Architecture

See the [System Overview](../architecture/system-overview.md) for the full component map. At a high level the repo has four layers:

- control plane: package entrypoint and CLI orchestration
- benchmark and data plane: fixtures, local packs, and adapters
- memory planning plane: chunk scoring, write policy, and layered ranked stores
- inference and evaluation plane: backends, prompting, metrics, and artifacts

## 4. Benchmarks In The Repo

Configured benchmark families:

- LongBench v2
- LongBench
- BABILong
- ZeroSCROLLS
- LoCoMo
- LoCoMo-Plus
- Evo-Memory
- MemoryBench

Current live local slices:

- `longbench_v2`: 4 local samples
- `locomo`: 2 local samples
- `locomo_plus`: 3 local samples
- `memorybench`: 2 imported local-pack samples
- `evo_memory`: 2 imported local-pack samples
- `longbench`: 28 imported LongBench English smoke samples across 14 tasks

Important limitation:

- only LongBench currently has a dedicated upstream adapter
- the other benchmark families are still curated local slices or imported seed packs

## 5. Baselines And Model Paths

Implemented baselines:

- `attention_baseline`
- `gated_deltanet_fla`

Working live backend:

- `lmstudio_local`

Prepared backend surfaces:

- `openai_compatible_http`
- `ollama_local`
- WSL runtime scripts for vLLM and SGLang

Pending stronger baselines:

- PERK
- MesaNet
- TTT-E2E
- qTTT
- MemoryLLM
- M+
- FwPKM
- GradMem

## 6. Current Empirical Status

Primary current suite artifact:

- `results/evaluations/memory_suite__lmstudio_local__qwen3.5-9b-vlm__longbench_v2__locomo__locomo_plus__memorybench__evo_memory.json`

Current five-benchmark rollup on `qwen3.5-9b-vlm`:

- mean TYPHON token F1: `0.5034`
- mean baseline token F1: `0.2183`
- mean full-context token F1: `0.5559`
- mean TYPHON vs baseline delta: `+0.2851`

Important caveat:

- those positive results are on curated local slices and seed packs, not on large official benchmark reproductions

Current LongBench status is materially weaker:

- LongBench adapter is working end to end
- imported smoke pack contains 28 samples across 14 English tasks
- first offline compare on the smoke slice showed TYPHON roughly tied with the local baseline
- first live LM Studio compare on the imported sample had baseline ahead of heuristic TYPHON, while full-context failed due to an LM Studio HTTP 400 on a long prompt

Interpretation:

- the harness is real
- the current heuristic memory policy is useful on curated local slices
- the official benchmark path is already exposing where the heuristics stop being good enough

## 7. What Is Proven Vs What Is Hypothesis

Proven in this repo:

- the benchmark and evaluation substrate is usable
- the repo can compare multiple memory strategies on the same live model
- the heuristic context-selection policy can beat a constrained local baseline on several local slices
- the repo can ingest one real upstream benchmark into its normalized local format
- the repo can run one real external baseline path through WSL

Not yet proven:

- that learned TYPHON memory beats heuristic selection
- that TYPHON beats strong baselines on official benchmark slices
- that the current local gains generalize broadly
- that the current architecture is competitive with published memory systems under matched compute

## 8. Major Gaps

Primary open gaps:

- only one upstream benchmark adapter exists
- only one external baseline wrapper exists
- no learned memory module exists yet
- no training loop exists for model-side memory updates
- no official benchmark reproduction claims are credible yet

## 9. Technical Risks

Primary risks:

- overfitting to curated local slices
- mistaking prompt compression gains for true memory learning
- underestimating the difficulty jump from local slices to official benchmarks
- runtime friction on a 10 GB workstation
- allowing docs and folder layout to drift into ambiguity

## 10. Recommended Next Steps

Ordered next steps:

1. expand the LongBench adapter beyond the smoke slice and add length-bucketed evaluation
2. add the LoCoMo QA upstream adapter
3. improve external baseline coverage beyond the first Gated DeltaNet path
4. only then move TYPHON from heuristic memory selection to a minimal learned adapter path

## 11. Documents The PM Should Use

- [System Overview](../architecture/system-overview.md)
- [ADR Index](../adr/README.md)
- [Experiment Matrix](experiment-matrix.md)
- [Cursor Handoff](cursor-handoff.md)
- [Benchmark Packs Runbook](../runbooks/benchmark-packs.md)
- [WSL Runtime Runbook](../runbooks/wsl-runtime.md)
