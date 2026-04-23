# Technical PM Brief

This document is the current-state briefing for a technical project manager reviewing TYPHON as of 2026-04-22.

## 1. What TYPHON Is Right Now

TYPHON is currently a research substrate and evaluation harness for hierarchical memory selection, not yet a learned SOTA memory model.

What exists today:

- a Python package and CLI named `typhon`
- a benchmark registry and local data layer
- a heuristic TYPHON v0 memory planner
- a local-only baseline
- model-backed evaluation against LM Studio on `qwen3.5-9b-vlm`
- local benchmark slices across five benchmark families
- aggregate evaluation artifacts and leaderboards

What does not exist yet:

- learned fast-weight updates
- real neural episodic memory
- persistent cross-session memory
- upstream benchmark adapters for official datasets
- wrapped external SOTA baselines such as Gated DeltaNet, PERK, MesaNet, MemoryLLM, or FwPKM

## 2. Project Goal

The intended research direction is a hierarchical memory system for long-context and cross-episode tasks:

- local exact recall
- fast adaptive memory
- sparse episodic memory
- slow cross-episode memory

The current repo approximates that idea with heuristic chunk scoring and layered context selection so we can test whether the retrieval policy itself is useful before implementing heavier model-side memory mechanisms.

## 3. Current System Architecture

The repo has four major layers.

### 3.1 Control Plane

Files:

- `pyproject.toml`
- `src/typhon/cli.py`

Responsibilities:

- package entrypoint
- command routing
- orchestration of benchmark loading, artifact generation, and evaluation

Tooling convention:

- all repo commands are intended to run through `uv`

### 3.2 Benchmark and Data Plane

Files:

- `src/typhon/benchmarks/base.py`
- `src/typhon/benchmarks/registry.py`
- `src/typhon/benchmarks/datasets.py`
- `src/typhon/benchmarks/packs.py`
- `src/typhon/benchmarks/importer.py`
- `configs/benchmarks/*.json`
- `data/benchmarks/*`

Responsibilities:

- benchmark metadata registry
- fixture loading for smoke tests
- repo-local sample discovery
- preferred manifest-backed benchmark pack format
- import and validation of normalized local benchmark packs

Current preferred local format:

- `data/benchmarks/<benchmark_id>/pack.json`
- `data/benchmarks/<benchmark_id>/packs/<pack_id>/samples.jsonl`

Legacy support remains for:

- `data/benchmarks/<benchmark_id>/samples.jsonl`
- `data/benchmarks/<benchmark_id>/samples.json`

### 3.3 Memory Planning Plane

Files:

- `src/typhon/trainers/common.py`
- `src/typhon/trainers/v0.py`
- `src/typhon/policies/heuristic.py`
- `src/typhon/memory/store.py`
- `src/typhon/baselines/local_exact.py`

Responsibilities:

- chunk long context into segments
- compute heuristic features per chunk
- decide which chunks belong in which memory layer
- build selected-context outputs for TYPHON v0
- build selected-context outputs for the local baseline

Current memory layers in code:

- `local_exact`
- `fast_weight`
- `episodic`
- `cross_episode`

Important technical point:

- these layers are currently ranked stores, not learned memory modules

### 3.4 Inference and Evaluation Plane

Files:

- `src/typhon/inference/*`
- `src/typhon/eval/*`
- `results/evaluations/*`

Responsibilities:

- connect to local or remote inference backends
- compare full context, local baseline, and TYPHON-selected context on the same model
- aggregate metrics and emit JSON artifacts

Current backends:

- `extractive_heuristic`
- `ollama_local`
- `lmstudio_local`
- `openai_compatible_http`

Primary live backend on this workstation:

- `lmstudio_local`
- model: `qwen3.5-9b-vlm`

## 4. End-to-End Data Flow

One evaluation run currently works like this:

1. CLI command selects benchmark, backend, model, and runtime parameters.
2. Benchmark registry resolves either fixtures or local benchmark samples.
3. TYPHON v0 chunks the context and scores each chunk.
4. The heuristic write policy routes strong chunks into layered stores.
5. TYPHON merges selected records into a smaller context view.
6. The baseline builds its own smaller context view from only the recent accessible window.
7. The inference layer sends three prompts to the same model:
   - full context
   - local baseline context
   - TYPHON-selected context
8. Evaluation computes token-level F1, recall, and exact-match style metrics against local reference answers.
9. JSON artifacts are written under `results/evaluations/`.

## 5. What the Heuristic Planner Actually Uses

Per chunk, the current planner estimates:

- question-term overlap
- novelty ratio
- numeric-signal presence
- latent conversational-constraint flag
- family persistence bias

These are combined into:

- `surprise`
- `gradient_norm`
- `predicted_utility`
- `novelty`

Then the write policy applies simple thresholds:

- `fast_weight` when utility or adaptation signal is high
- `episodic` when novelty or surprise is high
- `cross_episode` when the task family implies persistence and the chunk looks important

This is useful because it lets us test the structure of memory allocation before implementing expensive learned updates.

## 6. Benchmarks in the Repo

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

Important limitation:

- these are curated local slices, not official upstream benchmark mirrors

## 7. Baselines and Model Paths

### 7.1 Implemented Baseline

Implemented:

- `attention_baseline`

Behavior:

- sees only the recent accessible local window
- ranks chunks by overlap and light recency bias
- does not use fast-weight, episodic, or cross-episode memory

### 7.2 Pending SOTA Baselines

Configured but not yet wrapped:

- Gated DeltaNet
- PERK
- MesaNet
- TTT-E2E
- qTTT
- MemoryLLM
- M+
- FwPKM
- GradMem

### 7.3 Model Serving

Working:

- LM Studio local OpenAI-compatible serving

Partially prepared:

- WSL + vLLM
- WSL + SGLang
- Ollama local path

## 8. Current Empirical Status

Primary current suite artifact:

- `results/evaluations/memory_suite__lmstudio_local__qwen3.5-9b-vlm__longbench_v2__locomo__locomo_plus__memorybench__evo_memory.json`

Current five-benchmark rollup on `qwen3.5-9b-vlm`:

- mean TYPHON token F1: `0.5034`
- mean baseline token F1: `0.2183`
- mean full-context token F1: `0.5559`
- mean TYPHON vs baseline delta: `+0.2851`

Per-benchmark current deltas:

- `locomo`: `+0.4335`
- `longbench_v2`: `+0.3243`
- `locomo_plus`: `+0.2889`
- `memorybench`: `+0.2111`
- `evo_memory`: `+0.1679`

Interpretation:

- the current TYPHON policy consistently beats the constrained local baseline
- on some conversational slices, TYPHON matches full-context performance
- on continual-learning and streaming slices, TYPHON still improves over the baseline but remains behind full context

## 9. What Is Proven vs What Is Hypothesis

### Proven in This Repo

- the heuristic context-selection policy is useful under constrained recent-context access
- the repo can compare multiple context strategies on the same live model
- the data and evaluation surface now scales beyond hand-edited root-level sample files
- local benchmark coverage now spans long-context, conversational, continual-learning, and streaming families

### Not Yet Proven

- that a learned TYPHON memory module will outperform current heuristic selection
- that the architecture will beat official SOTA baselines under matched compute
- that the current local slices reflect official benchmark difficulty
- that the current retrieval policy generalizes without heavy prompt sensitivity

## 10. Major Gaps

The main open gaps are:

- no upstream dataset ingestion adapters
- no learned memory modules
- no external baseline wrappers
- no training loop for model-side memory updates
- no official benchmark reproduction claims
- no cost-normalized comparison against paper baselines

## 11. Technical Risks

Primary risks:

- overfitting conclusions to small curated local slices
- mistaking prompt compression gains for true memory learning
- evaluating against incomplete reference answers
- Windows local runtime friction around `uv` cache permissions and backend orchestration
- VRAM constraints on the 10 GB GPU limiting model and server choices

## 12. Why the Current System Works

The current system works because TYPHON reconstructs a better prompt than the local-only baseline.

The baseline usually loses earlier facts because it only has recent context.

TYPHON often restores:

- the early rule
- the latent user preference
- the prior lesson from a previous episode

That is why it often approaches full-context behavior despite using a smaller selected context.

This is an important result, but it is still retrieval-and-selection behavior, not learned memory in the stronger research sense.

## 13. Commands a PM Should Know

Inspection:

- `uv run typhon list-benchmarks`
- `uv run typhon inspect-benchmark-data --benchmark memorybench`
- `uv run typhon validate-benchmark-pack --benchmark memorybench`
- `uv run typhon list-inference-backends`
- `uv run typhon list-backend-models --backend lmstudio_local`

Import and normalize data:

- `uv run typhon import-benchmark-pack --benchmark memorybench --input data/imports/memorybench_seed_v1.jsonl --pack-id seed_v1 --description "Seed local MemoryBench pack" --sample-id-field id --task-type-field kind --question-field prompt --context-field history --reference-answer-field gold --metadata-field details`

Live evaluation:

- `uv run typhon evaluate-memory-strategies --backend lmstudio_local --model qwen3.5-9b-vlm --benchmark memorybench --sample-source local --chunk-size 24 --local-window-tokens 24 --request-timeout-seconds 600`
- `uv run typhon evaluate-memory-suite --backend lmstudio_local --model qwen3.5-9b-vlm --benchmark longbench_v2 --benchmark locomo --benchmark locomo_plus --benchmark memorybench --benchmark evo_memory --sample-source local --chunk-size 24 --local-window-tokens 24 --request-timeout-seconds 600`

## 14. Recommended Conversation Topics for the PM

The most productive PM review questions now are:

1. Should the next milestone be upstream benchmark adapters or external baseline wrappers?
2. Which benchmark family should become the first official reproduction target?
3. Do we want to prioritize learned memory mechanisms or stronger baseline wrapping first?
4. What counts as a credible internal success criterion before any paper-facing claims?
5. How much of the repo should remain heuristic scaffolding versus production research code?
6. What is the first compute-budget envelope we will treat as the official local target?
7. Which local slices should be expanded next to reduce prompt-specific overfitting risk?

## 15. Recommended Next Steps

Ordered next steps:

1. Build benchmark-specific import adapters for one real upstream benchmark family.
2. Expand local pack sizes beyond the current tiny seed slices.
3. Re-run the same single-model suite on larger local slices.
4. Wrap one strong external baseline, preferably a local retention or local exact baseline that fits the 10 GB workstation.
5. Only after that, start implementing a learned memory module behind the current TYPHON selection interfaces.

## 16. Repo Documents to Read Next

- `README.md`
- `docs/TYPHON_CURSOR_HANDOFF.md`
- `docs/BENCHMARK_PACKS.md`
- `docs/decision-log.md`
- `docs/experiment-matrix.md`
- `docs/SOTA_MODEL_BACKENDS.md`
