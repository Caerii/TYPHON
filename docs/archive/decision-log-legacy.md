# Legacy Decision Log

This file preserves the pre-ADR flat decision log for historical continuity.

The canonical source for active architectural decisions is now the [ADR directory](../adr/README.md).

## 2026-04-22 Legacy Entries

### 1. PyTorch-first research substrate

Decision:

- the repo substrate is Python-first and PyTorch-oriented, even though some baselines such as TTT-E2E are in JAX

Reason:

- the local implementation needs one stable control plane before cross-framework baseline integration
- FLA, Gated DeltaNet, MemoryLLM, PERK, and most likely TYPHON v0 fit naturally into a PyTorch-first repo

Consequence:

- JAX baselines will be wrapped later behind adapters rather than shaping the repo core now

### 2. JSON configs before YAML

Decision:

- use JSON for the initial benchmark and architecture configs

Reason:

- avoids adding a parser dependency before the repo can even run
- keeps smoke-test and registry validation on the Python standard library

Consequence:

- config ergonomics are slightly worse, but the substrate is portable immediately

### 3. Smoke tests write deterministic planning artifacts

Decision:

- the first executable path is a deterministic smoke runner that validates benchmark metadata and produces memory-allocation plans

Reason:

- the repo currently has no model code or dataset integrations, so the fastest honest way to create runnable value is to exercise the evaluation substrate and allocation logic

Consequence:

- smoke outputs are planning artifacts, not benchmark claims
- the next implementation step is wiring real dataset loaders into the same interfaces

### 4. TYPHON v0 starts with fast-weight adapters, not full solvers

Decision:

- the first architecture config targets a backbone plus parameter-efficient fast memory rather than a full MesaNet-like solver

Reason:

- adapter-based fast memory is cheaper to implement, easier to ablate, and realistic for local iteration

Consequence:

- MesaNet remains a baseline and design reference
- TYPHON v0 will prioritize controllability over maximal novelty

### 5. Use uv as the project tool runner

Decision:

- standardize command execution on `uv`

Reason:

- keeps environment management, script execution, and future dependency locking under one tool

Consequence:

- docs and local validation should use `uv run ...`
- dependency updates should be followed by `uv lock`

### 6. RTX 3080 workstation profile drives TYPHON v0

Decision:

- add an explicit runtime profile for an RTX 3080 10 GB workstation and tune TYPHON v0 around it

Reason:

- the local desktop constraint is now known and should shape practical implementation choices

Consequence:

- the first executable pipeline favors bounded local windows and adapter-style fast memory
- heavy solver-style TTT methods remain comparison targets, not the first local implementation

### 7. First baseline path is a local exact retrieval baseline

Decision:

- implement `attention_baseline` first as a local-exact-only runner on the same benchmark/runtime surface as TYPHON v0

Reason:

- gives TYPHON a concrete comparison artifact without pulling external repos into the critical path yet
- keeps the next step honest: TYPHON should beat something implemented under the same local constraints

Consequence:

- baseline and TYPHON artifact generation now share chunking and feature estimation helpers
- external baseline wrappers remain the next integration step after local comparison infrastructure

### 8. Comparison artifacts should be generated inside the repo

Decision:

- add a first `compare-v0` command that runs the local baseline path and TYPHON v0 path on the same benchmark slice and emits a comparison summary

Reason:

- separate artifacts are useful, but the repo also needs a machine-readable comparison surface for iterative evaluation

Consequence:

- comparison work can now stay inside this repo without depending on notebook-only analysis

### 9. Benchmark runners should support local data assets

Decision:

- extend the benchmark registry to discover repo-local benchmark samples under `data/benchmarks/<benchmark_id>/`

Reason:

- the next credible step after fixtures is a local data path that real runners can consume without redesigning the repo later

Consequence:

- runners now accept `fixture`, `local`, or `auto` sample sources
- benchmark data can be added incrementally while preserving the existing smoke-test flow

### 10. Local samples should carry reference answers for evaluation

Decision:

- extend local benchmark samples and runner artifacts with optional reference answers and heuristic scoring

Reason:

- structural memory differences are not enough; the repo needs a first quantitative evaluation loop before external model wrappers arrive

Consequence:

- baseline and TYPHON artifacts now carry prediction blocks when references exist
- the CLI now supports aggregate evaluation summaries and metric deltas

### 11. Evaluation needs explicit context-pressure controls

Decision:

- add overrideable chunk size and local-window tokens across the runner and evaluation surfaces

Reason:

- with default local samples and default chunking, the baseline and TYPHON can still observe nearly the same context, hiding the effect of memory allocation

Consequence:

- the repo can now force controlled memory-pressure experiments on the same local samples
- evaluation summaries record override provenance in both filenames and payloads

### 12. Local real-model execution should pivot to WSL plus one HTTP interface

Decision:

- use WSL2 Ubuntu as the preferred runtime for Linux-first servers and add a generic OpenAI-compatible inference backend in the repo

Reason:

- this machine has a working Ubuntu WSL2 environment with GPU visibility and `uv`
- vLLM and SGLang are strongest as Linux-first serving stacks, while Ollama remains the pragmatic Windows-native option
- a stable OpenAI-compatible backend lets the benchmark and evaluation surfaces stay fixed while the local server changes

Consequence:

- the repo now supports `openai_compatible_http` alongside `ollama_local` and the local extractive control backend
- future local serving work should start a server inside WSL and target it through `--base-url`

### 13. Bootstrap vLLM in WSL before chasing more wrappers

Decision:

- bootstrap a real vLLM environment in Ubuntu WSL before adding more backend wrappers

Reason:

- the repo now has a stable OpenAI-compatible inference surface
- this machine can run Linux GPU workloads inside WSL2
- one live vLLM path is more valuable than multiple unvalidated wrapper stubs

Consequence:

- `.venv-wsl-vllm` is now the first Linux server environment for this repo
- the active server target is `Qwen/Qwen3-4B-Instruct-2507` on port `8000`

### 14. LM Studio should be a first-class local backend, not just a generic URL override

Decision:

- add `lmstudio_local` as an explicit inference backend alias

Reason:

- LM Studio is live on this machine and already serving a usable model over an OpenAI-compatible interface
- the generic backend is still necessary, but a first-class alias makes the local workflow clearer and less error-prone

Consequence:

- the repo now supports `lmstudio_local` with the default base URL `http://localhost:1234/v1`
- model-backed evaluation can now target LM Studio without repeating the base URL on every command

### 15. Live evaluation should compare memory strategies on the same model

Decision:

- add a model-backed memory-strategy evaluation path that compares `full_context`, `attention_baseline`, and `typhon_v0` on the same sample and backend

Reason:

- backend reachability alone does not show whether TYPHON's selected context helps a real model
- comparing strategies on the same model isolates the effect of context selection from differences in model quality

Consequence:

- the repo now supports `evaluate-memory-strategies`
- LM Studio-backed runs show TYPHON-selected context can match full context while outperforming the constrained local-only baseline on the current local samples

### 16. Short-answer prompting should be explicit in the eval surface

Decision:

- require minimal answer-span outputs for `short_text` and classification-style local evaluations

Reason:

- verbose answers were lowering token precision and obscuring the effect of memory selection

Consequence:

- inference prompts now tell the model to return the shortest factual answer phrase
- memory-strategy comparison results are easier to interpret

### 17. Multi-model live evaluation should be one command, not notebook glue

Decision:

- add a memory-suite runner plus a backend model-list command

Reason:

- once the LM Studio path worked, the next bottleneck was experiment management rather than connectivity
- repeating single-model commands by hand is error-prone and makes model ranking harder to track

Consequence:

- the repo now supports `list-backend-models`
- the repo now supports `evaluate-memory-suite` to produce one leaderboard across multiple models and benchmarks

### 18. The default live path should use one hosted LM Studio model at a time

Decision:

- standardize the primary live evaluation path on `qwen3.5-9b-vlm` alone for this workstation

Reason:

- this RTX 3080 machine has 10 GB VRAM
- swapping or co-hosting multiple local models in LM Studio is the wrong default for stable benchmarking on this hardware
- the goal is consistent memory-strategy evaluation, not model-management churn

Consequence:

- the canonical live suite now uses only `qwen3.5-9b-vlm`
- a dedicated preset exists at `configs/live_eval/lmstudio_qwen35_9b_vlm.json`

### 19. Long live suites need explicit timeout and error accounting

Decision:

- add request-timeout control and per-strategy error accounting to the live inference and memory-suite paths

Reason:

- broader suites on local backends can exceed optimistic per-request timeouts
- one slow request should not destroy a full experiment artifact

Consequence:

- inference commands now accept `--request-timeout-seconds`
- memory-strategy summaries now record `error_count`

### 20. Preferred local benchmark data should use manifest-backed packs

Decision:

- standardize future local benchmark ingestion on `pack.json` plus per-pack sample files under `data/benchmarks/<benchmark_id>/packs/`

Reason:

- the repo had outgrown one-off root-level `samples.jsonl` files
- multiple curated slices per benchmark need explicit provenance and stable grouping
- the evaluation surface should not depend on hand-editing one monolithic local sample file

Consequence:

- the repo now supports `import-benchmark-pack` and `validate-benchmark-pack`
- the loader prefers manifest-backed local packs and still falls back to legacy local files when necessary

### 21. Expand live local coverage into continual-learning and streaming families

Decision:

- seed local imported packs for `memorybench` and `evo_memory` and run them through the live LM Studio memory-compare path

Reason:

- LongBench v2 and LoCoMo slices alone were not enough to test whether the current TYPHON context-selection effect generalizes across the intended research families
- the importer needed to prove value on real repo data, not just on paper

Consequence:

- `memorybench` and `evo_memory` now have manifest-backed local packs
- both benchmarks now have live memory-strategy evaluation artifacts on `qwen3.5-9b-vlm`

### 22. LongBench should enter through a dedicated upstream adapter, not manual sample authoring

Decision:

- add a dedicated `import-longbench` adapter that reads the official Hugging Face `THUDM/LongBench` dataset and normalizes English-task rows into manifest-backed local benchmark packs

Reason:

- the repo needed its first real upstream benchmark path before more baseline work
- LongBench is a better first adapter target than LongBench v2 for this workstation and current harness shape
- official LongBench rows include multi-answer gold labels, so the evaluation surface had to support `reference_answers` rather than only a single string

Consequence:

- the repo now supports `uv run --extra adapter_hf typhon import-longbench --config ...`
- local LongBench data now lives under `data/benchmarks/longbench/`
- prediction scoring now supports multiple acceptable reference answers

### 23. The first official LongBench result should be recorded even when it is unfavorable

Decision:

- keep the first imported LongBench smoke and live-eval result in the repo as evidence, even though the current heuristic TYPHON path does not beat the local baseline on the first sample

Reason:

- upstream benchmark integration is only credible if negative or mixed results are preserved
- the imported LongBench sample is materially harder than the repo's earlier curated slices and already exposed a full-context request failure on the local LM Studio server

Consequence:

- the experiment matrix now records the imported LongBench smoke pack and first live artifact
- LongBench becomes the calibration benchmark for the next phase of adapter and baseline work, not just a planned target

### 24. The first external Gated DeltaNet baseline should execute through WSL, not the Windows uv environment

Decision:

- implement `gated_deltanet_fla` as a WSL-backed baseline wrapper that calls a Linux Torch/FLA runtime from the Windows CLI

Reason:

- the Windows uv environment is running Python `3.14` without Torch, while the existing WSL venv already has CUDA-visible Torch and Transformers
- FLA is Linux-first in practice for this workstation, and the user explicitly asked to use WSL for Linux tooling

Consequence:

- the repo now includes `scripts/wsl/run_gated_deltanet_fla.py`
- the baseline wrapper bridges prompts and responses across Windows and WSL instead of trying to import Torch in the Windows process
- `scripts/wsl/bootstrap_fla.sh` is now the reproducible install step for the external baseline runtime

### 25. Community Gated DeltaNet checkpoints need explicit compatibility conversion

Decision:

- support the first external Gated DeltaNet baseline with a checkpoint-conversion shim rather than waiting for official NVIDIA weights

Reason:

- NVIDIA explicitly does not release pretrained Gated DeltaNet weights
- the practical third-party checkpoint used here, `linear-moe-hub/Gated-Deltanet-340M`, loads into the current FLA runtime only after:
  - importing `fla.models.gated_deltanet` explicitly for HF registration
  - disabling tied word embeddings for the load path
  - dropping legacy `.attn.D` tensors
  - splitting fused SwiGLU `gate_proj` weights into separate `gate_proj` and `up_proj` tensors

Consequence:

- the new baseline is a real runnable external model baseline, but it is not an official paper-weight reproduction
- the repo now makes checkpoint provenance and conversion assumptions explicit in the baseline config and artifacts

### 26. The first external baseline result should be kept even when it underperforms TYPHON

Decision:

- record the first `gated_deltanet_fla` LongBench result and comparison artifact as-is

Reason:

- the baseline had to truncate a `34,964` token prompt down to `2,048` input tokens because of model context limits
- direct QA prompting on an untuned or lightly tuned 340M community checkpoint is a meaningful systems datapoint even when the score is poor

Consequence:

- `results/baselines/gated_deltanet_fla__longbench__341d214a2c377691bf20e5a30b8f7979696bd19141ad9c9f.json` is now the first external baseline artifact
- `results/evaluations/compare__gated_deltanet_fla__vs__typhon_v0__longbench.json` now records TYPHON beating that first external baseline on the same 1-sample imported LongBench slice
