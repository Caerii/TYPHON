# TYPHON

Hierarchical cross-episodic associative memory.

Start here:

- [docs/TYPHON_CURSOR_HANDOFF.md](docs/TYPHON_CURSOR_HANDOFF.md)
- [docs/TECHNICAL_PM_BRIEF.md](docs/TECHNICAL_PM_BRIEF.md)
- [docs/WSL_RUNTIME.md](docs/WSL_RUNTIME.md)
- [docs/BENCHMARK_PACKS.md](docs/BENCHMARK_PACKS.md)

Current implementation baseline:

- project skeleton under `src/typhon`
- benchmark registry and smoke-test fixtures under `configs/benchmarks`
- repo-local benchmark assets under `data/benchmarks`
- benchmark-pack importer and manifest support under `src/typhon/benchmarks`
- upstream LongBench adapter configs under `configs/benchmarks/adapters`
- runtime profiles under `configs/runtime`
- baseline configs under `configs/baselines`
- live eval presets under `configs/live_eval`
- CLI entrypoint in `src/typhon/cli.py`
- wrapper script at `scripts/eval/smoke_test.py`
- first executable TYPHON v0 pipeline in `src/typhon/trainers/v0.py`
- first baseline runner in `src/typhon/baselines/local_exact.py`
- inference backends in `src/typhon/inference`
- tracking docs in `docs/decision-log.md` and `docs/experiment-matrix.md`

Quick start from the repo root:

```powershell
uv run python scripts/eval/smoke_test.py list-benchmarks
uv run python scripts/eval/smoke_test.py smoke-test --benchmark longbench_v2
uv run python scripts/eval/smoke_test.py smoke-test --family conversational_memory
uv run typhon list-baselines
uv run typhon list-inference-backends
uv run typhon list-backend-models --backend lmstudio_local
uv run typhon inspect-benchmark-data --benchmark longbench_v2
uv run typhon validate-benchmark-pack --benchmark memorybench
uv run --extra adapter_hf typhon import-longbench --config configs/benchmarks/adapters/longbench_english_smoke.json --replace
uv run typhon validate-benchmark-pack --benchmark longbench
uv run typhon smoke-test --benchmark longbench --sample-source local --sample-limit 10
uv run typhon inspect-inference-backend
uv run typhon profile-runtime
uv run typhon run-baseline --baseline gated_deltanet_fla --benchmark longbench --sample-source local --sample-limit 1
uv run typhon evaluate-compare --baseline gated_deltanet_fla --benchmark longbench --sample-source local --sample-limit 1
uv run typhon import-benchmark-pack --benchmark memorybench --input data/imports/memorybench_seed_v1.jsonl --pack-id seed_v1 --description "Seed local MemoryBench pack" --sample-id-field id --task-type-field kind --question-field prompt --context-field history --reference-answer-field gold --metadata-field details
uv run typhon run-baseline --baseline attention_baseline --benchmark longbench_v2 --sample-source local
uv run typhon run-v0 --benchmark longbench_v2 --sample-source local
uv run typhon compare-v0 --baseline attention_baseline --benchmark longbench_v2 --sample-source local
uv run typhon evaluate-baseline --baseline attention_baseline --benchmark longbench_v2 --sample-source local
uv run typhon evaluate-v0 --benchmark longbench_v2 --sample-source local
uv run typhon evaluate-compare --baseline attention_baseline --benchmark longbench_v2 --sample-source local
uv run typhon evaluate-compare --baseline attention_baseline --benchmark longbench_v2 --sample-source local --chunk-size 24 --local-window-tokens 24
uv run typhon run-inference-backend --backend extractive_heuristic --model local-extractive --benchmark longbench_v2 --sample-source local
uv run typhon evaluate-inference-backend --backend extractive_heuristic --model local-extractive --benchmark longbench_v2 --sample-source local
uv run typhon evaluate-memory-strategies --backend lmstudio_local --model qwen3.5-9b-vlm --benchmark longbench_v2 --sample-source local --sample-limit 1 --chunk-size 24 --local-window-tokens 24
uv run typhon evaluate-memory-suite --backend lmstudio_local --model qwen3.5-9b-vlm --benchmark longbench_v2 --benchmark locomo --benchmark locomo_plus --benchmark memorybench --benchmark evo_memory --sample-source local --chunk-size 24 --local-window-tokens 24 --request-timeout-seconds 600
uv run typhon evaluate-memory-strategies --backend lmstudio_local --model qwen3.5-9b-vlm --benchmark memorybench --sample-source local --chunk-size 24 --local-window-tokens 24 --request-timeout-seconds 600
uv run typhon evaluate-memory-strategies --backend lmstudio_local --model qwen3.5-9b-vlm --benchmark evo_memory --sample-source local --chunk-size 24 --local-window-tokens 24 --request-timeout-seconds 600
uv run typhon evaluate-memory-strategies --backend lmstudio_local --model qwen3.5-9b-vlm --benchmark longbench --sample-source local --sample-limit 1 --request-timeout-seconds 600
```

The smoke runner does not claim model accuracy. It validates benchmark metadata, chunking assumptions, and a first-pass TYPHON memory-allocation plan, then writes JSON artifacts under `results/smoke/`.

`run-v0` is the first executable TYPHON memory pipeline. It is still heuristic, but it now:

- detects the local runtime and selects a workstation profile
- adjusts chunk sizing to the active hardware recommendation
- computes per-chunk write signals
- allocates writes across local, fast-weight, episodic, and cross-episode stores
- emits a TYPHON artifact under `results/typhon/`

`run-baseline` currently supports the first local exact baseline:

- `attention_baseline`: only recent accessible context plus overlap-ranked retrieval
- `gated_deltanet_fla`: WSL-backed model baseline using FLA's Hugging Face-compatible Gated DeltaNet runtime

`compare-v0` emits a side-by-side structural summary of a baseline artifact and a TYPHON v0 artifact for the same benchmark.

Evaluation commands:

- `evaluate-baseline` writes aggregate scoring summaries for one baseline
- `evaluate-v0` writes aggregate scoring summaries for TYPHON v0
- `evaluate-compare` writes metric deltas between the baseline and TYPHON v0
- `run-inference-backend` writes model-backed or backend-backed sample artifacts
- `evaluate-inference-backend` writes aggregate scoring summaries for inference backends
- `evaluate-memory-strategies` compares `full_context`, `attention_baseline`, and `typhon_v0` on the same live backend
- `evaluate-memory-suite` runs that comparison across multiple models and benchmarks and writes one leaderboard artifact

Inference backend notes:

- `lmstudio_local` targets LM Studio's OpenAI-compatible server on `http://localhost:1234/v1`
- `openai_compatible_http` is the generic path for vLLM, SGLang, llama.cpp, LM Studio, or any compatible server
- `ollama_local` targets Ollama on `http://localhost:11434`
- `list-backend-models` prints the model ids a backend reports as available

Current live backend result:

- on the hosted LM Studio model `qwen3.5-9b-vlm`, the constrained-memory ablation shows `typhon_v0` matching `full_context` while beating the local-only baseline on the repo-local LongBench v2 and LoCoMo-Plus samples
- the canonical live-eval path on this machine is now the single hosted model `qwen3.5-9b-vlm`
- the current primary suite artifact is `results/evaluations/memory_suite__lmstudio_local__qwen3.5-9b-vlm__longbench_v2__locomo__locomo_plus__memorybench__evo_memory.json`
- the current five-benchmark rollup is mean `typhon_v0` F1 `0.5034` vs baseline `0.2183`, a mean delta of `+0.2851`
- imported local packs now exist for `memorybench` and `evo_memory`, and both have live memory-compare artifacts under `results/evaluations/`
- an official LongBench English smoke pack now exists under `data/benchmarks/longbench/packs/hf_english_smoke_v1/` with `28` imported Hugging Face samples across `14` tasks
- the first live LongBench adapter artifact is `results/evaluations/memory_compare__lmstudio_local__qwen3.5-9b-vlm__longbench.json`; on its first imported sample, the local baseline scored token F1 `0.129`, `typhon_v0` scored `0.0606`, and the full-context request hit an LM Studio `HTTP 400`, which is useful evidence that the official benchmark is materially harder than the current curated local slices
- the first external Gated DeltaNet artifact is `results/baselines/gated_deltanet_fla__longbench__341d214a2c377691bf20e5a30b8f7979696bd19141ad9c9f.json`; it runs through WSL with `flash-linear-attention` and the third-party checkpoint `linear-moe-hub/Gated-Deltanet-340M`
- that first external baseline sample scored token F1 `0.0` after truncating a `34,964` token prompt to the model's `2,048` token limit, and the comparison artifact `results/evaluations/compare__gated_deltanet_fla__vs__typhon_v0__longbench.json` shows TYPHON ahead on that same 1-sample slice by `+0.0377` token F1

Stress-test knobs:

- `--chunk-size` forces finer chunking on the same sample
- `--local-window-tokens` constrains how much recent context the local baseline can access
- these are useful for testing whether TYPHON preserves early information better than a local-only baseline

Local data convention:

- place benchmark assets under `data/benchmarks/<benchmark_id>/`
- preferred format is a benchmark pack manifest at `data/benchmarks/<benchmark_id>/pack.json`
- preferred sample files live under `data/benchmarks/<benchmark_id>/packs/<pack_id>/samples.jsonl`
- legacy `samples.jsonl` and `samples.json` at the benchmark root are still supported
- runner commands accept `--sample-source fixture|local|auto`
- `validate-benchmark-pack` checks manifest-backed or legacy local data before running experiments
- `import-benchmark-pack` normalizes external JSON or JSONL files into the preferred local pack format

Project tooling convention:

- use `uv run ...` for commands
- use `uv lock` when dependencies change
- use `uv run --extra adapter_hf ...` for the official LongBench Hugging Face adapter path, because that importer depends on `datasets<4` for dataset-script compatibility

WSL runtime note:

- this machine has Ubuntu on WSL2 with GPU visibility and `uv` available
- for Linux-first serving stacks such as vLLM or SGLang, prefer running the server inside WSL and pointing `openai_compatible_http` at it
- for the current Windows-native path, `ollama_local` remains the lowest-friction local backend
- for the external Gated DeltaNet baseline, install FLA into the WSL environment with `wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/bootstrap_fla.sh"`

Primary live preset:

- see `configs/live_eval/lmstudio_qwen35_9b_vlm.json`
- this preset deliberately uses one hosted LM Studio model at a time to avoid VRAM churn on the 10 GB GPU
