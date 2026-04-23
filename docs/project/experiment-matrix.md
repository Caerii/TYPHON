# Experiment Matrix

This is the canonical project tracker for benchmark coverage, baseline coverage, runtime status, and headline empirical artifacts.

## Benchmark Status

| Benchmark | Family | Status | Local runner | Notes |
| --- | --- | --- | --- | --- |
| LongBench v2 | long_context_reasoning | local slice evaluated | live local sample path | 4 local samples plus LM Studio memory compare |
| LongBench | long_context_reasoning | imported official smoke pack | live imported pack path | 28 Hugging Face English-task samples across 14 tasks; smoke-tested and first LM Studio memory compare recorded |
| BABILong | long_context_reasoning | scaffolded | smoke only | config and fixture added |
| ZeroSCROLLS | long_context_reasoning | scaffolded | smoke only | config and fixture added |
| LoCoMo | conversational_memory | local slice evaluated | live local sample path | 2 local samples plus LM Studio memory compare |
| LoCoMo-Plus | conversational_memory | local slice evaluated | live local sample path | 3 local samples plus LM Studio memory compare |
| Evo-Memory | streaming_agentic_memory | local slice evaluated | live imported pack path | 2 imported local-pack samples plus LM Studio memory compare |
| MemoryBench | continual_learning | local slice evaluated | live imported pack path | 2 imported local-pack samples plus LM Studio memory compare |

## Baseline Status

| Baseline | Role | Status | Implementation source | Next action |
| --- | --- | --- | --- | --- |
| Attention baseline | local exact recall | implemented | local heuristic runner | keep as the simple in-repo control condition |
| Gated DeltaNet via FLA | local retention baseline | implemented | WSL-backed FLA wrapper plus third-party 340M checkpoint | evaluate on larger LongBench slices and decide whether to keep this checkpoint family or switch to a stronger one |
| PERK | fast-weight baseline | pending | official repo verified | inspect API and wrap |
| MesaNet | fast-weight solver baseline | pending | FLA | inspect layer API and wrap |
| TTT-E2E | continual-learning baseline | pending | official JAX repo verified | decide adapter boundary |
| qTTT | targeted inference adaptation | pending | paper verified | locate or implement minimal reproduction |
| MemoryLLM | latent memory baseline | pending | official repo verified | inspect config surface |
| M+ | persistent retrieval-backed baseline | pending | official repo verified | inspect long-term memory hooks |
| FwPKM | sparse episodic baseline | pending | official repo verified | inspect integration cost |
| GradMem | loss-driven memory writing | pending | paper verified | locate code or implement minimal version |

## Inference Backend Status

| Backend | Role | Status | Runtime target | Next action |
| --- | --- | --- | --- | --- |
| Extractive heuristic | control condition | implemented | local Python | keep as deterministic fallback |
| Ollama local | Windows-native model backend | implemented | Windows localhost | start server and run a real local model |
| LM Studio local | Windows-native OpenAI-compatible backend | implemented | Windows localhost | keep as the canonical live local path |
| OpenAI-compatible HTTP | generic server-backed model backend | implemented | WSL or remote localhost | connect vLLM, SGLang, llama.cpp, or compatible servers |

## TYPHON Roadmap Status

| Milestone | Status | Blocking issue |
| --- | --- | --- |
| Repo substrate | implemented | none |
| Benchmark registry | implemented | broader upstream adapters still missing |
| Smoke-test runner | implemented | none |
| Local benchmark asset discovery | implemented | local slices remain small |
| Heuristic scoring and evaluation summaries | implemented | still depends on heuristic memory selection |
| Context-pressure overrides | implemented | most useful on curated slices so far |
| Runtime profiling | implemented | none |
| TYPHON v0 config | implemented | refine after stronger benchmark and baseline coverage |
| TYPHON v0 executable pipeline | implemented | uses heuristic signals, not learned updates |
| Baseline wrappers | in progress | first external baseline exists; broader coverage still missing |
| Comparison artifacts | implemented | external-baseline comparisons are still sparse |
| Real benchmark loaders | in progress | LongBench adapter implemented; LoCoMo and others still pending |
| Model-backed inference backends | in progress | LM Studio path works; broader runtime coverage remains |
| WSL vLLM runtime | in progress | environment exists; no stable benchmark-serving path validated yet |
| Benchmark-pack importer | implemented | benchmark-specific import adapters still need expansion |

## Live Memory Compare

| Backend | Model | Benchmark | Setting | Result |
| --- | --- | --- | --- | --- |
| LM Studio local | qwen3.5-9b-vlm | LongBench imported smoke pack | default live compare, `1` sample | baseline F1 `0.129`, `typhon_v0` F1 `0.0606`; full-context request failed with LM Studio `HTTP 400` on the first imported sample |
| LM Studio local | qwen3.5-9b-vlm | LongBench v2 local sample | `chunk=24`, `window=24` | `typhon_v0` F1 `0.6154` vs baseline `0.2857`; `typhon_v0` matches full context |
| LM Studio local | qwen3.5-9b-vlm | LoCoMo-Plus local sample | `chunk=24`, `window=24` | `typhon_v0` F1 `0.6` vs baseline `0.0`; `typhon_v0` matches full context |
| LM Studio local | qwen3.5-9b-vlm | MemoryBench local pack | `chunk=24`, `window=24` | `typhon_v0` F1 `0.3929` vs baseline `0.1818`; full context `0.5` |
| LM Studio local | qwen3.5-9b-vlm | Evo-Memory local pack | `chunk=24`, `window=24` | `typhon_v0` F1 `0.3987` vs baseline `0.2308`; full context `0.5357` |

## External Baseline Status

| Baseline | Checkpoint | Runtime | Benchmark | Samples | Result |
| --- | --- | --- | --- | --- | --- |
| `gated_deltanet_fla` | `linear-moe-hub/Gated-Deltanet-340M` | WSL + `flash-linear-attention` | LongBench imported smoke pack | `1` | token F1 `0.0`; prompt truncated from `34,964` to `2,048` tokens |
| `gated_deltanet_fla` vs `typhon_v0` | same | same | LongBench imported smoke pack | `1` | TYPHON delta `+0.0377` token F1, `+0.25` token recall |

## LongBench Adapter Status

| Item | Status | Notes |
| --- | --- | --- |
| `reference_answers` support | implemented | scoring now selects the best match across multiple official answers |
| Hugging Face import command | implemented | `import-longbench` reads `THUDM/LongBench` and writes a manifest-backed local pack |
| English smoke config | implemented | `configs/benchmarks/adapters/longbench_english_smoke.json` imports 2 samples per task |
| Full English config | scaffolded | `configs/benchmarks/adapters/longbench_english_full.json` exists but has not been imported locally yet |
| Offline smoke validation | implemented | `smoke-test` succeeded on 10 imported samples |
| Offline baseline vs TYPHON compare | implemented | 4-sample compare artifact exists; TYPHON and baseline are roughly tied on the current smoke slice |
| External Gated DeltaNet baseline | implemented | WSL-backed FLA wrapper runs on the imported LongBench pack; first artifact and compare summary now exist |

## Live Memory Suite

Primary artifact:

- `results/evaluations/memory_suite__lmstudio_local__qwen3.5-9b-vlm__longbench_v2__locomo__locomo_plus__memorybench__evo_memory.json`

Primary single-model leaderboard:

| Benchmark | Samples | Typhon F1 | Baseline F1 | Full-context F1 | Typhon delta |
| --- | --- | --- | --- | --- | --- |
| LoCoMo | `2` | `0.5585` | `0.125` | `0.5585` | `+0.4335` |
| LongBench v2 | `4` | `0.5224` | `0.1981` | `0.5409` | `+0.3243` |
| LoCoMo-Plus | `3` | `0.6445` | `0.3556` | `0.6445` | `+0.2889` |
| MemoryBench | `2` | `0.3929` | `0.1818` | `0.5` | `+0.2111` |
| Evo-Memory | `2` | `0.3987` | `0.2308` | `0.5357` | `+0.1679` |

Primary single-model rollup:

| Model | Benchmarks | Mean Typhon F1 | Mean baseline F1 | Mean full-context F1 | Mean Typhon delta |
| --- | --- | --- | --- | --- | --- |
| qwen3.5-9b-vlm | `5` | `0.5034` | `0.2183` | `0.5559` | `+0.2851` |
