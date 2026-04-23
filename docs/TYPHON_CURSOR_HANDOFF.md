# TYPHON Cursor Handoff

Last verified: 2026-04-22
Audience: the next Codex/Cursor session working locally in this repo
Role: execution lead operating with project-manager discipline

## 1. Program Intent

TYPHON is not a single hack for longer context windows. It is a research program for hierarchical adaptive memory in LLM systems, with emphasis on both:

- within-sequence long-context reasoning
- cross-episode memory for agentic and continual-learning settings

The core bet is that no single mechanism is sufficient:

- attention-style exact recall is strong locally but expensive and brittle at scale
- test-time training / fast weights can compress context adaptively but usually remain sequence-local
- latent memory and retriever-backed memory help persistence but do not by themselves solve write allocation and interference
- benchmark results increasingly show that explicit recall alone is not enough; models also need evolving procedural and cognitive memory

## 2. What Has Been Verified

This section is grounded in primary or official sources checked on 2026-04-22.

### Core methods

| Method | Verified source | Why it matters to TYPHON | Notes |
| --- | --- | --- | --- |
| Titans | https://arxiv.org/abs/2501.00663 | Surprise-gated neural long-term memory with momentum/forgetting is the clearest recent write-policy reference point. | `https://github.com/ABehrouz/Titans` exists but was empty on 2026-04-22. Treat the paper as primary source, not the repo. |
| MemoryLLM | https://arxiv.org/abs/2402.04624 | Latent memory pool inside the transformer gives a strong "in-model memory" baseline. | Official repo: https://github.com/wangyu-ustc/MemoryLLM |
| M+ | https://arxiv.org/abs/2502.00592 | Extends MemoryLLM with long-term memory and a co-trained retriever; strong reference for persistent memory beyond a single sequence. | Same official repo as MemoryLLM. |
| MesaNet | https://arxiv.org/abs/2506.05233 | Best reference for chunkwise, locally optimal test-time training with explicit compute-allocation tradeoffs. | Implemented in FLA: https://github.com/fla-org/flash-linear-attention |
| PERK | https://arxiv.org/abs/2507.06415 | Parameter-efficient test-time adaptation via LoRA-style memory is a practical fast-weight candidate. | Public repo: https://github.com/eric11eca/perk |
| qTTT | https://arxiv.org/abs/2512.13898 | Query-only test-time training is the cleanest evidence that small, targeted inference-time updates can outperform more "thinking" tokens on long-context tasks. | Paper verified; no official code repo was confirmed in this pass. |
| TTT-E2E | https://arxiv.org/abs/2512.23675 | Reframes long-context modeling as continual learning; useful as a strong sequence-local compression baseline. | Official repo: https://github.com/test-time-training/e2e |
| GradMem | https://arxiv.org/abs/2603.13875 | Gradient-based writing into memory tokens is a clean loss-driven write mechanism. | Paper verified; code not confirmed in this pass. |
| FwPKM | https://arxiv.org/abs/2601.00671 | Strong sparse episodic-memory reference with chunk-level updates and long-context extrapolation. | Official repo: https://github.com/SakanaAI/fast-weight-product-key-memory |
| Gated DeltaNet | https://github.com/NVlabs/GatedDeltaNet | Strong efficient local-memory / retention backbone and baseline for sequence modeling. | Use as a backbone or as a "recent exact-ish memory" comparator. |

### Benchmarks

| Benchmark | Verified source | Why it matters |
| --- | --- | --- |
| LongBench v2 | https://arxiv.org/abs/2412.15204 | Realistic long-context reasoning across documents, dialogue, code, and structured data. |
| LongBench | https://aclanthology.org/2024.acl-long.172 | Older, cheaper long-context benchmark for early smoke tests. |
| BABILong | https://arxiv.org/abs/2406.10149 | Controlled reasoning-in-a-haystack stress test. |
| ZeroSCROLLS | https://arxiv.org/abs/2305.14196 | Long-text understanding benchmark still used in recent TTT papers. |
| LoCoMo | https://arxiv.org/abs/2402.17753 | Very long-term conversational memory benchmark with QA, summarization, and multimodal dialogue tasks. |
| LoCoMo-Plus | https://arxiv.org/abs/2602.10715 | Tests cognitive memory beyond factual recall; important to avoid optimizing only for explicit retrieval. |
| Evo-Memory | https://arxiv.org/abs/2511.20857 | Streaming, self-evolving memory benchmark for agentic task sequences. |
| MemoryBench | https://arxiv.org/abs/2510.17281 | Continual learning and user-feedback benchmark; tests declarative and procedural memory. |

## 3. Working Thesis

TYPHON should be a four-layer memory system:

1. Local exact / near-exact recall
2. Fast-weight adaptive memory
3. Sparse episodic memory
4. Slow cross-episode memory

The key research question is not "can we add more memory?" It is:

How should an LLM allocate write, read, and consolidation budget across multiple memory timescales so that future loss drops more than compute and interference rise?

## 4. Non-Negotiable Design Principles

- Build around hierarchical memory, not only long-context sequence modeling.
- Treat write policy as first-class. Do not bolt on memory without a theory of when to write.
- Keep compute accounting explicit. Every result must report training FLOPs, inference FLOPs, active memory size, and latency.
- Separate short-term adaptation from persistent storage to reduce interference.
- Evaluate on at least one benchmark from each family before making strong claims:
  - long-context reasoning
  - conversational memory
  - streaming / agentic memory
  - continual learning from feedback
- Start with a minimal prototype that is ablatable and debuggable. Do not jump straight to the full system.

## 5. Proposed Architecture

### Layer 1: Local exact / retention memory

Purpose: handle recent dependencies, local copying, near-field retrieval, and strong token-level modeling.

Candidate backbones:

- FlashAttention-style full or windowed attention
- FoX inside FLA
- Gated DeltaNet

Initial recommendation:

- use a PyTorch backbone with clean interfaces and strong kernels
- prefer FLA-backed implementations where possible, because it already hosts FoX, MesaNet, and Gated DeltaNet-adjacent infrastructure

### Layer 2: Fast-weight adaptive memory

Purpose: compress useful chunk-level structure from the current sequence into quickly writable parameters or states.

Candidate implementations:

- MesaNet-style chunkwise solver
- PERK-style LoRA adapter updated at test time
- GradMem-style memory tokens updated by gradient descent
- TTT-E2E-inspired continual weight updates for specific submodules

Initial recommendation:

- first prototype this layer as a parameter-efficient adapter rather than a full solver
- implement a common interface:
  - `prepare_context(chunks)`
  - `write(batch, features)`
  - `read(hidden_state)`
  - `reset()`

### Layer 3: Sparse episodic memory

Purpose: store rare, high-value events and key-value associations that should not be kept only in transient fast weights.

Candidate implementation:

- FwPKM-style sparse PKM with chunk-level local updates on activated slots

Key requirement:

- writing must be selective and auditable
- keep per-write metadata: source chunk, timestamp, utility score, overwrite target, retrieval hits

### Layer 4: Slow cross-episode memory

Purpose: retain information across sessions, tasks, or episodes with controlled consolidation.

Candidate implementation:

- M+-style long-term memory plus retriever
- append-only or bounded store with periodic consolidation and decay

Key requirement:

- episode boundary handling must be explicit
- consolidation should run less frequently than fast writes

## 6. Write Policy

TYPHON should use a utility-based write policy rather than a fixed heuristic.

### Inputs to the policy

- surprise or prediction error
- reconstruction loss or next-token loss gradient norms
- chunk rarity or retrieval novelty
- predicted future usefulness
- storage cost
- expected interference risk

### Actions

- no write
- write to fast-weight memory
- write to episodic memory
- consolidate to long-term memory
- evict or decay low-value memory

### Objective

Use a resource-rational objective of the form:

`choose action a to maximize E[future_loss_reduction | a, state] - lambda_compute * cost_compute(a) - lambda_mem * cost_memory(a) - lambda_interference * risk(a)`

This does not need to be fully learned on day one. Start with a heuristic proxy:

- surprise score
- utility predictor from held-out replay gain
- explicit budget caps per layer

## 7. Main Workstreams

### W0. Program scaffolding

Deliverables:

- repo structure
- dependency strategy
- experiment registry
- decision log
- benchmark matrix

Acceptance criteria:

- a new session can boot the project without reverse-engineering the repo
- every experiment has a config, output directory, and budget record

### W1. Benchmark substrate

Deliverables:

- loaders and runners for LongBench v2, LongBench, BABILong, ZeroSCROLLS, LoCoMo, LoCoMo-Plus, Evo-Memory, and MemoryBench
- a unified evaluation interface

Acceptance criteria:

- one command per benchmark family runs a smoke test
- benchmark adapters normalize prompt format, context segmentation, and metrics

### W2. Baseline reproduction

Deliverables:

- baseline matrix with status for MemoryLLM, M+, MesaNet, PERK, qTTT, TTT-E2E, GradMem, FwPKM, Gated DeltaNet, and RAG baselines

Acceptance criteria:

- at least 3 baselines run end to end on one benchmark family
- every reproduced score includes exact commit, model, context length, and budget

### W3. TYPHON v0

Deliverables:

- backbone + fast-weight layer only
- no episodic or cross-episode memory yet
- clean ablations versus backbone-only

Acceptance criteria:

- runs on LongBench smoke set
- demonstrates either better accuracy at matched budget or similar accuracy at lower inference cost

### W4. TYPHON v1

Deliverables:

- add sparse episodic memory
- add write metadata and tracing

Acceptance criteria:

- measurable gain on retrieval-heavy or reasoning-in-a-haystack tasks
- episodic writes can be inspected and attributed to future retrievals

### W5. TYPHON v2

Deliverables:

- add cross-episode memory and consolidation
- add Evo-Memory and MemoryBench runs

Acceptance criteria:

- model shows positive reuse across episodes or feedback rounds
- catastrophic drift is bounded by explicit metrics

### W6. Theory and analysis

Deliverables:

- retrieval SNR analysis by layer
- interference analysis
- write-allocation analysis

Acceptance criteria:

- theory predicts at least one empirical pattern seen in ablations

## 8. Recommended Repo Layout

Use this unless a better local convention already exists:

```text
TYPHON/
  README.md
  docs/
    TYPHON_CURSOR_HANDOFF.md
    decision-log.md
    experiment-matrix.md
  configs/
    benchmarks/
    baselines/
    typhon/
  scripts/
    setup/
    eval/
    train/
  src/
    typhon/
      backbones/
      memory/
      policies/
      retrieval/
      trainers/
      eval/
      utils/
  third_party/
  results/
    smoke/
    baselines/
    typhon/
  notebooks/
```

Tooling convention:

- use `uv` for local commands and dependency management
- prefer `uv run ...` over raw `python ...`
- refresh `uv.lock` when dependencies change

Rules:

- keep third-party repos isolated under `third_party/`
- do not modify third-party code until a baseline is reproduced
- wrap baseline calls with thin adapters inside `src/typhon/`
- write every major decision into `docs/decision-log.md`

## 9. Immediate Execution Order

Do these in order. Do not skip ahead to architecture experiments before the substrate exists.

1. Create the repo skeleton from Section 8.
2. Add `docs/decision-log.md` and `docs/experiment-matrix.md`.
3. Build a benchmark registry with stubs for all target benchmarks.
4. Start with two benchmark families for smoke tests:
   - LongBench or LongBench v2
   - LoCoMo
5. Reproduce one strong local-memory baseline:
   - Gated DeltaNet or a strong attention baseline
6. Reproduce one fast-adaptation baseline:
   - PERK, MesaNet, or TTT-E2E
7. Implement TYPHON v0:
   - backbone + fast-weight adapter
   - simple surprise-based write heuristic
8. Run matched-budget comparisons on the first benchmark family.
9. Only after v0 is stable, add FwPKM-style episodic memory.
10. Only after episodic memory works, add cross-episode consolidation and Evo-Memory / MemoryBench evaluation.

## 10. First Sprint Definition

Sprint goal:

Establish a reproducible substrate and a minimal TYPHON v0 that can be compared against at least one backbone baseline on a small but real long-context benchmark slice.

Concrete sprint outputs:

- project skeleton exists
- benchmark smoke tests run
- one baseline reproduced
- one TYPHON v0 training or inference path runs
- one comparison table is generated

Sprint should stop if:

- baseline reproduction is still broken
- benchmark data is not normalized
- compute budgets are not being logged

Do not hide infra failures under architecture work.

## 11. Open Questions That Must Be Resolved Early

- Which base scale is realistic for local experimentation: under 1B, 1B to 4B, or adapter-only on an external pretrained model?
- Is the first implementation PyTorch-only, or do we also need JAX compatibility because TTT-E2E is in JAX?
- Which benchmark slice becomes the primary inner-loop development target?
- Should long-term memory be per-layer or global at v1?
- Is write-policy learning deferred until heuristics are stable, or tackled immediately with meta-learning / RL?

Default recommendation:

- PyTorch-first
- adapter-based fast memory first
- LongBench or BABILong for early iteration
- LoCoMo after the first stable prototype
- Evo-Memory and MemoryBench after persistent-memory support exists

## 12. Risks

- Benchmark sprawl: too many datasets before any stable runner exists
- Baseline sprawl: too many partially integrated codebases
- Compute opacity: unfair comparisons because TTT methods hide extra inference work
- Memory contamination: cross-episode store polluted by low-value writes
- Interface churn: memory modules added before a stable backbone API exists

Mitigation:

- force every experiment through a config and budget ledger
- gate new layers behind ablations
- keep memory interfaces narrow and explicit

## 13. Suggested Prompt For The Next Codex Session

Paste this into the next local Codex/Cursor session:

```text
You are continuing the TYPHON research repo. Read docs/TYPHON_CURSOR_HANDOFF.md first and act as the execution lead for the next sprint. Your immediate job is not to brainstorm; it is to create the repo skeleton, add a decision log and experiment matrix, and implement benchmark registry stubs plus the first smoke-test runner. Keep all work reproducible, budget-aware, and ablatable. Do not start the full architecture until the benchmark substrate and at least one baseline path are in place. Record every major decision in docs/decision-log.md.
```

## 14. Source Links

Methods:

- Titans: https://arxiv.org/abs/2501.00663
- MemoryLLM: https://arxiv.org/abs/2402.04624
- M+: https://arxiv.org/abs/2502.00592
- MesaNet: https://arxiv.org/abs/2506.05233
- PERK: https://arxiv.org/abs/2507.06415
- qTTT: https://arxiv.org/abs/2512.13898
- TTT-E2E: https://arxiv.org/abs/2512.23675
- GradMem: https://arxiv.org/abs/2603.13875
- FwPKM: https://arxiv.org/abs/2601.00671
- Gated DeltaNet repo: https://github.com/NVlabs/GatedDeltaNet
- FLA repo: https://github.com/fla-org/flash-linear-attention
- MemoryLLM / M+ repo: https://github.com/wangyu-ustc/MemoryLLM
- PERK repo: https://github.com/eric11eca/perk
- FwPKM repo: https://github.com/SakanaAI/fast-weight-product-key-memory
- TTT-E2E repo: https://github.com/test-time-training/e2e
- LoCoMo-Plus repo: https://github.com/xjtuleeyf/Locomo-Plus

Benchmarks:

- LongBench v2: https://arxiv.org/abs/2412.15204
- LongBench: https://aclanthology.org/2024.acl-long.172
- BABILong: https://arxiv.org/abs/2406.10149
- ZeroSCROLLS: https://arxiv.org/abs/2305.14196
- LoCoMo: https://arxiv.org/abs/2402.17753
- LoCoMo-Plus: https://arxiv.org/abs/2602.10715
- Evo-Memory: https://arxiv.org/abs/2511.20857
- MemoryBench: https://arxiv.org/abs/2510.17281
- MemoryBench repo: https://github.com/LittleDinoC/MemoryBench
