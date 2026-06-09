# 0009 — Graphiti cross-episode baseline as the long-term memory tier

- Status: accepted
- Date: 2026-06-06

## Context

TYPHON's memory hierarchy names four tiers — `local_exact`, `fast_weight`,
`episodic`, `cross_episode` — but the implemented baselines only exercise the
lower end (`attention_baseline` = local-exact recall; `gated_deltanet_fla` = a
full-context architectural backbone). The `system-overview` explicitly lists
"persistent cross-session consolidation" as **not yet provided**. We have no
baseline that actually recalls a fact stated in one session when the question is
asked in a much later one — which is exactly what `conversational_memory` and
`continual_learning` benchmarks (LoCoMo, MemoryBench, EvoMemory) stress.

Separately, the broader platform is adopting **Graphiti** (getzep/graphiti,
Apache-2.0) — a bi-temporal knowledge graph for agent memory — as the durable,
long-term tier of its memory stack. Rather than guess whether a temporal graph
helps, we want to measure it on the same bench, against the same baselines.

## Decision

Add a `graphiti_cross_episode` baseline:

- `configs/baselines/graphiti_cross_episode.json` (a normal `BaselineSpec`).
- `src/typhon/baselines/graphiti_cross_episode.py`, dispatched from `run_baseline`.
- It splits each sample's context into ordered sessions, ingests them as episodes
  into Graphiti (per-sample `group_id`, increasing valid-times), then answers from
  facts retrieved **across** episodes.
- It reuses `build_prediction_block` — the *same* extractive prediction path as
  `attention_baseline` — so a head-to-head isolates the one variable that matters:
  retrieval from a temporal graph vs. retrieval from a local window.
- `graphiti-core` is an **optional** dependency (a graph backend + an extraction
  LLM). The import is guarded; when graphiti-core or `GRAPH_URI` is absent, the
  runner emits a `not_executed` artifact in the normal shape rather than failing,
  so the harness still imports, lists, and dry-runs the baseline.

## Consequences

- TYPHON gains its first `cross_episode` baseline and a way to put a number on the
  platform's memory bet ("temporal graph lifts cross-episode recall by X% at Y
  cost vs. these baselines"), on `conversational_memory` / `continual_learning`.
- New runtime dependencies for a real run: a graph backend (Neo4j by default) and
  an extraction LLM. These stay optional; CI / no-backend environments degrade to
  `not_executed`.
- Current limitations (tracked in the runner): a fresh per-sample graph rather
  than one shared graph; synthetic session valid-times rather than the benchmark's
  real timestamps; extraction quality/cost bound to the configured LLM. Sharing a
  graph across samples and wiring real timestamps are the obvious next steps.
- The platform-side plan and framing live in the app repo (the memory note and the
  "Memory Stack" / "Memory, With a Past Tense" essays); this ADR records only the
  harness-side decision.

## Results & follow-up (2026-06-06)

First real runs, on Together.ai (no local GPU). Full write-up + raw artifacts:
`results/cross_episode/`.

Implementation notes that mattered:

- **Backend:** Together.ai, OpenAI-compatible. Extraction LLM
  `meta-llama/Llama-3.3-70B-Instruct-Turbo`; embedder
  `intfloat/multilingual-e5-large-instruct` (1024-dim). Cheaper Qwen3-235B
  (`-tput`, $0.20/$0.60) is a stronger extractor in isolation but **over-generates
  under strict constrained decoding** (fills the 16K budget without closing JSON →
  parse fails), so Llama-3.3-70B is the effective choice. Everything is env-driven
  (`LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`/`EMB_MODEL`/`EMB_DIM`).
- **Schema enforcement (decisive):** added `StrictSchemaClient`, which routes
  schema'd calls through the OpenAI SDK's `beta.chat.completions.parse` path
  (strict, closed schema) rather than the stock generic client's best-effort
  `json_schema`. Without `strict`, value/attribute edges were silently dropped.
- Session-marker stripping (`Session N:`/`Turn N:`) before ingest; `current_facts_only`
  so the long-term tier answers "what is true now".

Two scenarios, each isolating one claim (same extractive prediction path as
`attention_baseline`, so retrieval is the only variable):

Three compounding fixes were applied and measured (full write-up: `results/cross_episode/`):

| | window_recall (graphiti) | supersession stale-leak |
|---|---|---|
| edge-only, no guidance | 1/5 | 0/3 |
| + guided extraction (`entity_types` + instructions) | 3/5 | 0/3 |
| + combined edge+node search (`COMBINED_HYBRID_SEARCH_RRF`) | **5/5** | **3/3** |

Attention is **0/5** on window_recall throughout (can only see the filler session) and
leaks **3/3** on supersession throughout.

Conclusions:

1. **The win:** a persistent tier recalls *every* fact a bounded window structurally
   cannot (0/5 → 5/5). Guided extraction (capture values/roles/orgs as nodes) plus
   node-attribute retrieval were both required.
2. **The honest tradeoff:** node summaries unlocked that recall but aggregate history in
   present tense, so combined search **reintroduces the stale leak** (0/3 → 3/3). Edges
   are bi-temporally clean (0/3) but lossy. A `node_text_mode=current_edges` knob (name +
   current incident edges, no summary) cuts the leak to **1/3** and keeps window_recall 5/5,
   but loses value/attribute recall (0.85 → 0.37) and can't fix the case where the stale
   value *is* an entity name (`Postgres`). **Entity-name and value facts want opposite
   modes** — and the write-time fix is harder than it looks (see 5).
3. **Stability:** 3 trials, **std=0 on every headline metric** (window_recall 5/5,
   supersession leak 3/3 each trial) — deterministic at temp=0 once extraction is guided.
4. **Shared-graph-per-conversation** is implemented (ingest a conversation once, answer all
   its QA against it; validated on real LoCoMo — a single graph for conv-26). Real LoCoMo is
   **wired and run** (`locomo_real` benchmark + importer + CLI) — generalization + cost in (7).
5. **The write-time fix was attempted in the fork and does NOT resolve the leak.** We forked
   graphiti-core (`superintelligent-graphiti`) and filtered superseded edges out of the
   summary build (`_build_edges_by_node`, env-gated by `GRAPHITI_SUMMARIZE_CURRENT_ONLY`).
   The leak stayed **3/3** — because the node `summary` is LLM-generated + cumulative (not
   the edge-append we filtered; not regenerated on supersession), and the leaking sample had
   *zero* edges. Reverted. Even a full summary regeneration would miss values never captured
   as current edges (extraction quality) and keep legitimate current mentions of old values
   (`db`: "use SQLite, replacing Postgres"). So the tradeoff is **fundamental** to Graphiti's
   design + cheap-LLM extraction; the real resolution needs structured current-attribute
   extraction **and** summary regeneration. Navigate it today with `node_text_mode`.
6. **Lever 1 — structured current-attributes resolve the tradeoff (best-of-both, same day).**
   The prediction in (5) is borne out: a typed `Configurable {current_value}` entity type (the
   only typed entity, so attribute extraction runs only for changeable things) plus a
   `node_text_mode="attributes"` retrieval mode (structured attribute → current incident edges
   → drop stale-only → summary only when edgeless) deliver best-of-both at full recall (1.0),
   not the 0.37 the precision-only modes paid. **The leak metric was also fixed** (it was
   fragile — a substring test that fires even on a correct answer that names the value it
   superseded): disentangled into `stale_dominant` (stale present **and** current absent — the
   real failure), `clean` (current present **and** stale absent), and `current_recalled`.
   Re-scored (no re-run): **stale_dominant graphiti 0/3 = attention 0/3; clean graphiti 2/3 vs
   attention 0/3; current_recalled 3/3 both; window_recall 5/5 vs 0/5.** So the real win is
   *cleanliness* — graphiti returns the current value alone (rate-limit via the attribute,
   owner via bi-temporally invalidated edges) where attention dumps both — plus the categorical
   window_recall win. The one non-`clean` case (`db`, Postgres→SQLite) is **not a failure**: it
   recalls SQLite and stale doesn't dominate; it just can't be clean because the correct current
   fact itself says "SQLite … replacing Postgres". Part (5)'s summary-regeneration was therefore
   *not* needed: it cannot help db, and the attribute path resolves the cases it was meant to.
   Forcing db via extraction (anchor the project so an edge forms) **regressed** window_recall
   and was reverted. Snapshot:
   `results/cross_episode/comparison_lever1_attributes_2026-06-06.{json,txt}`; pinned by a live
   integration test (`tests/test_graphiti_live.py`).
7. **Real-LoCoMo generalization + cost (2026-06-07) — the "X% lift at Y cost" number.** Beyond
   the synthetic probes, on real LoCoMo (conv-26, 12 multi-hop QA, shared-graph): graphiti mean
   token_recall **0.1528 vs attention 0.0917 = +66.6% relative**, at a **one-time ~$0.46 per
   conversation** (216 LLM calls, 519,831 tokens at $0.88/1M, 41 episodes; ~$0.038/QA amortized).
   Per-sample it is a *mixed* result — graphiti wins 3 (q011 a perfect 1.0 vs 0.0), loses 4,
   ties 5 — the aggregate win driven by cross-session questions a window can't reach. Honest
   caveats: absolute recall is low for both (extractive prediction, not generative answering);
   **2 extraction calls hit the 16K completion cap and failed even after retry** (Llama-3.3-70B
   over-generates on messier real episodes → lost facts, wasted tokens in the $0.46); n=1
   conversation. Snapshot: `results/cross_episode/locomo_real_compare_2026-06-07.{json,txt}`
   (driver `scripts/run_locomo_real.py`). Cost is instrumented per-group on every artifact
   (`cost` block + `BudgetLedger`).
8. **SOTA axis + Tier 2 Phase 1 reranker (2026-06-07/08).** SOTA agent-memory numbers (Mem0/Zep)
   are an **LLM-judge** binary-correctness score over a *generated* answer, not extractive recall;
   `typhon.eval.generation` adds a generative reader + LLM-judge (J-score), applied post-hoc over the
   *same* retrieved facts (`scripts/judge_run.py`). First Tier-2 retrieval lever: a backend-agnostic
   **`ListwiseReranker`** (`reranker.py`), because Graphiti's stock cross-encoder ranks via OpenAI-only
   token `logprobs` + `logit_bias` tokenizer IDs and **crashes on Together/local** (empty
   `logprobs.content` → `zip(strict=True)` raises). The replacement scores all candidates in **one**
   strict-structured-output call (RankGPT-style), Pydantic-validated, usage-counted, fail-closed,
   model-via-config. On real-LoCoMo (conv-26, 12 QA), reranking **lowered token_recall 0.194→0.125 but
   raised J-score 0.083→0.167 (1/12→2/12 correct)** — the metrics diverge *because* the reranker demotes
   lexical-overlap distractors (which inflate token_recall) and promotes genuinely-relevant facts (q004
   identity: "transgender conference" reranked to #1 → correct). Confirms J-score is the right axis;
   n=1, so +1 answer is a signal not a sweep. Snapshot:
   `results/cross_episode/tier2_phase1_reranker_2026-06-08.{json,txt}` (driver `scripts/tier2_compare.py`).

Foundations: `StrictSchemaClient`, guided extraction, combined edge+node search
(`search_mode` + `node_text_mode` knobs), shared-graph-per-conversation, real-LoCoMo importer
(`locomo_real`), an eval aggregation module, the package decomposition, and **TYPHON's first
test suite** (`tests/`, 36 tests, `pytest` config in `pyproject.toml`).
