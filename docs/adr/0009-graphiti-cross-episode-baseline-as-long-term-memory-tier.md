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
   modes** — the true fix is bi-temporal node summaries.
3. **Stability:** 3 trials, **std=0 on every headline metric** (window_recall 5/5,
   supersession leak 3/3 each trial) — deterministic at temp=0 once extraction is guided.
4. Real LoCoMo is **wired** (`locomo_real` benchmark + importer + CLI). Next levers:
   **bi-temporal / current-only node summaries** (best-of-both) and
   **shared-graph-per-conversation** (efficient graphiti on real LoCoMo).

Foundations added: `StrictSchemaClient`, guided extraction, combined edge+node search
(`search_mode` + `node_text_mode` knobs), real-LoCoMo importer (`locomo_real`), an eval
aggregation module, and **TYPHON's first test suite** (`tests/`, 27 tests, `pytest` config
in `pyproject.toml`).
