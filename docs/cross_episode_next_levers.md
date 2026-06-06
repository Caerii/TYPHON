# Cross-episode memory — next levers (scoped)

Follow-ups from the `graphiti_cross_episode` work (PR: branch `feat/graphiti-cross-episode`;
results in `results/cross_episode/`; decision in `docs/adr/0009-…`). Each lever below is
scoped: problem → design → where → effort/risk → validation → open questions.

The honest state we're building on: a persistent graph recalls what a bounded window can't
(window_recall **0/5 → 5/5**, std=0), but the **recall↔temporal-precision tradeoff is
fundamental** — node summaries are LLM-generated + cumulative, so they leak superseded
values; the retrieval-time `node_text_mode` knobs cut the leak only by losing value recall.
Levers 1–2 attack that and the extraction ceiling; lever 3 lands the whole thing in the
platform.

---

## Lever 1 — Structured current-attributes + summary regeneration (the true best-of-both)

**⚠️ Both halves probed — neither is a quick win (status: open, harder than scoped).**
- *Attributes:* a throwaway test with a typed `Configurable {current_value}` entity type left
  `node.attributes = null` on all three supersession cases — the subjects weren't classified
  to the custom type / the attribute pass didn't populate the field. Getting values into
  structured attributes needs real work on entity modeling + ensuring the attribute-extraction
  pass runs and fills fields.
- *Summary regen (fork):* already attempted and reverted (see ADR 0009 §5) — the leak is the
  cumulative LLM summary, not the edge-append, and a sample with zero edges still leaks.
- *Silver lining:* the generated summaries are already fairly temporal (`SQLite will be used`,
  `Postgres was chosen`, `Sam owns`); the failure is that the extractive `predict_answer`
  still selects the stale sentence — so a **temporal-aware reranker** (prefer current-tense /
  current-edge-aligned sentences) may be a lighter third angle than either half above.

**Problem.** A changed fact ("rate limit 100 → 40") leaks the stale value because the node
`summary` is LLM-written from the *first* episode and never regenerated. The
`current_edges`/`bitemporal` modes avoid the summary but then lose values that live only in
it (`40`, `March 14`): recall 0.85 → 0.37. Evidence: `comparison_forkfix_attempt_*` (a naive
edge-filter did nothing — leak stayed 3/3).

**Design (two parts, do attributes first).**
1. **Structured current-attributes (mostly our layer, no fork).** Graphiti re-extracts node
   attributes *per episode with the prior attributes as context* and merges them
   (`extract_attributes_from_nodes` → `apply_capped_attributes`), so a typed field can be
   **updated to current** on supersession. Define `entity_types` with typed fields — e.g. a
   `State`/`Value` model carrying `current_value: str | None` + `as_of: str | None` — so
   "rate limit" lands as a *structured attribute* on the entity, not free text. Retrieval
   already surfaces `node.attributes` in `summary` mode; prefer attributes over the prose
   summary when present.
2. **Summary regeneration (fork).** When `GRAPHITI_SUMMARIZE_CURRENT_ONLY` is set, rebuild
   `node.summary` from **current edges only** instead of appending to the cumulative base:
   in `node_operations._extract_entity_summaries_batch`, start from `''` (not `node.summary`)
   and drop superseded edges; mirror it in the `_process_summary_flight` LLM path so the
   stale base isn't fed to the model.

**Where.** `superintelligent-graphiti/graphiti_core/utils/maintenance/node_operations.py`
(+ `attribute_utils.py`) for (2); our `extraction.py` (typed entity types) + `facts._node_text`
(attribute preference) for (1).

**Effort / risk.** Medium–high. Risk: changes Graphiti's summary behavior (regression-test
other paths); per-node attribute extraction adds LLM cost; typed extraction may still miss a
value the model never structured.

**Validation.** Re-run `compare_baselines` expecting **window 5/5 + supersession leak 0/3 +
recall ≈ 0.85** in `summary` mode. Add a fork unit test for the regen path + a baseline test
for attribute-preferring `_node_text`. Contribute (2) upstream as an opt-in config field
(cleaner than the env flag).

**Open questions.** Does `apply_capped_attributes` overwrite-to-current or keep-prior on
conflict (verify the merge, line ~251)? Reliability of typed-value extraction on terse text?
Attribute-extraction cost at scale.

---

## Lever 2 — Session chunking for long episodes (real-LoCoMo extraction)

**✅ Implemented + validated** (`_chunk_text` + `max_episode_chars`, default 2000; 5 tests).
On real LoCoMo (conv-26, shared-graph) it **eliminated the extraction overflow** — 0
length-limit errors (vs many before) — and lifted recall from **all-zero → 2/6 partial**
(`locomo_real_chunked_2026-06-06.json`). Honest read: it's the necessary *unblocker*;
real-LoCoMo recall stays low because of extraction *quality* + multi-hop QA hardness (which
Lever 1 and richer retrieval attack), not overflow.

**Problem.** Real LoCoMo sessions are large multi-turn dialogues; extraction overflows the
16K completion budget → length-limit → retries → dropped edges → **recall 0** on `conv-26`
(run `bovh7xvbd`). The synthetic probes don't hit this because their sessions are short.

**Design.** Chunk each session body into sub-episodes under a configurable budget
(`max_episode_chars` / `max_episode_tokens`, default ~2–4k chars) **before** `add_episode`,
splitting on **turn boundaries** (never mid-turn) to keep relations intact, with stable
increasing valid-times within the session so order is preserved. Optionally a small overlap
to avoid splitting a relation across the boundary.

**Where.** Our layer — `sessions.py` (extend `_episodes` to sub-chunk) or
`graph._ingest_episodes`. Pure and unit-testable (chunk-boundary tests).

**Effort / risk.** Low–medium. Risk: a relation split across chunks is dropped (mitigate with
turn-boundary chunking + overlap); more episodes ⇒ more LLM calls (cost/time).

**Validation.** Re-run `locomo_real` (shared-graph): extraction completes with **no
length-limit errors** and **recall > 0**; report chunked vs unchunked. Pairs with Lever 1
(more facts captured → more for attributes/summaries to get right).

**Open questions.** Chunk size vs extraction quality sweet spot; overlap size; whether to
pre-summarize a session before ingest vs raw chunking.

---

## Lever 3 — Graphiti as a derived projection of the TBD graph (platform "one graph, many views")

**Problem.** In the platform, Graphiti must be a **derived** cross-episode lens over the
canonical work graph (`public.tbds` + the TBD ledger), *not a second source of truth* (the
One-Graph rule). Today it's standalone (benchmark only). The substrate already exists:
`supabase/schemas/parts/24_tbd_semantic.sql` (`tbds`, `tbd_embeddings`), `28_tbd_rpcs.sql`,
`31_tbd_semantic_triggers.sql`, and a `tbd-embed` edge function.

**Design.** A one-way **projection/sync**: TBD-ledger events (and TBD lifecycle changes) →
Graphiti episodes, `group_id = workspace/project`, one episode per ledger event, each
carrying `tbd_id` + ledger-event id as `source_description` so retrieval traces back to TBDs
(provenance). The graph is **rebuildable from the ledger** (idempotent projection — never
authoritative). Hosted Graphiti (a `services/memory-graph/` service) backs it; the platform
queries it for "what's been done / decided across episodes," distinct from the existing
`tbd_embeddings` *within-graph* semantic search.

**Where.** PLATFORM (app repo), not TYPHON: an edge function / worker reading the ledger +
the hosted Graphiti service. Cross-ref the `memory-graph` prototype and the "Memory Stack" /
"Memory, With a Past Tense" essays.

**Effort / risk.** High — a live platform service. Must honor the **enforcement-on** law:
memory writes → `audit_events`; retrieval carries provenance; governed by authz (dregg).
Risk: becoming a second source of truth (avoid — strictly derived); PII; hosting cost
(~+30% AWS, per the Magnolia hosting note).

**Validation.** Rebuild-from-ledger is idempotent (same graph from the same ledger);
retrieval returns TBD-grounded facts with provenance back to `tbd_id`; `audit_events` row per
memory write; a load test at workspace scale.

**Open questions.** Sync cadence (Realtime trigger vs batch worker); hosting model; how it
composes with `tbd_embeddings` semantic search; governance surface (grants for memory
read/write).

---

### Suggested order

1. **Lever 2** (chunking) — small, unblocks real-LoCoMo numbers, low risk.
2. **Lever 1** (attributes, then summary regen) — the real best-of-both; do attributes first
   (our layer) since they may suffice, then the fork regen if still leaking.
3. **Lever 3** (platform projection) — the productization; depends on 1–2 being solid and on
   platform hosting + governance decisions.
