# Cross-episode memory: `graphiti_cross_episode` vs `attention_baseline`

Run date: 2026-06-06. Raw artifacts: `comparison_*_2026-06-06.{json,txt}`.

> **Latest result (same day): the Lever 1 best-of-both.** Structured current-attributes
> cut the supersession leak **3/3 → 1/3 while keeping window_recall 5/5 at full recall
> (1.0)** — the best-of-both the tradeoff section below called unreachable at retrieval
> time. See **[§ Lever 1](#lever-1--structured-current-attributes-best-of-both-realized-2026-06-06)**;
> snapshot `comparison_lever1_attributes_2026-06-06.{json,txt}`. The sections above are
> the journey that located the tradeoff Lever 1 then resolved.

## Stack

- **Extraction LLM:** `meta-llama/Llama-3.3-70B-Instruct-Turbo` via **Together.ai**.
  Cheaper Qwen3-235B (`-tput`, $0.20/$0.60) is a stronger extractor in isolation but
  **over-generates under strict constrained decoding** (fills the 16K budget without
  closing JSON → parse fails); Llama-3.3-70B terminates cleanly.
- **Embedder:** `intfloat/multilingual-e5-large-instruct` (1024-dim) — only serverless
  embedder on Together.
- **Graph:** Neo4j 5 (local). **Schema:** `StrictSchemaClient` routes schema'd calls
  through the OpenAI SDK `beta.chat.completions.parse` path (strict constrained decoding,
  honored by Together) — the stock generic client's non-strict `json_schema` silently
  dropped value/attribute edges.

## Design

Both baselines share the same extractive prediction (`build_prediction_block`), so
retrieval is the only variable. Two scenarios isolate the two claims:

- **supersession** (`locomo`, full window): a fact is stated then changed — does the
  answer leak the *stale* value? (temporal correctness)
- **window_recall** (`locomo_window`, `chunk=40/window=40` → 1 accessible chunk): the
  answer-bearing session is pushed **out** of the attention window — can the persistent
  store still recall it? (persistence). The attention window is a hard filter
  (`local_exact.py:171,208`), so the session-1 answer is provably unreachable for attention.

## What moved the numbers (each fix compounded)

| change | window_recall (graphiti) | supersession stale-leak (graphiti) |
|---|---|---|
| edge-only search, no guidance | 1/5 (0.20) | 0/3 |
| **+ guided extraction** (`entity_types` + instructions → capture values/roles/orgs as nodes) | 3/5 (0.60) | 0/3 |
| **+ combined edge+node search** (`COMBINED_HYBRID_SEARCH_RRF`, surfaces node name/summary/attrs) | **5/5 (1.00)** | **3/3** ⚠️ |

Attention is **0/5** on window_recall throughout (it can only see the filler session)
and leaks **3/3** on supersession throughout (top-2 extraction grabs both current and
stale sentences).

## The headline result (combined search)

| scenario | metric | attention | graphiti |
|---|---|---|---|
| window_recall | recall / hits | **0.0 / 0/5** | **1.0 / 5/5** |
| supersession | stale leaked (lower better) | 3/3 | 3/3 |
| supersession | mean recall | 0.90 | 0.85 |

**The win:** a persistent memory tier recalls **every** fact a bounded window
structurally cannot (0/5 → 5/5: Dana, Lisbon, Pixel, Marcus, Northwind).

**The honest tradeoff:** node summaries are what unlocked that recall — but Graphiti's
node `summary` aggregates history in present tense ("public API rate limit is 100…
lowered to 40"; "Priya owns billing… moved away"), so combined search **reintroduces
the stale leak** (0/3 → 3/3). Edges carry bi-temporal validity and are clean (0/3) but
lossy. **Neither single mode wins both axes.**

Two knobs navigate this: `search_mode` (`edges` | `combined`, default `combined`) and
`node_text_mode` (`summary` | `current_edges`, default `summary`). Measured:

| `node_text_mode` | window_recall | superseded leak | supersession recall |
|---|---|---|---|
| `summary` (default, recall-max) | 5/5 | 3/3 | 0.85 |
| `current_edges` (top-K current edges) | 5/5 | **1/3** | 0.37 |
| `bitemporal` (full-graph current edges; drop stale-only nodes) | 5/5 | **1/3** | 0.37 |

The precision modes drop the history-aggregating summary (name + current incident edges);
`bitemporal` additionally fetches each node's *full* incident-edge set and drops stale-only
nodes. Both keep window_recall perfect (5/5) and cut the leak to 1/3, but lose value/attribute
recall (0.85 → 0.37): those facts (`40 rpm`, `March 14`, `peanuts`) live in the node
**summary**, which Graphiti generates from *all* edges at write-time. The last leak is
irreducible at retrieval time — the stale value *is* an entity name (`Postgres`) that still
carries a current edge.

**The located limit:** entity-name facts and value facts want opposite retrieval modes, and
no retrieval-time knob resolves both.

**Write-time fix attempted in the fork — and it is not a quick win** (see
`comparison_forkfix_attempt_2026-06-06.txt`). We forked graphiti-core
(`superintelligent-graphiti`) and filtered superseded edges out of the node-summary build
(`_build_edges_by_node`, gated by `GRAPHITI_SUMMARIZE_CURRENT_ONLY`). It made **no
difference** (supersession leak stayed 3/3), because the leak is not in the edge-fact
append we filtered: the node `summary` is **LLM-generated and cumulative**, written from
session 1's episode (`"public API rate limit is 100 requests per minute"`) and *not
regenerated* when session 3 supersedes it — and for that sample the graph had **zero
edges**, so the filter never applied. We reverted the change.

Even a full summary *regeneration* (rebuild each node's summary from current edges only)
would not fully resolve it: the current value `40` was never captured as an edge
(extraction quality), and a legitimate current fact can mention the old value (`db`:
"use SQLite, replacing Postgres"). **So the recall↔precision tradeoff is fundamental to
Graphiti's design (LLM-written cumulative summaries) plus cheap-LLM extraction — not a
3-line fix.** The real resolution is two harder pieces: (1) extraction that records
value-updates as **structured, current node attributes**, and (2) **regenerating** node
summaries on supersession. Until then, navigate the tradeoff with `node_text_mode`.

## Lever 1 — structured current-attributes (best-of-both, realized 2026-06-06)

The two-part resolution the tradeoff section predicted — "(1) structured current node
attributes, (2) summary regeneration" — is now built and measured for part (1), and **part
(1) alone reaches the best-of-both** for the cases with extractable structure. Snapshot:
`comparison_lever1_attributes_2026-06-06.{json,txt}`.

**Mechanism (our layer — no fork divergence):**

- A typed `Configurable {current_value}` entity type — the *only* typed entity, so the
  per-node attribute-extraction pass (and its LLM cost) runs **only** for changeable things.
  Graphiti re-extracts node attributes per episode with the prior attributes as context and
  merges them, so `current_value` is **updated to the current value on supersession**
  (validated end-to-end: rate limit 100→40; committed live regression in
  `tests/test_graphiti_live.py`).
- A `node_text_mode="attributes"` retrieval mode that is a layered preference:
  structured attribute → else current incident edges (bi-temporally clean) → else *drop*
  stale-only nodes → else (edgeless only) the summary. It **never** returns the
  history-aggregating summary when a currency signal exists.

**Result.** *First, the metric was fixed* — the original `stale_leaked` (does the stale token
appear anywhere?) is fragile: it fires even on a correct answer that faithfully names the value
it superseded (db: "SQLite … *replacing Postgres*"). It is now disentangled into three signals
(`aggregate.py`): `stale_dominant` (stale present **and** current absent — the real failure),
`clean` (current present **and** stale absent — the strict ideal), and `current_recalled`
(current present at all). The loose `stale_leaked` is kept for continuity but over-counts.

| metric (supersession, n=3) | attention | graphiti (lever 1) |
|---|---|---|
| **stale DOMINANT** — stale won (lower better) | 0/3 | **0/3** |
| **clean current-only** (higher better) | 0/3 | **2/3** |
| current recalled | 3/3 | 3/3 |
| stale present — loose/old (FYI) | 3/3 | 1/3 |
| window_recall — recall / hits | 0.0 / 0/5 | **1.0 / 5/5** |

The true story the refined metric tells: **neither baseline lets the stale value *dominate*
(both recall the in-window current value), but graphiti returns a *clean current-only* answer
2/3 of the time where attention dumps both values every time (clean 0/3).** That cleanliness —
plus the categorical window_recall win (5/5 vs 0/5) and full value recall (1.0, not the 0.37 of
the precision-only `current_edges`/`bitemporal` modes) — is the best-of-both: the value now
lives in a clean structured attribute instead of only in the cumulative summary. Per case:

- **rate limit (100→40): CLEAN** — `current_value` carries 40; a single retrieved fact, no "100".
- **owner (Priya→Sam): CLEAN** — extraction formed bi-temporally invalidated edges
  ("Priya owns" → invalid, "Sam owns" → current); the attributes mode renders nodes from
  current edges and drops the stale summary.
- **db (Postgres→SQLite): recalled + not stale-dominant, but not *clean*.** It recalls the
  current value (SQLite) and the stale value does **not** dominate, so it is **not a real
  failure** under the fixed metric — but it cannot be `clean` because **the correct current
  fact itself names the old value**: SQLite's faithful summary is "SQLite will be used …
  *replacing the initial choice of Postgres*". Two things conspire: extraction modeled
  Postgres/SQLite as separate *edgeless* `System` nodes (no `Configurable` slot, no
  invalidatable edge → no currency signal to drop the stale node), and even a perfect retrieval
  can't strip "Postgres" from a faithful current statement. Only abstractive answer generation
  or a reliably-extracted `current_value` slot would make it clean; forcing the latter via
  extraction instructions (anchor the project so a `--uses-->` edge forms and supersedes)
  **regressed** window_recall (5/5→4/5), so it was reverted (`_runs_lever1c`) — a useful
  negative: pushing extraction past its reliable envelope costs recall elsewhere.

**Honest bottom line:** structured current-attributes turn the recall↔precision tradeoff from
"pick one" into best-of-both: **no stale-domination (0/3), clean current-only answers 2/3**
(vs attention 0/3) *at full value recall (1.0)*, plus the categorical window_recall win. The
one non-`clean` case (db) is not a memory failure — the source's faithful current statement
names the superseded value, so only abstractive answer generation could make it clean. Part (2)
of the
prediction (fork summary-regeneration) was therefore **not** pursued: it provably cannot help
db (Postgres is edgeless; SQLite's summary legitimately names it), and the structured-attribute
path resolves the cases summary-regen was meant to.

## Foundations added this pass

- `StrictSchemaClient` (strict structured outputs over Together).
- Guided extraction (`GUIDED_ENTITY_TYPES` + `GUIDED_EXTRACTION_INSTRUCTIONS`),
  session-marker stripping, `current_facts_only`/`prefer_current`, combined edge+node
  retrieval, and the `search_mode` + `node_text_mode` knobs (pure `_node_text` builder).
- Real-LoCoMo importer (`locomo_importer` + CLI + `locomo_real`) and an eval aggregation
  module (`typhon.eval.aggregate`).
- **TYPHON's first test suite** (`tests/`, 37 pure tests: session splitting, fact ordering,
  availability gating, strict-client wiring, node-text modes incl. `attributes`, dry-run
  artifact shape, importer mapping, eval aggregation) **plus an opt-in live integration test**
  (`test_graphiti_live.py`, skipped without a backend) that pins the supersession fix.
  `pytest` config in `pyproject.toml`.

## Caveats / next

- Synthetic probe sets (8 supersession + 5 window), but **stable across 3 trials — std=0
  on every headline metric** (window_recall 5/5 and supersession 8/8 non-empty each trial;
  `trials_summary_2026-06-06.txt`). Deterministic at temp=0 once extraction is guided.
- Real LoCoMo is now **wired** (`locomo_real` benchmark + `locomo_importer` + CLI; attention
  runs, recall 0.0 on out-of-window answers). Efficient graphiti runs await
  **shared-graph-per-conversation** (ingest each conversation once).
- ✅ Best-of-both temporal precision + recall — **done via Lever 1** (structured
  current-attributes + the `attributes` retrieval mode; see the Lever 1 section). Residual:
  the db prose-only / named-entity case (the floor, partly a metric artifact).
- Fresh per-sample graph + synthetic valid-times remain (baseline `limitations`).
