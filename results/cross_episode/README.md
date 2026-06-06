# Cross-episode memory: `graphiti_cross_episode` vs `attention_baseline`

Run date: 2026-06-06. Raw artifacts: `comparison_*_2026-06-06.{json,txt}`.

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

## Foundations added this pass

- `StrictSchemaClient` (strict structured outputs over Together).
- Guided extraction (`GUIDED_ENTITY_TYPES` + `GUIDED_EXTRACTION_INSTRUCTIONS`),
  session-marker stripping, `current_facts_only`/`prefer_current`, combined edge+node
  retrieval, and the `search_mode` + `node_text_mode` knobs (pure `_node_text` builder).
- Real-LoCoMo importer (`locomo_importer` + CLI + `locomo_real`) and an eval aggregation
  module (`typhon.eval.aggregate`).
- **TYPHON's first test suite** (`tests/`, 33 tests: session splitting, fact ordering,
  availability gating, strict-client wiring, node-text modes, dry-run artifact shape,
  importer mapping, eval aggregation). `pytest` config in `pyproject.toml`.

## Caveats / next

- Synthetic probe sets (8 supersession + 5 window), but **stable across 3 trials — std=0
  on every headline metric** (window_recall 5/5 and supersession 8/8 non-empty each trial;
  `trials_summary_2026-06-06.txt`). Deterministic at temp=0 once extraction is guided.
- Real LoCoMo is now **wired** (`locomo_real` benchmark + `locomo_importer` + CLI; attention
  runs, recall 0.0 on out-of-window answers). Efficient graphiti runs await
  **shared-graph-per-conversation** (ingest each conversation once).
- Best-of-both temporal precision + recall (bi-temporal node summaries).
- Fresh per-sample graph + synthetic valid-times remain (baseline `limitations`).
