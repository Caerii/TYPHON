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

`_order_facts` ranks current edges before current nodes (edges are bi-temporally
authoritative), but predict_answer's top-2 still selects the high-overlap stale sentence
from a node summary. `search_mode` is configurable (`edges` | `combined`); default is
`combined` (recall-max). The best-of-both — bi-temporal / current-only node summaries —
is tracked as future work.

## Foundations added this pass

- `StrictSchemaClient` (strict structured outputs over Together).
- Guided extraction (`GUIDED_ENTITY_TYPES` + `GUIDED_EXTRACTION_INSTRUCTIONS`),
  session-marker stripping, `current_facts_only`/`prefer_current`, combined edge+node
  retrieval, `search_mode` knob.
- **TYPHON's first test suite** (`tests/test_graphiti_cross_episode.py`, 17 tests:
  session splitting, fact ordering, availability gating, strict-client wiring, dry-run
  artifact shape). `pytest` config in `pyproject.toml`.

## Caveats / next

- Synthetic probe sets (8 supersession + 5 window). `run_trials.sh` runs N fresh trials
  for mean±std (extraction is stochastic).
- Real LoCoMo (`locomo10.json`, 10 conv / ~1986 QA, real timestamps) fetched + scoped;
  needs a **shared-graph-per-conversation** mode (ingest each conversation once).
- Best-of-both temporal precision + recall (bi-temporal node summaries).
- Fresh per-sample graph + synthetic valid-times remain (baseline `limitations`).
