# SOTA agent memory (June 2026) — where we stand and how to get a lot better

Research synthesis for the `cross_episode` tier (`graphiti_cross_episode` baseline). Goal: understand
the state of the art and turn it into a ranked, concrete plan. Sources at the bottom.

## TL;DR — the one finding that reframes everything

Every headline SOTA number (Mem0 **92.5**, ByteRover **92–94%**, MemMachine **84.9%**, Zep **75%**) is an
**LLM-as-judge binary-correctness score over a *generated* answer**. Our **0.153 is extractive
token-recall** — a different metric on a different (narrower) prediction path (`build_prediction_block`
selects retrieved sentences; it never composes an answer, and we never judge correctness).
**The two are not comparable.** Before any optimization, the highest-ROI move is to climb onto the axis
everyone reports on: a **generative reader** + an **LLM-as-judge**. Until then "+66% relative recall" is a
real but unshippable retrieval diagnostic.

## SOTA landscape (treat the leaderboard skeptically — it's a vendor war)

| System | LoCoMo (J) | LongMemEval | Tokens/query | Notes |
|---|---|---|---|---|
| Mem0 (Apr-2026 algo) | **92.5** | **94.4** | ~6,900 | temporal +29.6, multi-hop +23.1 vs their 2025 algo |
| ByteRover 2.0 | 92–94% | — | — | leaderboard claim |
| MemMachine | 84.9% | — | — | claims to beat Mem0/Zep |
| Memori | 82.0% | — | — | |
| Zep / **Graphiti (our engine)** | 75.1% (disputed; Mem0 says 58.4%) | **71.2%** (temporal sub-task 63.8 vs Mem0 49) | **~600,000** | temporal-KG strength, but heavy |
| full-context baseline | **~73%** | — | ~26,000 | **beats Mem0's graph (~68%) on LoCoMo** |

Mem0's own per-category LoCoMo (LLM-judge): single-hop 67, **multi-hop 51**, open-domain 73, **temporal 55**.
Even the leader is weakest on multi-hop + temporal — exactly the categories a temporal KG should win.

**Two reframes:**
1. **LoCoMo is saturated.** Conversations are only 16–26K tokens — they fit in a modern context window, and a
   dumb full-context baseline (~73%) *beats* memory systems. Adversarial category 5 has broken ground truth.
   Chasing LoCoMo % is a dead end.
2. **The real benchmarks are bigger:** **LongMemEval** (LongMemEval_S ≈ 115K tokens / 30–40 sessions;
   LongMemEval_M ≈ 1.5M / 500 sessions) and **BEAM** (1M–10M). There memory beats context and SOTA *drops*
   (Mem0 64→48 from 1M→10M tokens). That is where temporal-KG memory should win.

## What the winners actually do

**Mem0 (cheapest + top score):**
- **Extract** from a rolling window (running summary + recent msgs + current pair), single LLM pass; agent-stated
  facts are first-class too.
- **Reconcile** each new fact against top-10 similar memories with an LLM that picks **ADD / UPDATE / DELETE /
  NOOP** (no separate classifier). Graph variant marks conflicting relations **invalid, not deleted** (≈ our
  bi-temporal invalidation / `Configurable`).
- **Multi-signal retrieval:** 3 parallel passes — semantic + keyword + **entity-match** — fused; entity matches
  *boost* the combined score. Plus reranking, async writes, timestamp-on-update.
- **Natural-language memory** ≈ 7–14K tokens/conv vs **Graphiti's ~600K** (≈ our 520K) — a ~40× cost gap.

**HippoRAG 2 (multi-hop SOTA):** dual-node KG (passage + phrase nodes, synonym links); query→triple embedding
seeds **Personalized PageRank** over the graph so activation spreads through entity chains — it can retrieve a
passage containing *none* of the query's words if it's linked through intermediates. +12.5% Recall@5 over
NER-to-node; +7% on associative tasks. This is the mechanism for the multi-hop questions we *lose*
(real-LoCoMo q002/q003/q006).

**LongMemEval's framework** (indexing → retrieval → reading): session decomposition (value granularity),
fact-augmented key expansion (indexing), **time-aware query expansion** (retrieval) — all portable.

## Where we stand (honest gap analysis)

| Axis | Us | SOTA | Gap |
|---|---|---|---|
| Answer path | extractive selection | generative + LLM-judge | ❌ not measuring the right thing |
| Cost | ~520K tok/conv ($0.46) | 7–14K (Mem0) | ❌ ~40× heavy (Graphiti side) |
| Multi-hop | loses 4/12 on real-LoCoMo | PPR graph traversal | ❌ no traversal/reranker |
| Temporal / supersession | `Configurable` + bi-temporal (done) | Mem0 just added it | ✅ genuine strength |
| Extraction robustness | 2 calls hit the 16K cap | chunked / multi-pass | ❌ caps recall |
| Reranking | RRF only (cross-encoder skipped) | reranker default | ❌ free lever unused |

## The plan — ranked by ROI, mapped to our stack

**Tier 1 — get onto the SOTA axis (do first; everything else is unmeasurable without it).**
1. **Generative answer synthesis** — an LLM "reader" composes an answer from retrieved facts (alongside, not
   replacing, the extractive diagnostic). Applied identically to *both* baselines, so it still isolates retrieval.
2. **LLM-as-judge** — a J-score (binary correctness, the LoCoMo/Mem0 protocol) in `eval`. Then re-run and report
   a number directly comparable to 92.5 / 75. Build it post-hoc over frozen artifacts (free re-judge, like
   `--rescore-dir`) so we reuse expensive ingestion.

**Tier 2 — retrieval power (the multi-hop/temporal wins).**
3. Turn on **Graphiti's reranker** + center-node / node-distance search for multi-hop (we use RRF only and skip the
   cross-encoder — free).
4. **PPR-style traversal** (HippoRAG 2) over the Graphiti graph for associative multi-hop.
5. **Query decomposition + time-aware query expansion** (LongMemEval recipe).

**Tier 3 — cost & robustness (deployable at SIG scale).**
6. **Fix the 16K extraction cap** (smaller chunks / multi-pass / swap extractor) — directly recovers lost recall.
7. **Cut token cost** — investigate Graphiti's ~40× overhead (cumulative summaries + per-node attribute passes +
   dedup calls); consider a Mem0-style cheap NL-memory tier.
8. **Async / background writes** (SOTA default).

**Tier 4 — pick the right fight + differentiate.**
9. **Target LongMemEval_S, not LoCoMo** — LoCoMo is saturated; LongMemEval is where temporal-KG memory beats
   context.
10. **SIG's moat = governed memory.** The 2026 security survey names **"mnemonic sovereignty"** — verifiable
    governance over what's written, who reads, when updates are authorized, what's auditable/forgettable — as an
    *open problem*. That is literally SIG's thesis (`audit_events`, capabilities, dregg). And our
    supersession/`Configurable` work already addresses Mem0's named open problem ("memory staleness: employer
    changes → confidently wrong"). No memory vendor in this race has governance — that's our differentiation,
    not a faster %.

## Status (2026-06-08): Tier 1 done, Tier 2 Phase 1 done

- **Tier 1 (on the SOTA axis):** generative reader + LLM-judge shipped (`typhon.eval.generation`,
  `scripts/judge_run.py`). First J-scores on real-LoCoMo: graphiti 16.7% vs attention 0%.
- **Tier 2, Phase 1 (reranker):** shipped a backend-agnostic listwise reranker (`reranker.py`) —
  Graphiti's stock cross-encoder needs OpenAI-only token logprobs and crashes on Together/local.
  Result: reranking **raised J-score 0.083→0.167** while *lowering* token_recall (it demotes the
  lexical-overlap distractors that inflate recall but don't answer). See
  `results/cross_episode/tier2_phase1_reranker_2026-06-08.{txt,json}`.
- **Still open:** Tier 2 Phases 2-3 (PPR multi-hop, query decomposition), Tier 3 (the 16K
  extraction cap + cost), Tier 4 (LongMemEval + governed memory).

## Three further directions (memory as a *sleep/consolidation* system)

Beyond the ranked tiers, a frame worth pursuing: SOTA memory systems only *write on ingest*. Human
memory also writes *offline*, during sleep — replay, consolidation, pruning. Three concrete ideas:

1. **The Dream Pass — offline consolidation as a first-class write path.** A periodic background job
   that *replays* recent episodes over the graph: re-weight edges by traversal frequency, merge
   duplicate entities, decay/retire stale nodes, and **precompute the spreading-activation (PPR)
   landscape** so morning retrieval is cheap (turns the expensive part of HippoRAG-2 into an offline
   cost). Maps cleanly onto Graphiti as a scheduled "consolidate(group_id)" pass.
2. **Confidence-superposition retrieval.** Instead of a hard current-vs-superseded cut at read time,
   return the bi-temporal candidates *together*, each with a **time-decayed confidence**, and let the
   reader weigh them — some questions want the *old* value. Generalizes today's `node_text_mode` /
   `current_facts_only` into a soft, query-conditioned blend.
3. **Lethe — a governed forgetting budget.** Forgetting as a first-class, *audited* operation (not an
   accident of TTLs): the "right to be forgotten" as a memory primitive on the audit ledger, behind a
   capability. This is precisely the **"mnemonic sovereignty"** open problem (arXiv:2604.16548) we
   named as SIG's differentiation — no memory vendor in the race governs *deletion*.

## Sources
- Zep — "Is Mem0 Really SOTA in Agent Memory?" https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/
- Mem0 — "State of AI Agent Memory 2026" https://mem0.ai/blog/state-of-ai-agent-memory-2026
- Mem0 paper, arXiv:2504.19413 https://arxiv.org/html/2504.19413v1
- Zep 84% correction (zep-papers #5) https://github.com/getzep/zep-papers/issues/5
- LongMemEval, arXiv:2410.10813 (ICLR 2025) https://arxiv.org/abs/2410.10813
- HippoRAG 2 — MarkTechPost https://www.marktechpost.com/2025/03/03/hipporag-2-advancing-long-term-memory-and-contextual-retrieval-in-large-language-models/ ; repo https://github.com/osu-nlp-group/hipporag
- Graph-based Agent Memory survey, arXiv:2602.05665 https://arxiv.org/html/2602.05665v1
- Security of Long-Term Memory / "mnemonic sovereignty", arXiv:2604.16548 https://arxiv.org/html/2604.16548v1
- ByteRover LoCoMo leaderboard https://www.byterover.dev/blog/benchmark-ai-agent-memory ; MemMachine https://memmachine.ai/blog/2025/09/memmachine-reaches-new-heights-on-locomo/
