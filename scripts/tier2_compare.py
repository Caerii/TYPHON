"""Tier 2, Phase 1 comparison: RRF vs Cross-Encoder reranker on real-LoCoMo.

Phase 1 tests whether enabling Graphiti's cross-encoder reranker + BFS improves
retrieval, measured via token_recall. Runs two baseline variants side-by-side.

Expected improvement: +5-10% token_recall on multi-hop questions (q002, q003, q006).

Usage:
    TOGETHER_API_KEY=... GRAPH_URI=bolt://localhost:7687 GRAPH_PASSWORD=... \
        python scripts/tier2_compare.py [--shared-key conversation]

Then judge both runs with:
    TOGETHER_API_KEY=... python scripts/judge_run.py \
        --results-dir results/cross_episode/_runs_real \
        --baselines graphiti_rrf,graphiti_cross_encoder \
        --prefix tier2_phase1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typhon.baselines.registry import BaselineRegistry
from typhon.benchmarks.registry import BenchmarkRegistry
from typhon.baselines.local_exact import run_baseline
from typhon.eval.aggregate import aggregate_artifacts

BENCHMARK = "locomo_real"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tier 2 Phase 1: RRF vs Cross-Encoder token_recall comparison."
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "cross_episode" / "_runs_real"),
    )
    parser.add_argument(
        "--shared-key",
        default="conversation",
        help="Metadata key grouping a conversation's QA (shared ingestion).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GRAPHITI_SHARED_GRAPH_KEY", args.shared_key)

    br = BaselineRegistry.load()
    kr = BenchmarkRegistry.load()

    print("=" * 80)
    print("Tier 2, Phase 1: Cross-Encoder Reranker Comparison")
    print("=" * 80)
    print(f"Benchmark: {BENCHMARK} (shared_key={args.shared_key})")
    print()

    # Run RRF baseline
    print("1. Running RRF baseline (search_variant='rrf')...")
    artifacts_rrf = run_baseline(
        baseline_registry=br,
        benchmark_registry=kr,
        baseline_id="graphiti_cross_episode",
        benchmark_id=BENCHMARK,
        family=None,
        output_dir=out_dir / "graphiti_rrf",
        dry_run=False,
        sample_source="local",
        sample_limit=None,
        chunk_size_override=None,
        local_window_tokens_override=None,
    )
    agg_rrf = aggregate_artifacts(artifacts_rrf)
    print(f"   [OK] {len(artifacts_rrf)} artifacts")

    # Run Cross-Encoder variant
    print("\n2. Running cross-encoder variant (search_variant='cross_encoder')...")
    artifacts_ce = run_baseline(
        baseline_registry=br,
        benchmark_registry=kr,
        baseline_id="graphiti_cross_episode_cross_encoder",
        benchmark_id=BENCHMARK,
        family=None,
        output_dir=out_dir / "graphiti_cross_encoder",
        dry_run=False,
        sample_source="local",
        sample_limit=None,
        chunk_size_override=None,
        local_window_tokens_override=None,
    )
    agg_ce = aggregate_artifacts(artifacts_ce)
    print(f"   [OK] {len(artifacts_ce)} artifacts")

    # Aggregate and compare
    print("\n" + "=" * 80)
    print("Token Recall Results (extractive, sentence-level overlap)")
    print("=" * 80)
    print(f"{'Variant':25} {'n':>3}  {'Mean Recall':>14}  {'Improvement':>12}")
    print("-" * 70)

    token_recall_rrf = agg_rrf.get("mean_token_recall") or 0.0
    token_recall_ce = agg_ce.get("mean_token_recall") or 0.0

    improvement = ((token_recall_ce - token_recall_rrf) / token_recall_rrf * 100) if token_recall_rrf else 0

    print(f"{'RRF':25} {agg_rrf['n']:>3}  {token_recall_rrf:>14.4f}  {'—':>12}")
    print(f"{'Cross-Encoder':25} {agg_ce['n']:>3}  {token_recall_ce:>14.4f}  {improvement:+7.1f}%")

    print("\n" + "=" * 80)
    print("Next: Judge both runs for J-score (LLM binary correctness)")
    print("=" * 80)
    print(f"""
To evaluate J-score on both variants, run judge_run.py:

    TOGETHER_API_KEY=... python scripts/judge_run.py \\
        --results-dir {out_dir} \\
        --baselines graphiti_rrf,graphiti_cross_encoder \\
        --prefix tier2_phase1_judged

Expected: cross-encoder +5-10% on multi-hop questions (q002, q003, q006).
    """)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
