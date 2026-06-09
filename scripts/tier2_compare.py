"""Tier 2, Phase 1: RRF vs listwise-reranked retrieval on real-LoCoMo, end-to-end.

Tests whether reranking the graph-search candidates lifts answer quality. The cross_encoder
variant routes Graphiti's reranker through our backend-agnostic ListwiseReranker (the stock
OpenAIRerankerClient needs OpenAI-only logprobs and crashes on Together — see reranker.py).

Both variants ingest the SAME conversation into their OWN clean graph (Neo4j is wiped between
them, so the second can't see the first's episodes); only the search/rerank differs. Each is
then scored two ways: extractive token_recall AND the SOTA-axis J-score (an LLM composes an
answer from the retrieved facts, a second LLM grades it — the metric Mem0/Zep report).

One reproducible command produces the full comparison:
    TOGETHER_API_KEY=... GRAPH_URI=bolt://localhost:7687 GRAPH_USER=neo4j GRAPH_PASSWORD=... \
        python scripts/tier2_compare.py

Caveat: ingestion is re-run per variant (temp=0, so the two graphs are near-identical but not
bit-identical); a stricter design would ingest once and search the one graph two ways.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typhon.baselines.local_exact import run_baseline
from typhon.baselines.registry import BaselineRegistry
from typhon.benchmarks.registry import BenchmarkRegistry
from typhon.eval.aggregate import aggregate_artifacts
from typhon.eval.generation import decorate_artifact, make_chat_from_env

BENCHMARK = "locomo_real"

# (baseline_id, output-subdir) for each variant. The baseline_id selects the config
# (search_variant lives in its settings); the subdir keeps the two runs' artifacts apart.
VARIANTS = [
    ("graphiti_cross_episode", "graphiti_rrf"),
    ("graphiti_cross_episode_cross_encoder", "graphiti_cross_encoder"),
]


def _wipe_neo4j(container: str, user: str, password: str) -> None:
    """DETACH DELETE every node so each variant ingests into a clean graph."""
    try:
        subprocess.run(
            ["docker", "exec", container, "cypher-shell", "-u", user, "-p", password,
             "MATCH (n) DETACH DELETE n;"],
            check=True, capture_output=True, text=True, timeout=60,
        )
        print(f"   [neo4j] wiped {container}")
    except Exception as exc:  # noqa: BLE001
        print(f"   [neo4j] WARNING: wipe failed ({exc}); graphs may be contaminated")


def _run_variant(br, kr, baseline_id: str, out_dir: Path) -> list[dict]:
    return run_baseline(
        baseline_registry=br, benchmark_registry=kr, baseline_id=baseline_id,
        benchmark_id=BENCHMARK, family=None, output_dir=out_dir, dry_run=False,
        sample_source="local", sample_limit=None,
        chunk_size_override=None, local_window_tokens_override=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier 2 Phase 1: RRF vs listwise-reranked.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "cross_episode" / "_runs_real"),
    )
    parser.add_argument("--shared-key", default="conversation")
    parser.add_argument("--neo4j-container", default="sig-graphiti-neo4j")
    parser.add_argument("--no-judge", action="store_true", help="skip the J-score (token_recall only).")
    parser.add_argument("--price-per-1m", type=float, default=0.88)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GRAPHITI_SHARED_GRAPH_KEY", args.shared_key)
    neo4j_user = os.environ.get("GRAPH_USER", "neo4j")
    neo4j_pw = os.environ.get("GRAPH_PASSWORD", "password")

    chat = None if args.no_judge else make_chat_from_env()
    if not args.no_judge and chat is None:
        print("FATAL: no LLM key for judge (set TOGETHER_API_KEY) or pass --no-judge.", file=sys.stderr)
        return 1

    br = BaselineRegistry.load()
    kr = BenchmarkRegistry.load()

    print("=" * 84)
    print("Tier 2, Phase 1: RRF vs Listwise-Reranked Retrieval")
    print(f"Benchmark: {BENCHMARK} (shared_key={args.shared_key})  judge={not args.no_judge}")
    print("=" * 84)

    rows: dict[str, dict] = {}
    for i, (baseline_id, subdir) in enumerate(VARIANTS, 1):
        variant = "RRF" if subdir.endswith("rrf") else "Cross-Encoder (listwise)"
        print(f"\n{i}. {variant}  [{baseline_id}]")
        _wipe_neo4j(args.neo4j_container, neo4j_user, neo4j_pw)
        artifacts = _run_variant(br, kr, baseline_id, out_dir / subdir)
        n_ok = sum(1 for a in artifacts if a.get("status") == "ok")
        print(f"   ingested + searched: {len(artifacts)} QA ({n_ok} ok)")

        judge_chat = None
        if not args.no_judge:
            judge_chat = make_chat_from_env()  # fresh per variant -> per-variant judge cost
            assert judge_chat is not None
            for art in artifacts:
                decorate_artifact(art, judge_chat, generate=True, judge=True,
                                  judge_model=judge_chat.model, reader_model=judge_chat.model)
                # persist the verdict back into the frozen artifact
                p = Path(art["artifact_path"])
                if p.parent.exists():
                    p.write_text(json.dumps(art, indent=2), encoding="utf-8")
            print(f"   judged: {judge_chat.usage['total_tokens']} tok")

        agg = aggregate_artifacts(artifacts)
        rows[subdir] = {
            "baseline_id": baseline_id,
            "n": agg["n"],
            "mean_token_recall": agg.get("mean_token_recall") or 0.0,
            "llm_judge_accuracy": agg.get("llm_judge_accuracy"),
            "llm_judge_correct": agg.get("llm_judge_correct"),
            "llm_judge_n": agg.get("llm_judge_n"),
            "judge_tokens": judge_chat.usage["total_tokens"] if judge_chat else 0,
        }

    # --- comparison table ---
    print("\n" + "=" * 84)
    print("Results")
    print("=" * 84)
    header = f"{'variant':26}{'n':>4}{'token_recall':>14}{'J-acc':>8}{'correct':>9}{'judge_tok':>11}"
    print(header)
    print("-" * len(header))
    rrf = rows.get("graphiti_rrf", {})
    for subdir, r in rows.items():
        jacc = r["llm_judge_accuracy"]
        jacc_s = f"{jacc:.4f}" if isinstance(jacc, float) else "—"
        corr = f"{r['llm_judge_correct']}/{r['llm_judge_n']}" if r["llm_judge_n"] else "—"
        name = "RRF (baseline)" if subdir.endswith("rrf") else "Cross-Encoder rerank"
        print(f"{name:26}{r['n']:>4}{r['mean_token_recall']:>14.4f}{jacc_s:>8}{corr:>9}{r['judge_tokens']:>11}")

    # --- deltas vs RRF ---
    ce = rows.get("graphiti_cross_encoder", {})
    if rrf and ce:
        print("\nDeltas (Cross-Encoder vs RRF):")
        d_recall = ce["mean_token_recall"] - rrf["mean_token_recall"]
        print(f"  token_recall: {d_recall:+.4f}")
        if isinstance(ce.get("llm_judge_accuracy"), float) and isinstance(rrf.get("llm_judge_accuracy"), float):
            d_j = ce["llm_judge_accuracy"] - rrf["llm_judge_accuracy"]
            print(f"  J-score:      {d_j:+.4f}  ({rrf['llm_judge_accuracy']:.4f} -> {ce['llm_judge_accuracy']:.4f})")

    report = {"benchmark": BENCHMARK, "shared_key": args.shared_key, "variants": rows}
    (out_dir / "tier2_phase1_compare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'tier2_phase1_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
