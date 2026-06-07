"""Generalization + cost run on the *real* LoCoMo benchmark (not synthetic probes).

`locomo_real` is real LoCoMo QA: many multi-hop questions over one long, multi-session
conversation. It has no supersession/window probe tags, so the head-to-head here is general
cross-episode QA quality (token recall / exact match) plus the **cost** of the persistent
tier — the "Y" in "X% lift at Y cost". The conversation is ingested ONCE (shared-graph), so
graphiti's extraction cost is a one-time per-conversation cost amortized across its QA.

``attention_baseline`` runs dependency-free; ``graphiti_cross_episode`` needs a graph backend
+ an LLM (env: TOGETHER_API_KEY / GRAPH_*). Writes ``<output-dir>/locomo_real_compare.{json,txt}``.

Example:
    TOGETHER_API_KEY=... GRAPH_URI=bolt://localhost:7687 GRAPH_PASSWORD=... \
        python scripts/run_locomo_real.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typhon.baselines.local_exact import run_baseline  # noqa: E402
from typhon.baselines.registry import BaselineRegistry  # noqa: E402
from typhon.benchmarks.registry import BenchmarkRegistry  # noqa: E402
from typhon.eval.aggregate import aggregate_artifacts  # noqa: E402

BENCHMARK = "locomo_real"
BASELINES = ["attention_baseline", "graphiti_cross_episode"]


def _cost_summary(artifacts: list[dict]) -> dict:
    """Sum extraction cost across DISTINCT groups (shared-graph => one block repeated)."""
    by_group: dict[str, dict] = {}
    for art in artifacts:
        cost = art.get("cost") or {}
        gid = cost.get("group_id")
        if gid and gid not in by_group:
            by_group[gid] = cost
    return {
        "groups": len(by_group),
        "llm_calls": sum(c.get("llm_calls", 0) for c in by_group.values()),
        "total_tokens": sum(c.get("total_tokens", 0) for c in by_group.values()),
        "usd_estimate": round(sum(c.get("usd_estimate", 0.0) for c in by_group.values()), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-LoCoMo generalization + cost run.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "cross_episode" / "_runs_real"),
    )
    parser.add_argument("--shared-key", default="conversation",
                        help="metadata key that groups a conversation's QA (ingest once).")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ingest each conversation once; graphiti cost is then one-time per conversation.
    os.environ.setdefault("GRAPHITI_SHARED_GRAPH_KEY", args.shared_key)

    br = BaselineRegistry.load()
    kr = BenchmarkRegistry.load()

    report: dict = {"benchmark": BENCHMARK, "shared_key": args.shared_key, "baselines": {}}
    lines = [f"### {BENCHMARK}  (shared_key={args.shared_key})", "=" * 60,
             f"{'baseline':28}{'n':>4}{'mean_recall':>14}{'exact':>7}{'tokens':>10}{'usd':>10}"]
    for baseline in BASELINES:
        artifacts = run_baseline(
            baseline_registry=br, benchmark_registry=kr, baseline_id=baseline,
            benchmark_id=BENCHMARK, family=None, output_dir=out_dir / baseline,
            dry_run=False, sample_source="local", sample_limit=None,
            chunk_size_override=None, local_window_tokens_override=None,
        )
        agg = aggregate_artifacts(artifacts)
        cost = _cost_summary(artifacts)
        report["baselines"][baseline] = {"aggregate": agg, "cost": cost}
        lines.append(
            f"{baseline:28}{agg['n']:>4}{agg['mean_token_recall'] or 0.0:>14}"
            f"{agg['exact_match_count']:>7}{cost['total_tokens']:>10}{cost['usd_estimate']:>10}"
        )

    (out_dir / "locomo_real_compare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "locomo_real_compare.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_dir / 'locomo_real_compare.json'} and .txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
