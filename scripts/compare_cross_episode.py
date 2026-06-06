"""Reproduce the cross-episode comparison (the numbers in ``results/cross_episode/``).

Runs ``attention_baseline`` and ``graphiti_cross_episode`` on the cross-episode probe
benchmarks and scores them with ``typhon.eval.aggregate``. Two scenarios isolate the two
claims a long-term memory tier should satisfy:

  - **supersession** (``locomo``, full window): a fact is stated then changed — does the
    answer leak the *stale* value? (temporal correctness)
  - **window_recall** (``locomo_window``, bounded window via ``--window``/``--chunk``): the
    answer-bearing session is pushed *out* of the attention window — can the persistent
    store still recall it? (persistence)

``attention_baseline`` runs with no external deps; ``graphiti_cross_episode`` needs a graph
backend + an LLM (env: ``TOGETHER_API_KEY`` / ``LLM_*`` / ``GRAPH_*``) and degrades to
``not_executed`` artifacts otherwise. Writes ``<output-dir>/comparison.{json,txt}``.

Example:
    GRAPHITI_SHARED_GRAPH_KEY= TOGETHER_API_KEY=... GRAPH_URI=bolt://localhost:7687 \
        python scripts/compare_cross_episode.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typhon.baselines.local_exact import run_baseline  # noqa: E402
from typhon.baselines.registry import BaselineRegistry  # noqa: E402
from typhon.benchmarks.registry import BenchmarkRegistry  # noqa: E402
from typhon.eval.aggregate import aggregate_artifacts  # noqa: E402

BASELINES = ["attention_baseline", "graphiti_cross_episode"]
SCENARIOS = [
    {"name": "supersession", "benchmark": "locomo", "chunk": None, "window": None},
    {"name": "window_recall", "benchmark": "locomo_window", "chunk": 40, "window": 40},
]


def _run(br, kr, baseline, benchmark, out_dir, chunk, window):
    return run_baseline(
        baseline_registry=br,
        benchmark_registry=kr,
        baseline_id=baseline,
        benchmark_id=benchmark,
        family=None,
        output_dir=out_dir / f"{benchmark}__{baseline}",
        dry_run=False,
        sample_source="local",
        sample_limit=None,
        chunk_size_override=chunk,
        local_window_tokens_override=window,
    )


def _row(label: str, a, g) -> str:
    return f"{label:34}{str(a):>12}{str(g):>12}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-episode baseline comparison.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "cross_episode" / "_runs"),
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    br = BaselineRegistry.load()
    kr = BenchmarkRegistry.load()

    report: dict = {"scenarios": {}}
    lines: list[str] = []
    for sc in SCENARIOS:
        agg = {
            baseline: aggregate_artifacts(_run(br, kr, baseline, sc["benchmark"], out_dir, sc["chunk"], sc["window"]))
            for baseline in BASELINES
        }
        report["scenarios"][sc["name"]] = {"benchmark": sc["benchmark"], **agg}
        a, g = agg["attention_baseline"], agg["graphiti_cross_episode"]
        lines += ["", f"### {sc['name']}  (benchmark={sc['benchmark']}, window={sc['window']})", "=" * 58]
        lines.append(_row("metric", "attention", "graphiti"))
        lines.append("-" * 58)
        if sc["name"] == "supersession":
            lines.append(_row("stale LEAKED (lower better)", f"{a['supersession_stale_leaked']}/{a['supersession_n']}", f"{g['supersession_stale_leaked']}/{g['supersession_n']}"))
        else:
            lines.append(_row("window_recall hits", f"{a['window_recall_hits']}/{a['window_recall_n']}", f"{g['window_recall_hits']}/{g['window_recall_n']}"))
            lines.append(_row("window_recall mean recall", a["window_recall_mean_recall"], g["window_recall_mean_recall"]))
        lines.append(_row("mean token_recall", a["mean_token_recall"], g["mean_token_recall"]))

    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_dir / 'comparison.json'} and comparison.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
