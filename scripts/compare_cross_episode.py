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
import glob
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


def _load_saved(rescore_dir: Path, benchmark: str, baseline: str) -> list[dict]:
    """Load already-written per-sample artifacts for re-scoring (no LLM, no graph).

    A prior run wrote ``<dir>/<benchmark>__<baseline>/*.json``; re-scoring those with the
    current ``aggregate`` is free and deterministic, so metric changes can be re-applied to
    a frozen prediction set without paying for another live run.
    """
    pattern = str(rescore_dir / f"{benchmark}__{baseline}" / "*.json")
    artifacts = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            artifacts.append(json.load(handle))
    if not artifacts:
        raise SystemExit(f"no artifacts to re-score at {pattern}")
    return artifacts


def _row(label: str, a, g) -> str:
    return f"{label:34}{str(a):>12}{str(g):>12}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-episode baseline comparison.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "cross_episode" / "_runs"),
    )
    parser.add_argument(
        "--rescore-dir",
        default=None,
        help="Re-score already-written artifacts under this dir (no LLM/graph) instead of "
        "running live. Use to re-apply metric changes to a frozen prediction set.",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rescore_dir = Path(args.rescore_dir) if args.rescore_dir else None

    br = None if rescore_dir else BaselineRegistry.load()
    kr = None if rescore_dir else BenchmarkRegistry.load()

    report: dict = {"scenarios": {}}
    lines: list[str] = []
    for sc in SCENARIOS:
        def _artifacts(baseline: str) -> list[dict]:
            if rescore_dir:
                return _load_saved(rescore_dir, sc["benchmark"], baseline)
            return _run(br, kr, baseline, sc["benchmark"], out_dir, sc["chunk"], sc["window"])

        agg = {baseline: aggregate_artifacts(_artifacts(baseline)) for baseline in BASELINES}
        report["scenarios"][sc["name"]] = {"benchmark": sc["benchmark"], **agg}
        a, g = agg["attention_baseline"], agg["graphiti_cross_episode"]
        lines += ["", f"### {sc['name']}  (benchmark={sc['benchmark']}, window={sc['window']})", "=" * 58]
        lines.append(_row("metric", "attention", "graphiti"))
        lines.append("-" * 58)
        if sc["name"] == "supersession":
            n = a["supersession_n"]
            lines.append(_row("stale DOMINANT (real fail, lower)", f"{a['supersession_stale_dominant']}/{n}", f"{g['supersession_stale_dominant']}/{n}"))
            lines.append(_row("clean current-only (higher)", f"{a['supersession_clean']}/{n}", f"{g['supersession_clean']}/{n}"))
            lines.append(_row("current recalled (higher)", f"{a['supersession_current_recalled']}/{n}", f"{g['supersession_current_recalled']}/{n}"))
            lines.append(_row("stale present, loose (FYI)", f"{a['supersession_stale_leaked']}/{n}", f"{g['supersession_stale_leaked']}/{n}"))
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
