"""Post-hoc generative-reader + LLM-as-judge over a frozen results dir -> SOTA-comparable J-score.

SOTA agent-memory numbers (Mem0/Zep/...) are an LLM-judge binary-correctness score over a
*generated* answer, not extractive token-recall. This driver climbs onto that axis WITHOUT
re-running ingestion: it reads each baseline's frozen artifacts, has an LLM compose an answer from
the *same* retrieved facts (``typhon.eval.generation``), grades it against the reference(s), and
reports per-baseline LLM-judge accuracy alongside the existing token metrics. The identical
reader+judge runs over every baseline, so the head-to-head still isolates retrieval.

Needs an LLM key (``TOGETHER_API_KEY`` / ``LLM_API_KEY``); writes the verdict back into each
artifact and a ``<results-dir>/<prefix>_judged.{json,txt}`` summary.

Example:
    TOGETHER_API_KEY=... python scripts/judge_run.py \
        --results-dir results/cross_episode/_runs_real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typhon.eval.aggregate import aggregate_artifacts  # noqa: E402
from typhon.eval.generation import decorate_artifact, make_chat_from_env  # noqa: E402

BASELINES = ["attention_baseline", "graphiti_cross_episode"]


def _load(dirpath: Path) -> list[tuple[Path, dict]]:
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(dirpath.glob("*.json"))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-hoc generative reader + LLM-judge (J-score).")
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "cross_episode" / "_runs_real"),
    )
    parser.add_argument("--baselines", nargs="*", default=BASELINES)
    parser.add_argument("--prefix", default="locomo_real", help="output filename prefix.")
    parser.add_argument("--limit", type=int, default=None, help="judge only the first N artifacts/baseline.")
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="judge the EXTRACTIVE predicted_answer instead of a freshly generated one.",
    )
    parser.add_argument("--price-per-1m", type=float, default=0.88, help="$/1M tokens for reader+judge.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if make_chat_from_env() is None:
        print("FATAL: no LLM key (set TOGETHER_API_KEY or LLM_API_KEY).", file=sys.stderr)
        return 1

    report: dict = {"results_dir": str(results_dir), "generate": not args.no_generate, "baselines": {}}
    lines = [
        f"### LLM-judge (J-score) over {results_dir.name}  (generate={not args.no_generate})",
        "=" * 72,
        f"{'baseline':28}{'n':>4}{'J-acc':>8}{'correct':>9}{'mean_recall':>13}{'judge_tok':>11}{'usd':>9}",
    ]
    for baseline in args.baselines:
        bdir = results_dir / baseline
        if not bdir.is_dir():
            continue
        chat = make_chat_from_env()  # fresh per baseline so cost is attributed per baseline
        assert chat is not None
        loaded = _load(bdir)
        if args.limit:
            loaded = loaded[: args.limit]
        artifacts: list[dict] = []
        for path, art in loaded:
            decorate_artifact(
                art, chat, generate=not args.no_generate, judge=True,
                judge_model=chat.model, reader_model=chat.model,
            )
            path.write_text(json.dumps(art, indent=2), encoding="utf-8")
            artifacts.append(art)
        agg = aggregate_artifacts(artifacts)
        cost = {
            "llm_calls": chat.usage["llm_calls"],
            "total_tokens": chat.usage["total_tokens"],
            "usd_estimate": round(chat.usage["total_tokens"] / 1_000_000 * args.price_per_1m, 6),
        }
        report["baselines"][baseline] = {
            "aggregate": {k: v for k, v in agg.items() if k != "rows"},
            "judge_cost": cost,
        }
        lines.append(
            f"{baseline:28}{agg['llm_judge_n']:>4}{agg['llm_judge_accuracy'] or 0.0:>8}"
            f"{agg['llm_judge_correct']:>9}{agg['mean_token_recall'] or 0.0:>13}"
            f"{cost['total_tokens']:>11}{cost['usd_estimate']:>9}"
        )

    out_json = results_dir / f"{args.prefix}_judged.json"
    out_txt = results_dir / f"{args.prefix}_judged.txt"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_json} and .txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
