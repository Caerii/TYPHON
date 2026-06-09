"""Aggregate and score baseline artifacts.

TYPHON baselines each emit a per-sample artifact with a ``prediction`` block
(``build_prediction_block``); this module rolls a list of those artifacts up into
per-baseline metrics so two baselines can be compared head-to-head on the same
benchmark. Beyond the standard token metrics it computes two cross-episode probes,
keyed off ``fixture.metadata``:

- supersession (probe="supersession"): a fact was later changed; did the answer give
  the *current* value, the *stale* one, or both? Measured with three booleans (below).
- ``window_hit`` (probe="window_recall"): the answer sat outside a bounded window;
  did retrieval recall it (token_recall > ``hit_threshold``)? (persistence)

**Supersession scoring — why three signals, not one.** The original single
``stale_leaked`` = "does the stale token appear anywhere in the answer" is fragile: it
fires even when the answer is *correct*, because a faithful current statement can name the
value it superseded (e.g. db: "SQLite … *replacing Postgres*" contains "Postgres"). So a
substring hit conflates "the stale value won" with "the answer correctly mentions history".
We disentangle them against ``reference_answers`` (the current value) and ``stale_value``:

- ``current_recalled`` = a reference (current) value is present — did we get it right at all?
- ``stale_dominant``  = stale present **and current absent** — the genuine failure (stale won).
- ``clean``           = current present **and stale absent** — the strict ideal (current-only).
- ``stale_leaked``    = stale present (the original loose signal; kept for continuity, but it
  over-counts — prefer ``stale_dominant`` for "did temporal correctness fail").

Pure / dependency-free, so it is unit-testable and reusable by any runner.
"""

from __future__ import annotations

from statistics import mean
from typing import Any


def _norm(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def _contains(haystack: str, needle: str | None) -> bool:
    """True if ``needle`` (normalized, non-empty) is a substring of ``haystack`` (normalized)."""
    n = _norm(str(needle)) if needle is not None else ""
    return bool(n) and n in _norm(haystack)


def score_artifact(artifact: dict[str, Any], *, hit_threshold: float = 0.5) -> dict[str, Any]:
    """Score a single baseline artifact into a flat row."""
    fixture = artifact.get("fixture") or {}
    metadata = fixture.get("metadata") or {}
    prediction = artifact.get("prediction") or {}
    metrics = prediction.get("metrics") or {}
    predicted = prediction.get("predicted_answer") or ""
    stale_value = metadata.get("stale_value")
    probe = metadata.get("probe", "other")
    recall = metrics.get("token_recall")

    cross_episode = (artifact.get("memory_state") or {}).get("cross_episode") or {}
    # LLM-as-judge verdict (binary correctness over the GENERATED answer — the SOTA axis), if a
    # generative-reader + judge pass decorated this artifact (typhon.eval.generation). None when
    # un-judged, so it never skews the count.
    judge = prediction.get("llm_judge") or {}
    llm_judge_correct = judge.get("correct") if isinstance(judge.get("correct"), bool) else None

    # Current (reference) value present? Supersession asks "what is true NOW", so the
    # reference answers ARE the current value; any one present means we recalled it.
    references = fixture.get("reference_answers") or prediction.get("reference_answers") or []
    current_recalled = any(_contains(predicted, ref) for ref in references)
    stale_present = _contains(predicted, stale_value)
    is_supersession = probe == "supersession"

    return {
        "sample_id": fixture.get("sample_id"),
        "probe": probe,
        "status": artifact.get("status"),
        "token_f1": metrics.get("token_f1"),
        "token_recall": recall,
        "token_precision": metrics.get("token_precision"),
        "exact_match": bool(metrics.get("exact_match")),
        "llm_judge_correct": llm_judge_correct,  # bool (SOTA J-score) or None if un-judged
        "retrieved_fact_count": cross_episode.get("retrieved_fact_count"),
        "stale_value": stale_value,
        # Supersession signals (None outside the supersession probe so they don't skew sums).
        "current_recalled": current_recalled if is_supersession else None,
        "stale_leaked": stale_present,  # loose: stale token present anywhere (over-counts)
        "stale_dominant": (stale_present and not current_recalled) if is_supersession else None,
        "clean": (current_recalled and not stale_present) if is_supersession else None,
        "window_hit": probe == "window_recall" and isinstance(recall, (int, float)) and recall > hit_threshold,
    }


def _safe_mean(values: list[Any]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(mean(nums), 4) if nums else None


def aggregate_artifacts(artifacts: list[dict[str, Any]], *, hit_threshold: float = 0.5) -> dict[str, Any]:
    """Roll a list of artifacts up into per-baseline aggregate metrics."""
    rows = [score_artifact(a, hit_threshold=hit_threshold) for a in artifacts]
    supers = [r for r in rows if r["probe"] == "supersession"]
    windows = [r for r in rows if r["probe"] == "window_recall"]
    judged = [r for r in rows if isinstance(r["llm_judge_correct"], bool)]
    judge_correct = sum(1 for r in judged if r["llm_judge_correct"])
    return {
        "n": len(rows),
        "mean_token_recall": _safe_mean([r["token_recall"] for r in rows]),
        "mean_token_f1": _safe_mean([r["token_f1"] for r in rows]),
        "exact_match_count": sum(1 for r in rows if r["exact_match"]),
        # LLM-as-judge accuracy (the SOTA axis): correct / judged. None when no judge ran.
        "llm_judge_n": len(judged),
        "llm_judge_correct": judge_correct,
        "llm_judge_accuracy": round(judge_correct / len(judged), 4) if judged else None,
        "supersession_n": len(supers),
        "supersession_stale_leaked": sum(1 for r in supers if r["stale_leaked"]),
        "supersession_stale_dominant": sum(1 for r in supers if r["stale_dominant"]),
        "supersession_clean": sum(1 for r in supers if r["clean"]),
        "supersession_current_recalled": sum(1 for r in supers if r["current_recalled"]),
        "window_recall_n": len(windows),
        "window_recall_hits": sum(1 for r in windows if r["window_hit"]),
        "window_recall_mean_recall": _safe_mean([r["token_recall"] for r in windows]),
        "rows": rows,
    }
