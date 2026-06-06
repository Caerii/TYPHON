"""Aggregate and score baseline artifacts.

TYPHON baselines each emit a per-sample artifact with a ``prediction`` block
(``build_prediction_block``); this module rolls a list of those artifacts up into
per-baseline metrics so two baselines can be compared head-to-head on the same
benchmark. Beyond the standard token metrics it computes two cross-episode probes,
keyed off ``fixture.metadata``:

- ``stale_leaked`` (probe="supersession"): a fact was later changed; did the
  predicted answer still contain the ``stale_value``? (temporal correctness)
- ``window_hit`` (probe="window_recall"): the answer sat outside a bounded window;
  did retrieval recall it (token_recall > ``hit_threshold``)? (persistence)

Pure / dependency-free, so it is unit-testable and reusable by any runner.
"""

from __future__ import annotations

from statistics import mean
from typing import Any


def _norm(text: str | None) -> str:
    return " ".join((text or "").lower().split())


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

    return {
        "sample_id": fixture.get("sample_id"),
        "probe": probe,
        "status": artifact.get("status"),
        "token_f1": metrics.get("token_f1"),
        "token_recall": recall,
        "token_precision": metrics.get("token_precision"),
        "exact_match": bool(metrics.get("exact_match")),
        "retrieved_fact_count": cross_episode.get("retrieved_fact_count"),
        "stale_value": stale_value,
        "stale_leaked": bool(stale_value) and (_norm(str(stale_value)) in _norm(predicted)),
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
    return {
        "n": len(rows),
        "mean_token_recall": _safe_mean([r["token_recall"] for r in rows]),
        "mean_token_f1": _safe_mean([r["token_f1"] for r in rows]),
        "exact_match_count": sum(1 for r in rows if r["exact_match"]),
        "supersession_n": len(supers),
        "supersession_stale_leaked": sum(1 for r in supers if r["stale_leaked"]),
        "window_recall_n": len(windows),
        "window_recall_hits": sum(1 for r in windows if r["window_hit"]),
        "window_recall_mean_recall": _safe_mean([r["token_recall"] for r in windows]),
        "rows": rows,
    }
