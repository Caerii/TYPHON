"""Tests for typhon.eval.aggregate (baseline artifact scoring/rollup)."""

from __future__ import annotations

from typhon.eval.aggregate import aggregate_artifacts, score_artifact


def _art(probe, predicted, recall, *, stale_value=None, f1=0.0, exact=False):
    return {
        "status": "ok",
        "fixture": {"sample_id": "s", "metadata": {"probe": probe, "stale_value": stale_value}},
        "prediction": {
            "predicted_answer": predicted,
            "metrics": {
                "token_recall": recall,
                "token_f1": f1,
                "token_precision": 0.0,
                "exact_match": exact,
            },
        },
        "memory_state": {"cross_episode": {"retrieved_fact_count": 1}},
    }


def test_score_detects_stale_leak():
    row = score_artifact(_art("supersession", "the rate limit is 100 and now 40", 0.5, stale_value="100"))
    assert row["stale_leaked"] is True


def test_score_clean_supersession_no_leak():
    row = score_artifact(_art("supersession", "the rate limit is 40 per minute", 1.0, stale_value="100"))
    assert row["stale_leaked"] is False


def test_score_window_hit_threshold():
    assert score_artifact(_art("window_recall", "Lisbon office", 1.0))["window_hit"] is True
    assert score_artifact(_art("window_recall", "lunch spots", 0.0))["window_hit"] is False


def test_aggregate_rolls_up_probes():
    artifacts = [
        _art("supersession", "rate limit 100 and 40", 1.0, stale_value="100"),  # leaked
        _art("supersession", "Sam owns billing", 1.0, stale_value="Priya owns"),  # clean
        _art("window_recall", "Northwind is the partner", 1.0),  # hit
        _art("window_recall", "lunch spots near office", 0.0),  # miss
    ]
    agg = aggregate_artifacts(artifacts)
    assert agg["n"] == 4
    assert agg["supersession_n"] == 2
    assert agg["supersession_stale_leaked"] == 1
    assert agg["window_recall_n"] == 2
    assert agg["window_recall_hits"] == 1
    assert agg["mean_token_recall"] == 0.75
    assert len(agg["rows"]) == 4
