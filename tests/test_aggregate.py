"""Tests for typhon.eval.aggregate (baseline artifact scoring/rollup)."""

from __future__ import annotations

from typhon.eval.aggregate import aggregate_artifacts, score_artifact


def _art(probe, predicted, recall, *, stale_value=None, references=None, f1=0.0, exact=False, judge=None):
    prediction = {
        "predicted_answer": predicted,
        "metrics": {
            "token_recall": recall,
            "token_f1": f1,
            "token_precision": 0.0,
            "exact_match": exact,
        },
    }
    if judge is not None:
        prediction["llm_judge"] = {"correct": judge, "target": "generative"}
    return {
        "status": "ok",
        "fixture": {
            "sample_id": "s",
            "reference_answers": references or [],
            "metadata": {"probe": probe, "stale_value": stale_value},
        },
        "prediction": prediction,
        "memory_state": {"cross_episode": {"retrieved_fact_count": 1}},
    }


def test_score_detects_stale_leak():
    row = score_artifact(_art("supersession", "the rate limit is 100 and now 40", 0.5, stale_value="100"))
    assert row["stale_leaked"] is True


def test_score_clean_supersession_no_leak():
    row = score_artifact(_art("supersession", "the rate limit is 40 per minute", 1.0, stale_value="100"))
    assert row["stale_leaked"] is False


def test_score_clean_supersession_is_clean_and_recalled():
    # current value present, stale absent -> clean + recalled, not dominant.
    row = score_artifact(
        _art("supersession", "the rate limit is 40 per minute", 1.0,
             stale_value="100", references=["40 per minute"])
    )
    assert row["current_recalled"] is True
    assert row["clean"] is True
    assert row["stale_dominant"] is False


def test_score_correct_answer_that_mentions_history_is_not_dominant():
    # THE METRIC-FIX CASE (db): a faithful current answer names the value it superseded.
    # Loose stale_leaked fires (over-counts), but the stale value did NOT win: current is
    # present, so stale_dominant is False. clean is False (stale token present).
    row = score_artifact(
        _art("supersession", "SQLite will be used, replacing the initial choice of Postgres", 1.0,
             stale_value="Postgres", references=["SQLite"])
    )
    assert row["stale_leaked"] is True       # loose signal over-counts
    assert row["current_recalled"] is True
    assert row["stale_dominant"] is False    # the real failure did NOT occur
    assert row["clean"] is False             # but it's not a clean current-only answer


def test_score_stale_dominant_when_current_absent():
    # Only the stale value present, current missing -> the genuine failure.
    row = score_artifact(
        _art("supersession", "the rate limit is 100 requests per minute", 0.0,
             stale_value="100", references=["40 per minute"])
    )
    assert row["current_recalled"] is False
    assert row["stale_dominant"] is True
    assert row["clean"] is False


def test_supersession_signals_are_none_outside_probe():
    row = score_artifact(_art("window_recall", "Lisbon", 1.0, references=["Lisbon"]))
    assert row["stale_dominant"] is None
    assert row["clean"] is None
    assert row["current_recalled"] is None


def test_score_window_hit_threshold():
    assert score_artifact(_art("window_recall", "Lisbon office", 1.0))["window_hit"] is True
    assert score_artifact(_art("window_recall", "lunch spots", 0.0))["window_hit"] is False


def test_aggregate_rolls_up_probes():
    artifacts = [
        # current + stale both present -> recalled, not dominant, not clean (loose-leaked).
        _art("supersession", "rate limit 100 and 40 per minute", 1.0,
             stale_value="100", references=["40 per minute"]),
        # current present, stale absent -> clean.
        _art("supersession", "Sam owns billing", 1.0,
             stale_value="Priya owns", references=["Sam"]),
        _art("window_recall", "Northwind is the partner", 1.0),  # hit
        _art("window_recall", "lunch spots near office", 0.0),  # miss
    ]
    agg = aggregate_artifacts(artifacts)
    assert agg["n"] == 4
    assert agg["supersession_n"] == 2
    assert agg["supersession_stale_leaked"] == 1       # loose: only the first has its stale token
    assert agg["supersession_stale_dominant"] == 0     # neither let the stale value win
    assert agg["supersession_clean"] == 1              # only the second is clean current-only
    assert agg["supersession_current_recalled"] == 2   # both recalled the current value
    assert agg["window_recall_n"] == 2
    assert agg["window_recall_hits"] == 1
    assert len(agg["rows"]) == 4


def test_score_carries_llm_judge_verdict():
    assert score_artifact(_art("other", "Lisbon", 1.0, judge=True))["llm_judge_correct"] is True
    assert score_artifact(_art("other", "Berlin", 0.0, judge=False))["llm_judge_correct"] is False
    # un-judged artifact -> None, so it never counts toward J-score.
    assert score_artifact(_art("other", "Lisbon", 1.0))["llm_judge_correct"] is None


def test_aggregate_llm_judge_accuracy():
    artifacts = [
        _art("other", "a", 1.0, judge=True),
        _art("other", "b", 1.0, judge=True),
        _art("other", "c", 0.0, judge=False),
        _art("other", "d", 0.0, judge=False),
    ]
    agg = aggregate_artifacts(artifacts)
    assert agg["llm_judge_n"] == 4
    assert agg["llm_judge_correct"] == 2
    assert agg["llm_judge_accuracy"] == 0.5


def test_aggregate_llm_judge_none_when_unjudged():
    agg = aggregate_artifacts([_art("other", "a", 1.0), _art("window_recall", "b", 1.0)])
    assert agg["llm_judge_n"] == 0
    assert agg["llm_judge_accuracy"] is None
