"""Unit tests for the listwise reranker's pure logic + the search-recipe resolver.

These exercise the scoring/ordering helpers and the variant→recipe mapping without a
graph backend or an LLM, so they run in CI with zero external deps. The async client
(``ListwiseReranker.rank``) is covered live in test_graphiti_live.py when a key is set.
"""

from __future__ import annotations

from typhon.baselines.graphiti_cross_episode.reranker import (
    _apply_scores,
    _build_messages,
    _fallback_ranking,
)


# --- _fallback_ranking: failure degrades to input order -----------------------


def test_fallback_preserves_input_order():
    passages = ["a", "b", "c"]
    ranked = _fallback_ranking(passages)
    assert [text for text, _ in ranked] == ["a", "b", "c"]


def test_fallback_scores_descend():
    ranked = _fallback_ranking(["a", "b", "c"])
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0


def test_fallback_empty():
    assert _fallback_ranking([]) == []


# --- _apply_scores: model scores → sorted ranking -----------------------------


def test_apply_scores_sorts_by_relevance_desc():
    passages = ["alpha", "beta", "gamma"]
    # model says gamma most relevant, alpha least
    ranked = _apply_scores(passages, [(0, 0.1), (1, 0.5), (2, 0.9)])
    assert [text for text, _ in ranked] == ["gamma", "beta", "alpha"]
    assert [round(s, 2) for _, s in ranked] == [0.9, 0.5, 0.1]


def test_apply_scores_missing_index_scores_zero():
    passages = ["alpha", "beta", "gamma"]
    # model only scored index 1; the others default to 0.0 and sink
    ranked = _apply_scores(passages, [(1, 0.8)])
    assert ranked[0] == ("beta", 0.8)
    assert {text for text, _ in ranked[1:]} == {"alpha", "gamma"}
    assert all(score == 0.0 for _, score in ranked[1:])


def test_apply_scores_clamps_out_of_range_values():
    passages = ["a", "b"]
    ranked = _apply_scores(passages, [(0, 5.0), (1, -3.0)])
    assert dict(ranked) == {"a": 1.0, "b": 0.0}


def test_apply_scores_ignores_out_of_range_index():
    passages = ["a", "b"]
    # index 7 doesn't exist; must not raise, must not appear
    ranked = _apply_scores(passages, [(0, 0.4), (7, 0.99)])
    assert {text for text, _ in ranked} == {"a", "b"}
    assert dict(ranked)["a"] == 0.4


def test_apply_scores_ties_keep_input_order():
    passages = ["first", "second", "third"]
    ranked = _apply_scores(passages, [(0, 0.5), (1, 0.5), (2, 0.5)])
    # stable sort: equal scores preserve original order
    assert [text for text, _ in ranked] == ["first", "second", "third"]


# --- _build_messages: prompt shape --------------------------------------------


def test_build_messages_numbers_passages():
    msgs = _build_messages("where is X?", ["fact one", "fact two"])
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    user = msgs[1]["content"]
    assert "where is X?" in user
    assert "[0] fact one" in user
    assert "[1] fact two" in user


def test_build_messages_system_warns_against_word_overlap():
    msgs = _build_messages("q", ["p"])
    # the system prompt must steer away from lexical-overlap false positives
    assert "shares a word" in msgs[0]["content"]
