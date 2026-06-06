"""Unit tests for the graphiti_cross_episode baseline's pure logic.

These exercise session splitting, fact ordering, availability gating, the strict
schema client wiring, and the dry-run artifact shape — all without a graph backend
or an LLM (dry_run / pure functions), so they run in CI with zero external deps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from typhon.baselines import graphiti_cross_episode as g
from typhon.baselines.local_exact import run_baseline
from typhon.baselines.registry import BaselineRegistry
from typhon.benchmarks.registry import BenchmarkRegistry


# --- session splitting --------------------------------------------------------

def test_split_sessions_session_markers_strip_prefix():
    ctx = "Session 1: Alpha beta. Session 2: Gamma delta."
    assert g._split_sessions(ctx, {}) == ["Alpha beta.", "Gamma delta."]


def test_split_sessions_turn_markers():
    ctx = "Turn 1: First thing. Turn 2: Second thing."
    assert g._split_sessions(ctx, {}) == ["First thing.", "Second thing."]


def test_split_sessions_double_newline_delimiter():
    ctx = "Para one stands alone.\n\nPara two is separate."
    assert g._split_sessions(ctx, {}) == ["Para one stands alone.", "Para two is separate."]


def test_split_sessions_custom_regex_setting():
    ctx = "a||b||c"
    assert g._split_sessions(ctx, {"session_regex": r"\|\|"}) == ["a", "b", "c"]


def test_split_sessions_single_block_no_markers():
    ctx = "Just one sentence with no markers at all."
    assert g._split_sessions(ctx, {}) == [ctx]


def test_episodes_are_indexed_by_sample_id():
    class _S:
        sample_id = "smp"
        context = "Session 1: one. Session 2: two."

    episodes = g._episodes(_S(), {})
    assert [name for name, _ in episodes] == ["smp-s000", "smp-s001"]
    assert [body for _, body in episodes] == ["one.", "two."]


# --- fact ordering (temporal preference + edge/node priority) ------------------

def _facts():
    return [
        {"fact": "edge_sup", "kind": "edge", "current": False},
        {"fact": "node_cur", "kind": "node", "current": True},
        {"fact": "edge_cur", "kind": "edge", "current": True},
    ]


def test_order_facts_current_only_drops_superseded_edges_first():
    ordered = g._order_facts(_facts(), {"current_facts_only": True})
    assert [f["fact"] for f in ordered] == ["edge_cur", "node_cur"]


def test_order_facts_prefer_current_orders_edges_nodes_superseded():
    ordered = g._order_facts(_facts(), {"current_facts_only": False, "prefer_current": True})
    assert [f["fact"] for f in ordered] == ["edge_cur", "node_cur", "edge_sup"]


def test_order_facts_current_only_falls_back_when_nothing_current():
    facts = [{"fact": "only_sup", "kind": "edge", "current": False}]
    assert g._order_facts(facts, {"current_facts_only": True}) == facts


def test_order_facts_empty():
    assert g._order_facts([], {"current_facts_only": True}) == []


# --- availability gating ------------------------------------------------------

def test_availability_requires_graph_uri(monkeypatch):
    monkeypatch.delenv("GRAPH_URI", raising=False)
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    ok, reason = g._availability({})
    assert ok is False and "GRAPH_URI" in (reason or "")


def test_availability_requires_llm_key(monkeypatch):
    monkeypatch.setenv("GRAPH_URI", "bolt://localhost:7687")
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    ok, reason = g._availability({})
    assert ok is False and "API key" in (reason or "")


def test_availability_ok_when_both_present(monkeypatch):
    monkeypatch.setenv("GRAPH_URI", "bolt://localhost:7687")
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    assert g._availability({}) == (True, None)


# --- strict schema client + guided extraction wiring --------------------------

def test_strict_schema_client_is_defined_and_subclasses_generic():
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    assert g.StrictSchemaClient is not None
    assert issubclass(g.StrictSchemaClient, OpenAIGenericClient)
    assert "_generate_response" in vars(g.StrictSchemaClient)


def test_guided_entity_types_present():
    assert set(g.GUIDED_ENTITY_TYPES) == {"Value", "Role", "System"}
    assert g.GUIDED_EXTRACTION_INSTRUCTIONS.strip()


# --- dry-run artifact shape (no graph backend, no LLM) ------------------------

def test_dry_run_emits_not_executed_artifact(tmp_path: Path):
    baselines = BaselineRegistry.load()
    benchmarks = BenchmarkRegistry.load()
    artifacts = run_baseline(
        baseline_registry=baselines,
        benchmark_registry=benchmarks,
        baseline_id="graphiti_cross_episode",
        benchmark_id="locomo_window",
        family=None,
        output_dir=tmp_path,
        dry_run=True,
        sample_source="fixture",
        sample_limit=None,
        chunk_size_override=None,
        local_window_tokens_override=None,
    )
    assert len(artifacts) == 1
    art = artifacts[0]
    assert art["status"] == "not_executed"
    for key in ("baseline", "benchmark", "memory_state", "prediction", "limitations"):
        assert key in art
    ce = art["memory_state"]["cross_episode"]
    assert ce["episode_count"] >= 1
    assert ce["retrieved_fact_count"] == 0


@pytest.mark.parametrize("baseline_id", ["graphiti_cross_episode"])
def test_baseline_is_registered(baseline_id: str):
    assert any(b.id == baseline_id for b in BaselineRegistry.load().list_baselines())


# --- node_text_mode (summary | current_edges | bitemporal) --------------------

def test_node_text_summary_includes_summary_and_attrs():
    text = g._node_text("API", "summary", summary="the public API", attributes={"rate_limit": "40 rpm"})
    assert "API" in text and "the public API" in text and "rate_limit: 40 rpm" in text


def test_node_text_current_edges_uses_current_facts():
    assert (
        g._node_text("public API", "current_edges", current_facts=["rate limit is 40 rpm"])
        == "public API: rate limit is 40 rpm"
    )


def test_node_text_current_edges_falls_back_to_name():
    assert g._node_text("Pixel", "current_edges", current_facts=[]) == "Pixel"


def test_node_text_bitemporal_name_only_when_no_incident():
    # Entity-name answers (Pixel/Northwind) resolve from the name alone.
    assert g._node_text("Pixel", "bitemporal", current_facts=[], incident_count=0) == "Pixel"


def test_node_text_bitemporal_drops_stale_only_node():
    # A node with incident edges but none current (e.g. a replaced 'Postgres') is dropped.
    assert g._node_text("Postgres", "bitemporal", current_facts=[], incident_count=2) is None


def test_node_text_bitemporal_surfaces_current_value():
    assert (
        g._node_text("public API", "bitemporal", current_facts=["rate limit is 40"], incident_count=3)
        == "public API: rate limit is 40"
    )


# --- shared-graph grouping ----------------------------------------------------

class _Sample:
    def __init__(self, sample_id, metadata=None):
        self.sample_id = sample_id
        self.metadata = metadata or {}


def test_group_samples_default_is_per_sample():
    samples = [_Sample("a"), _Sample("b")]
    groups = g._group_samples(samples, None)
    assert [k for k, _ in groups] == ["a", "b"]
    assert all(len(v) == 1 for _, v in groups)


def test_group_samples_shared_key_groups_by_conversation():
    samples = [
        _Sample("a", {"conversation": "c1"}),
        _Sample("b", {"conversation": "c1"}),
        _Sample("c", {"conversation": "c2"}),
    ]
    groups = dict((k, [s.sample_id for s in v]) for k, v in g._group_samples(samples, "conversation"))
    assert groups == {"c1": ["a", "b"], "c2": ["c"]}


def test_group_samples_missing_key_falls_back_to_sample_id():
    samples = [_Sample("a", {"conversation": "c1"}), _Sample("b", {})]
    groups = dict((k, [s.sample_id for s in v]) for k, v in g._group_samples(samples, "conversation"))
    assert groups == {"c1": ["a"], "b": ["b"]}


# --- session chunking (Lever 2) -----------------------------------------------

def test_chunk_text_disabled_or_short_returns_single():
    assert g._chunk_text("a. b. c.", 0) == ["a. b. c."]
    assert g._chunk_text("short", 100) == ["short"]
    assert g._chunk_text("   ", 10) == []


def test_chunk_text_packs_sentences_under_budget_preserving_words():
    text = "Alpha alpha. Beta beta. Gamma gamma."
    chunks = g._chunk_text(text, 14)
    assert len(chunks) >= 2
    assert all(len(c) <= 14 for c in chunks)
    assert set(" ".join(chunks).split()) == set(text.split())  # no content lost


def test_chunk_text_hard_splits_oversized_sentence():
    chunks = g._chunk_text("x" * 50, 20)
    assert all(len(c) <= 20 for c in chunks)
    assert "".join(chunks) == "x" * 50


class _LongSample:
    sample_id = "smp"
    context = "Session 1: " + "Aaaa bbbb cccc. " * 20  # one long session


def test_episodes_chunks_long_session_with_budget():
    eps = g._episodes(_LongSample(), {"max_episode_chars": 60})
    assert len(eps) > 1
    assert all(name.startswith("smp-s000-c") for name, _ in eps)
    assert all(len(body) <= 60 for _, body in eps)


def test_episodes_unchanged_when_chunking_disabled():
    class _S:
        sample_id = "smp"
        context = "Session 1: one. Session 2: two."

    assert [name for name, _ in g._episodes(_S(), {})] == ["smp-s000", "smp-s001"]
