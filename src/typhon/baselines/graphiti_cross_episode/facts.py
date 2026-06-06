"""Shape and temporally order retrieved graph results into prediction-ready facts (pure).

``_node_text`` renders a retrieved node into one retrieval string under the chosen
``node_text_mode``; ``_order_facts`` applies the long-term tier's temporal preference so
*current* facts answer "what is true now". Both are pure — no graph/LLM — so they are
unit-testable in isolation (see tests/test_graphiti_cross_episode.py).
"""

from __future__ import annotations

from typing import Any


def _node_text(
    name: str,
    mode: str,
    *,
    summary: str = "",
    attributes: dict[str, Any] | None = None,
    current_facts: list[str] | None = None,
    incident_count: int = 0,
) -> str | None:
    """Build a retrieved node's text for ``node_text_mode`` — or None to drop the node.

    - "summary" (default): name + regional summary + attributes (recall-max; the summary
      aggregates history so it can carry a since-superseded value).
    - "current_edges" / "bitemporal": name + current incident edge facts, dropping the
      history-aggregating summary. In "bitemporal", a *stale-only* node — one that has
      incident edges but none current (e.g. a replaced 'Postgres') — is dropped.
    """
    name = (name or "").strip()
    if mode in ("current_edges", "bitemporal"):
        node_facts = current_facts or []
        if mode == "bitemporal" and not node_facts and incident_count > 0:
            return None  # stale-only node
        return (f"{name}: {' '.join(node_facts)}".strip() if node_facts else name) or None
    summary = (summary or "").strip()
    attributes = attributes or {}
    attr_text = "; ".join(
        f"{key}: {value}" for key, value in attributes.items() if value not in (None, "", [], {})
    )
    text = f"{name} — {summary}" if (name and summary) else (summary or name)
    if attr_text:
        text = f"{text} ({attr_text})" if text else attr_text
    return text.strip() or None


def _order_facts(facts: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply the long-term tier's temporal preference to retrieved facts.

    ``current_facts_only`` drops superseded edges entirely (falling back to all facts if
    nothing is current); otherwise ``prefer_current`` orders current facts first. Edges
    carry bi-temporal validity, so a *current edge* is authoritative for "what is true
    now"; node text supplements recall but a node's regional summary can aggregate history,
    so it ranks AFTER current edges.
    """
    if not facts:
        return facts
    current_edges = [f for f in facts if f.get("current") and f.get("kind") != "node"]
    current_nodes = [f for f in facts if f.get("current") and f.get("kind") == "node"]
    superseded = [f for f in facts if not f.get("current")]
    if bool(settings.get("current_facts_only", False)):
        ordered = current_edges + current_nodes
        return ordered or facts
    if bool(settings.get("prefer_current", True)):
        return current_edges + current_nodes + superseded
    return facts
