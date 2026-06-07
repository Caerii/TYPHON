"""Shape and temporally order retrieved graph results into prediction-ready facts (pure).

``_node_text`` renders a retrieved node into one retrieval string under the chosen
``node_text_mode``; ``_order_facts`` applies the long-term tier's temporal preference so
*current* facts answer "what is true now". Both are pure — no graph/LLM — so they are
unit-testable in isolation (see tests/test_graphiti_cross_episode.py).
"""

from __future__ import annotations

from typing import Any


def _attr_text(attributes: dict[str, Any] | None) -> str:
    """Render non-empty node attributes as ``key: value; key: value`` (empty -> "").

    Attributes are structured current values (e.g. ``current_value: 40 requests per
    minute``) that graphiti updates to current on supersession, so they are the
    bi-temporally-clean alternative to the history-aggregating prose summary.
    """
    return "; ".join(
        f"{key}: {value}"
        for key, value in (attributes or {}).items()
        if value not in (None, "", [], {})
    )


def _summary_text(name: str, summary: str) -> str:
    """Join an entity name and its prose summary as ``name — summary`` (either may be empty)."""
    summary = (summary or "").strip()
    return f"{name} — {summary}" if (name and summary) else (summary or name)


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

    - "summary" (default): name + prose summary + attributes (recall-max; the summary
      aggregates history so it can carry a since-superseded value).
    - "attributes" (best-of-both): a layered preference that never returns the
      history-aggregating summary when a currency signal exists. In order:
        1. structured current attributes (e.g. ``current_value``) -> ``name (attrs)``;
        2. else the node's CURRENT incident edges (bi-temporally clean) -> ``name: facts``;
        3. else, if the node has incident edges but none current, it is *stale-only*
           (e.g. a replaced 'Postgres') and is dropped;
        4. else (an edgeless node with no currency signal at all) fall back to the prose
           summary, where it can still aid recall and nothing contradicts it.
      So changeable things surface their clean current value, entities with edges surface
      their current relations, superseded entities vanish, and only un-structured entities
      lean on the summary.
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
    attr_text = _attr_text(attributes)
    if mode == "attributes":
        if attr_text:
            return f"{name} ({attr_text})".strip() or None
        node_facts = current_facts or []
        if node_facts:
            return f"{name}: {' '.join(node_facts)}".strip() or None
        if incident_count > 0:
            return None  # stale-only node (has edges, none current) — drop, don't leak summary
        return _summary_text(name, summary).strip() or None  # edgeless: summary aids recall
    # "summary" (default): name + summary, then append attributes if any.
    text = _summary_text(name, summary)
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
