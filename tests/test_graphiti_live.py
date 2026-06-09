"""Opt-in *live* integration test for the graphiti_cross_episode baseline.

Unlike ``test_graphiti_cross_episode.py`` (pure / dry-run, zero deps), this exercises the
real ``add_episode`` pipeline against a graph backend + an extraction LLM. It is the
committed regression for **Lever 1**: a ``Configurable`` entity's typed ``current_value``
must update to the *current* value on supersession (100 -> 40), so the bi-temporally-clean
attribute — not the history-aggregating prose summary — carries the answer.

Skipped automatically unless a backend and key are configured, so CI and no-backend
environments stay green. Run it with::

    TOGETHER_API_KEY=... GRAPH_URI=bolt://localhost:7687 GRAPH_USER=neo4j \
        GRAPH_PASSWORD=... python -m pytest tests/test_graphiti_live.py -q
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from typhon.baselines import graphiti_cross_episode as g

_LIVE = bool(
    g.GRAPHITI_AVAILABLE
    and os.environ.get("GRAPH_URI")
    and (os.environ.get("TOGETHER_API_KEY") or os.environ.get("LLM_API_KEY"))
)
pytestmark = pytest.mark.skipif(
    not _LIVE, reason="live graphiti backend + LLM key required (set GRAPH_URI + TOGETHER_API_KEY)"
)

SUPERSESSION = [
    "We set the public API rate limit to 100 requests per minute.",
    "After the outage we lowered the public API rate limit to 40 requests per minute.",
]


async def _ingest_and_read(group_id: str) -> list[dict]:
    """Ingest the supersession episodes into ``group_id`` and read back the entity nodes."""
    from graphiti_core.nodes import EpisodeType  # local: only importable when GRAPHITI_AVAILABLE

    graphiti = g._build_graphiti({})
    base = datetime.now(timezone.utc)
    try:
        await graphiti.build_indices_and_constraints()
        for index, body in enumerate(SUPERSESSION):
            await graphiti.add_episode(
                name=f"e{index}",
                episode_body=body,
                source=EpisodeType.text,
                source_description="lever1-live-test",
                reference_time=base + timedelta(minutes=index),
                group_id=group_id,
                entity_types=g.GUIDED_ENTITY_TYPES,
                custom_extraction_instructions=g.GUIDED_EXTRACTION_INSTRUCTIONS,
            )
        records, _, _ = await graphiti.driver.execute_query(
            "MATCH (n:Entity {group_id: $gid}) "
            "RETURN n.name AS name, labels(n) AS labels, properties(n) AS props",
            gid=group_id,
        )
        return [
            {"name": r["name"], "labels": list(r["labels"]), "props": dict(r["props"])}
            for r in records
        ]
    finally:
        # Clean up this test's group so repeated runs never accumulate / contaminate.
        await graphiti.driver.execute_query(
            "MATCH (n {group_id: $gid}) DETACH DELETE n", gid=group_id
        )
        await graphiti.close()


def test_configurable_current_value_supersedes_to_current():
    """End-to-end: the changeable thing is classified Configurable and its current_value
    reflects the LATEST value (40), not the superseded one (100)."""
    nodes = asyncio.run(_ingest_and_read(f"live_lever1_{uuid.uuid4().hex[:8]}"))

    configurables = [n for n in nodes if "Configurable" in n["labels"]]
    assert configurables, f"no Configurable entity was extracted; nodes={[n['name'] for n in nodes]}"

    current_values = [str(n["props"].get("current_value", "")) for n in configurables]
    blob = " | ".join(current_values)
    assert "40" in blob, f"current_value did not capture the current value (40); got {blob!r}"
    assert "100" not in blob, f"current_value leaked the superseded value (100); got {blob!r}"
