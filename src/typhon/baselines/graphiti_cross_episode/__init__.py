"""Graphiti cross-episode memory baseline — the long-term tier of the memory stack.

Each sample's context is split into ordered sessions and ingested as episodes into a
bi-temporal knowledge graph (Graphiti); the question is answered from facts retrieved
*across* episodes. The prediction path is identical to ``attention_baseline``, so a
head-to-head isolates the one variable that matters: retrieval from a temporal graph vs.
retrieval from a local window. graphiti-core is an optional dependency; when it (or a
backend) is absent the runner emits a ``not_executed`` artifact instead of failing.

The implementation is decomposed into focused modules:

- ``_deps``      — the single guarded graphiti-core import (+ ``GRAPHITI_AVAILABLE``)
- ``client``     — ``StrictSchemaClient`` (strict structured outputs over Together)
- ``extraction`` — guided entity-type hints that let value/role/system edges resolve
- ``sessions``   — context → ordered episodes (pure)
- ``facts``      — node-text rendering + temporal ordering (pure)
- ``graph``      — build / ingest / search / group (the only module that touches Neo4j)
- ``runner``     — orchestration + artifact assembly (the public entry point)

This ``__init__`` is a thin facade so ``from typhon.baselines.graphiti_cross_episode
import run_graphiti_cross_episode_baseline`` and the test-facing helpers keep working.
"""

from __future__ import annotations

from ._deps import GRAPHITI_AVAILABLE
from .client import StrictSchemaClient
from .extraction import GUIDED_ENTITY_TYPES, GUIDED_EXTRACTION_INSTRUCTIONS
from .facts import _node_text, _order_facts
from .graph import _availability, _build_graphiti, _group_samples
from .runner import run_graphiti_cross_episode_baseline
from .sessions import _chunk_text, _episodes, _split_sessions

__all__ = [
    "run_graphiti_cross_episode_baseline",
    "GRAPHITI_AVAILABLE",
    "StrictSchemaClient",
    "GUIDED_ENTITY_TYPES",
    "GUIDED_EXTRACTION_INSTRUCTIONS",
    # Re-exported for tests / introspection (the package's internal helpers).
    "_split_sessions",
    "_episodes",
    "_chunk_text",
    "_node_text",
    "_order_facts",
    "_build_graphiti",
    "_group_samples",
    "_availability",
]
