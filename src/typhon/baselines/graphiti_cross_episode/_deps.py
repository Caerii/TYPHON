"""Optional graphiti-core dependency, imported once for the whole package.

graphiti-core is a heavy optional dependency (a graph backend client + an extraction
LLM). The import is guarded here so the harness can still register, list, and dry-run
the baseline when graphiti-core is absent; every other module in this package imports
these names from here rather than repeating the guard.

When the import fails, the client/recipe names are ``None`` and the error types fall
back to ``Exception``; ``GRAPHITI_AVAILABLE`` is ``False`` and ``IMPORT_ERROR`` holds
the reason. Callers gate real execution on availability (see ``graph._availability``).
"""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import openai

    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.edges import EntityEdge
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.errors import RateLimitError, RefusalError
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.search.search_config_recipes import (
        COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
        COMBINED_HYBRID_SEARCH_MMR,
        COMBINED_HYBRID_SEARCH_RRF,
        EDGE_HYBRID_SEARCH_CROSS_ENCODER,
        EDGE_HYBRID_SEARCH_MMR,
        EDGE_HYBRID_SEARCH_NODE_DISTANCE,
        EDGE_HYBRID_SEARCH_RRF,
        NODE_HYBRID_SEARCH_CROSS_ENCODER,
        NODE_HYBRID_SEARCH_MMR,
        NODE_HYBRID_SEARCH_NODE_DISTANCE,
        NODE_HYBRID_SEARCH_RRF,
    )

    GRAPHITI_AVAILABLE = True
    IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 - any import failure means "unavailable"
    openai = None  # type: ignore[assignment]
    Graphiti = None  # type: ignore[assignment]
    EpisodeType = None  # type: ignore[assignment]
    EntityEdge = None  # type: ignore[assignment]
    LLMConfig = None  # type: ignore[assignment]
    OpenAIGenericClient = None  # type: ignore[assignment]
    OpenAIEmbedder = None  # type: ignore[assignment]
    OpenAIEmbedderConfig = None  # type: ignore[assignment]
    OpenAIRerankerClient = None  # type: ignore[assignment]
    COMBINED_HYBRID_SEARCH_RRF = COMBINED_HYBRID_SEARCH_CROSS_ENCODER = COMBINED_HYBRID_SEARCH_MMR = None  # type: ignore[assignment]
    EDGE_HYBRID_SEARCH_RRF = EDGE_HYBRID_SEARCH_CROSS_ENCODER = EDGE_HYBRID_SEARCH_MMR = EDGE_HYBRID_SEARCH_NODE_DISTANCE = None  # type: ignore[assignment]
    NODE_HYBRID_SEARCH_RRF = NODE_HYBRID_SEARCH_CROSS_ENCODER = NODE_HYBRID_SEARCH_MMR = NODE_HYBRID_SEARCH_NODE_DISTANCE = None  # type: ignore[assignment]
    RateLimitError = RefusalError = Exception  # type: ignore[assignment,misc]
    GRAPHITI_AVAILABLE = False
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


__all__ = [
    "GRAPHITI_AVAILABLE",
    "IMPORT_ERROR",
    "openai",
    "Graphiti",
    "EpisodeType",
    "EntityEdge",
    "LLMConfig",
    "OpenAIGenericClient",
    "OpenAIEmbedder",
    "OpenAIEmbedderConfig",
    "OpenAIRerankerClient",
    "RateLimitError",
    "RefusalError",
    "COMBINED_HYBRID_SEARCH_RRF",
    "COMBINED_HYBRID_SEARCH_CROSS_ENCODER",
    "COMBINED_HYBRID_SEARCH_MMR",
    "EDGE_HYBRID_SEARCH_RRF",
    "EDGE_HYBRID_SEARCH_CROSS_ENCODER",
    "EDGE_HYBRID_SEARCH_MMR",
    "EDGE_HYBRID_SEARCH_NODE_DISTANCE",
    "NODE_HYBRID_SEARCH_RRF",
    "NODE_HYBRID_SEARCH_CROSS_ENCODER",
    "NODE_HYBRID_SEARCH_MMR",
    "NODE_HYBRID_SEARCH_NODE_DISTANCE",
]
