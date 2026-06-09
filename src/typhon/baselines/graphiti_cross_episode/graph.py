"""Graph I/O for the cross-episode baseline: build the client, ingest, search, group.

This is the only module that talks to graphiti/Neo4j. The LLM + embedder are
OpenAI-compatible and fully env-driven, so the same baseline runs against Together.ai
(default), a local LM Studio/Ollama server, or OpenAI without code changes. Graphiti
handles are typed ``Any`` because the concrete classes are optional (see ``_deps``).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from typhon.benchmarks.base import BenchmarkSample

from ._deps import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    COMBINED_HYBRID_SEARCH_MMR,
    COMBINED_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_MMR,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
    EntityEdge,
    EpisodeType,
    Graphiti,
    IMPORT_ERROR,
    LLMConfig,
    NODE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_MMR,
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
    OpenAIEmbedder,
    OpenAIEmbedderConfig,
    OpenAIRerankerClient,
)
from .client import StrictSchemaClient
from .extraction import GUIDED_ENTITY_TYPES, GUIDED_EXTRACTION_INSTRUCTIONS
from .facts import _node_text
from .reranker import ListwiseReranker
from .sessions import _episodes


def _availability(settings: dict[str, Any]) -> tuple[bool, str | None]:
    """Whether a real run can proceed: graphiti installed + graph URI + LLM key present."""
    if Graphiti is None:
        return False, f"graphiti-core not installed ({IMPORT_ERROR})"
    if bool(settings.get("require_graph_uri", True)):
        uri_env = str(settings.get("graph_uri_env", "GRAPH_URI"))
        if not os.environ.get(uri_env):
            return False, f"{uri_env} not set (no graph backend configured)"
    if bool(settings.get("require_llm_key", True)):
        if not (os.environ.get("LLM_API_KEY") or os.environ.get("TOGETHER_API_KEY")):
            return False, "no LLM API key (set TOGETHER_API_KEY or LLM_API_KEY)"
    return True, None


def _build_graphiti(settings: dict[str, Any]) -> Any:
    """Construct Graphiti with an OpenAI-compatible LLM + embedder (Together.ai default).

    Env overrides (so one baseline runs against any OpenAI-compatible backend):
    ``LLM_BASE_URL`` / ``LLM_API_KEY`` (or ``TOGETHER_API_KEY``), ``LLM_MODEL`` /
    ``SMALL_MODEL``, ``EMB_MODEL`` / ``EMB_DIM`` (must match the embedder dimension),
    ``GRAPHITI_MAX_COROUTINES``, and the graph connection via the ``*_env`` setting names.
    """
    assert StrictSchemaClient is not None  # guaranteed by _availability()
    base_url = os.environ.get(
        "LLM_BASE_URL", str(settings.get("llm_base_url", "https://api.together.xyz/v1"))
    )
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("TOGETHER_API_KEY", "")
    model = os.environ.get(
        "LLM_MODEL", str(settings.get("llm_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo"))
    )
    small_model = os.environ.get("SMALL_MODEL", str(settings.get("small_model", model)))
    emb_model = os.environ.get(
        "EMB_MODEL",
        str(settings.get("embedding_model", "intfloat/multilingual-e5-large-instruct")),
    )
    emb_dim = int(os.environ.get("EMB_DIM", str(settings.get("embedding_dim", 1024))))
    max_coroutines = int(
        os.environ.get("GRAPHITI_MAX_COROUTINES", str(settings.get("max_coroutines", 1)))
    )

    llm_config = LLMConfig(
        api_key=api_key, base_url=base_url, model=model, small_model=small_model, temperature=0.0
    )

    # Reranker selection. The cross_encoder recipe invokes ``cross_encoder.rank`` to reorder
    # candidates; graphiti's stock OpenAIRerankerClient needs OpenAI-only token logprobs that
    # Together/local backends don't return (it crashes there — see reranker.py). When the
    # variant actually uses the reranker, swap in the backend-agnostic ListwiseReranker; the
    # RRF/MMR recipes never call rank(), so the stock client is fine (and cheaper to build).
    search_variant = str(settings.get("search_variant", "rrf"))
    if search_variant == "cross_encoder" and ListwiseReranker is not None:
        cross_encoder = ListwiseReranker(config=llm_config)
    else:
        cross_encoder = OpenAIRerankerClient(config=llm_config)

    return Graphiti(
        os.environ.get(str(settings.get("graph_uri_env", "GRAPH_URI")), "bolt://localhost:7687"),
        os.environ.get(str(settings.get("graph_user_env", "GRAPH_USER")), "neo4j"),
        os.environ.get(str(settings.get("graph_password_env", "GRAPH_PASSWORD")), "password"),
        llm_client=StrictSchemaClient(config=llm_config),
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=api_key,
                base_url=base_url,
                embedding_model=emb_model,
                embedding_dim=emb_dim,
            )
        ),
        cross_encoder=cross_encoder,
        max_coroutines=max_coroutines,
    )


async def _ingest_episodes(
    graphiti: Any, sample: BenchmarkSample, settings: dict[str, Any], group_id: str
) -> None:
    """Ingest a sample's sessions as ordered episodes (increasing valid-time) into group_id."""
    base_time = datetime.now(timezone.utc)
    guided = bool(settings.get("guided_extraction", True))
    entity_types = GUIDED_ENTITY_TYPES if guided else None
    extraction_instructions = GUIDED_EXTRACTION_INSTRUCTIONS if guided else None
    for index, (name, body) in enumerate(_episodes(sample, settings)):
        await graphiti.add_episode(
            name=name,
            episode_body=body,
            source=EpisodeType.text,
            source_description=f"cross-episode · {sample.sample_id}",
            reference_time=base_time + timedelta(minutes=index),
            group_id=group_id,
            entity_types=entity_types,
            custom_extraction_instructions=extraction_instructions,
        )


def _resolve_search_recipe(search_mode: str, search_variant: str) -> Any:
    """Resolve the search recipe based on mode (edges vs combined) and variant (RRF, cross-encoder, etc).

    Variants (search_variant):
    - "rrf" (default): reciprocal rank fusion, fast baseline
    - "cross_encoder": cross-encoder reranker + BFS multi-hop (Tier 2 Phase 1)
    - "mmr": maximum marginal relevance
    - "node_distance": BFS node_distance reranker (Tier 2 Phase 2, edges-only)

    search_mode:
    - "combined" (default): edges + nodes
    - "edges": edges only
    - "nodes": nodes only
    """
    if search_mode == "edges":
        if search_variant == "cross_encoder":
            return EDGE_HYBRID_SEARCH_CROSS_ENCODER
        elif search_variant == "mmr":
            return EDGE_HYBRID_SEARCH_MMR
        elif search_variant == "node_distance":
            return EDGE_HYBRID_SEARCH_NODE_DISTANCE
        else:  # rrf or default
            return EDGE_HYBRID_SEARCH_RRF
    elif search_mode == "nodes":
        if search_variant == "cross_encoder":
            return NODE_HYBRID_SEARCH_CROSS_ENCODER
        elif search_variant == "mmr":
            return NODE_HYBRID_SEARCH_MMR
        elif search_variant == "node_distance":
            return NODE_HYBRID_SEARCH_NODE_DISTANCE
        else:  # rrf or default
            return NODE_HYBRID_SEARCH_RRF
    else:  # combined or default
        if search_variant == "cross_encoder":
            return COMBINED_HYBRID_SEARCH_CROSS_ENCODER
        elif search_variant == "mmr":
            return COMBINED_HYBRID_SEARCH_MMR
        else:  # rrf or default
            return COMBINED_HYBRID_SEARCH_RRF


async def _search_facts(
    graphiti: Any, question: str, settings: dict[str, Any], group_id: str, num_results: int
) -> list[dict[str, Any]]:
    """Retrieve facts for one question from group_id.

    Combined edge + node retrieval: edges carry entity-entity facts with bi-temporal
    validity; nodes carry the name/summary/attributes where a scalar or attribute fact
    (e.g. a rate limit) lives when it never became an edge. ``search_mode="edges"`` falls
    back to edge-only; node text follows ``node_text_mode`` (see facts._node_text).
    Episodes/communities are ignored so this stays graph retrieval, not raw-text RAG.
    """
    node_text_mode = str(settings.get("node_text_mode", "summary"))
    search_mode = str(settings.get("search_mode", "combined"))
    search_variant = str(settings.get("search_variant", "rrf"))
    recipe = _resolve_search_recipe(search_mode, search_variant)
    config = recipe.model_copy(update={"limit": num_results})
    results = await graphiti.search_(question, config=config, group_ids=[group_id])
    edges = list(results.edges)
    nodes = list(getattr(results, "nodes", []))

    # "bitemporal" and "attributes" both need each node's FULL incident edge set (not just
    # the top-K) so currency is judged over all of a node's relations: keep only current
    # edges (a value like "40 rpm" surfaces while the superseded "100" is excluded) and
    # know whether a node is stale-only (has edges, none current) so it can be dropped.
    # Runs before the driver closes.
    node_current_edges: dict[str, list[str]] = {}
    node_incident_count: dict[str, int] = {}
    if node_text_mode in ("bitemporal", "attributes"):
        for node in nodes:
            uuid = getattr(node, "uuid", None)
            if not uuid:
                continue
            incident = await EntityEdge.get_by_node_uuid(graphiti.driver, uuid)
            node_incident_count[uuid] = len(incident)
            node_current_edges[uuid] = [
                edge.fact
                for edge in incident
                if getattr(edge, "invalid_at", None) is None
                and getattr(edge, "expired_at", None) is None
            ]

    facts: list[dict[str, Any]] = []
    for edge in edges:
        valid_at = getattr(edge, "valid_at", None)
        invalid_at = getattr(edge, "invalid_at", None)
        facts.append(
            {
                "fact": edge.fact,
                "kind": "edge",
                "uuid": getattr(edge, "uuid", None),
                "valid_at": valid_at.isoformat() if valid_at else None,
                "invalid_at": invalid_at.isoformat() if invalid_at else None,
                "current": invalid_at is None,
            }
        )
    incident_current: dict[str, list[str]] = {}
    if node_text_mode == "current_edges":
        for edge in edges:
            if getattr(edge, "invalid_at", None) is not None:
                continue  # current edges only
            for endpoint in (getattr(edge, "source_node_uuid", None), getattr(edge, "target_node_uuid", None)):
                if endpoint:
                    incident_current.setdefault(endpoint, []).append(edge.fact)
    for node in nodes:
        uuid = getattr(node, "uuid", None)
        if node_text_mode in ("bitemporal", "attributes"):
            node_facts = node_current_edges.get(uuid, [])
            incident_count = node_incident_count.get(uuid, 0)
        else:
            node_facts = incident_current.get(uuid, [])
            incident_count = len(node_facts)
        text = _node_text(
            getattr(node, "name", ""),
            node_text_mode,
            summary=getattr(node, "summary", ""),
            attributes=getattr(node, "attributes", {}),
            current_facts=node_facts,
            incident_count=incident_count,
        )
        if not text:
            continue
        facts.append(
            {
                "fact": text,
                "kind": "node",
                "uuid": uuid,
                "valid_at": None,
                "invalid_at": None,
                "current": True,
            }
        )
    return facts


async def _run_one_graph(
    samples: list[BenchmarkSample], settings: dict[str, Any], group_id: str, num_results: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Build one graph, ingest the (shared) context once, answer every sample against it.

    Per-sample mode passes a single sample; shared-graph mode passes all the QA of one
    conversation (which share the same context), so the conversation is ingested once and
    every question is answered against that one graph — instead of re-ingesting per QA.

    Returns ``(facts_by_sample, usage)`` where ``usage`` is this graph's accumulated
    extraction-LLM token usage (the cost behind the recall). Usage is per-*group*: in
    shared-graph mode it is the one-time ingestion cost amortized across the group's QA.
    """
    assert Graphiti is not None  # guaranteed by _availability()
    graphiti = _build_graphiti(settings)
    out: dict[str, list[dict[str, Any]]] = {}
    usage: dict[str, int] = {}
    try:
        await graphiti.build_indices_and_constraints()
        await _ingest_episodes(graphiti, samples[0], settings, group_id)
        for sample in samples:
            out[sample.sample_id] = await _search_facts(
                graphiti, sample.question, settings, group_id, num_results
            )
        usage = dict(getattr(graphiti.llm_client, "usage", {}) or {})
        # Fold the reranker's own LLM calls into the reported total so the cost block
        # prices the rerank, not just extraction (the ListwiseReranker tracks usage; the
        # stock OpenAIRerankerClient does not, contributing nothing).
        rerank_usage = getattr(graphiti.cross_encoder, "usage", None)
        if isinstance(rerank_usage, dict):
            for key in ("llm_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] = int(usage.get(key, 0)) + int(rerank_usage.get(key, 0))
    finally:
        await graphiti.close()
    return out, usage


def _group_samples(
    samples: list[BenchmarkSample], shared_key: str | None
) -> list[tuple[str, list[BenchmarkSample]]]:
    """Group samples for graph reuse, preserving order.

    With a ``shared_key`` (e.g. "conversation"), samples sharing that metadata value reuse
    one graph (ingested once). Samples lacking the key — and the default ``shared_key=None``
    — get their own per-sample graph, which reproduces the original behavior exactly.
    """
    groups: dict[str, list[BenchmarkSample]] = {}
    order: list[str] = []
    for sample in samples:
        key = None
        if shared_key:
            value = (sample.metadata or {}).get(shared_key)
            key = str(value) if value is not None else None
        if not key:
            key = sample.sample_id
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(sample)
    return [(key, groups[key]) for key in order]
