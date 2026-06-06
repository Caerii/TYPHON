"""Graphiti cross-episode memory baseline.

The long-term tier of the memory stack, made measurable: each sample's context is
split into ordered sessions and ingested as episodes into a bi-temporal knowledge
graph (Graphiti), then the question is answered from facts retrieved *across*
episodes. The prediction path is identical to ``attention_baseline`` (extractive,
via ``build_prediction_block``), so a head-to-head isolates the one variable that
matters here — retrieval from a temporal graph vs. retrieval from a local window.

Graphiti is an optional, heavy dependency (a graph backend + an extraction LLM).
The import is guarded so the harness can still register and list this baseline;
when graphiti-core or a graph backend is absent, the runner emits a
``not_executed`` artifact in the normal shape instead of failing.

The LLM + embedder are OpenAI-compatible and fully env-driven, so the *same*
baseline runs against Together.ai (default), a local LM Studio/Ollama server, or
OpenAI without code changes. See ``_build_graphiti`` for the knobs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from typhon.baselines.base import BaselineSpec
from typhon.benchmarks.base import BenchmarkSample, BenchmarkSpec
from typhon.eval.heuristics import build_prediction_block
from typhon.experiments.budget import BudgetLedger
from typhon.runtime.base import RuntimeProfile

try:  # pragma: no cover - optional dependency
    import openai

    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.errors import RateLimitError, RefusalError
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.search.search_config_recipes import (
        COMBINED_HYBRID_SEARCH_RRF,
        EDGE_HYBRID_SEARCH_RRF,
    )

    _GRAPHITI_IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 - any import failure means "unavailable"
    openai = None  # type: ignore[assignment]
    Graphiti = None  # type: ignore[assignment]
    EpisodeType = None  # type: ignore[assignment]
    LLMConfig = None  # type: ignore[assignment]
    OpenAIGenericClient = None  # type: ignore[assignment]
    OpenAIEmbedder = None  # type: ignore[assignment]
    OpenAIEmbedderConfig = None  # type: ignore[assignment]
    OpenAIRerankerClient = None  # type: ignore[assignment]
    COMBINED_HYBRID_SEARCH_RRF = EDGE_HYBRID_SEARCH_RRF = None  # type: ignore[assignment]
    RateLimitError = RefusalError = Exception  # type: ignore[assignment,misc]
    _GRAPHITI_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


logger = logging.getLogger(__name__)


if OpenAIGenericClient is not None:

    class StrictSchemaClient(OpenAIGenericClient):  # type: ignore[misc,valid-type]
        """OpenAIGenericClient that ENFORCES the response schema.

        The stock generic client sends a ``json_schema`` *without* ``strict: true``,
        so an OpenAI-compatible backend treats it as best-effort: the model may omit
        fields, which is why value/attribute edges (e.g. "rate limit -> 40 requests
        per minute") were silently dropped, leaving some episodes with entities but
        no facts. This subclass routes schema'd calls through the OpenAI SDK's
        structured-output ``beta.chat.completions.parse`` path, which Together.ai
        honors as strict constrained decoding, and validates against the Pydantic
        model. Calls with no ``response_model`` fall back to plain ``json_object``.
        """

        async def _generate_response(  # type: ignore[override]
            self,
            messages,
            response_model=None,
            max_tokens=None,
            model_size=None,
        ) -> dict[str, Any]:
            openai_messages = []
            for message in messages:
                message.content = self._clean_input(message.content)
                if message.role in ("user", "system"):
                    openai_messages.append({"role": message.role, "content": message.content})
            try:
                if response_model is not None:
                    completion = await self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=openai_messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format=response_model,
                    )
                    parsed_message = completion.choices[0].message
                    refusal = getattr(parsed_message, "refusal", None)
                    if refusal:
                        raise RefusalError(refusal)
                    parsed = getattr(parsed_message, "parsed", None)
                    if parsed is not None:
                        return parsed.model_dump()
                    return json.loads(parsed_message.content or "{}")
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content or "{}")
            except openai.RateLimitError as exc:  # type: ignore[union-attr]
                raise RateLimitError from exc
            except RefusalError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("StrictSchemaClient response error: %s", exc)
                raise

else:  # graphiti-core unavailable

    StrictSchemaClient = None  # type: ignore[assignment]


# --- Guided extraction (the dropped-edge fix) ----------------------------------
# Edges are dropped when an endpoint NAME isn't in the extracted node set
# (edge_operations.py: "Target entity not found in nodes for edge relation").
# The root cause on terse, value-centric text is that the *node* step never
# creates an entity for a value/role/system, so the edge has nowhere to attach.
# These entity-type hints + instructions bias node extraction to capture them.


class Value(BaseModel):
    """A concrete value, quantity, setting, limit, date, or amount stated as a fact
    (e.g. '40 requests per minute', 'March 14', '$5,000'). Use the full phrase as the
    entity name so a relation can point at it."""


class Role(BaseModel):
    """A role, title, or responsibility (e.g. 'security lead', 'billing service owner',
    'on-call engineer')."""


class System(BaseModel):
    """A team, company, service, product, system, or named tool (e.g. 'platform team',
    'billing service', 'SQLite', 'Northwind')."""


GUIDED_ENTITY_TYPES: dict[str, type] = {"Value": Value, "Role": Role, "System": System}
GUIDED_EXTRACTION_INSTRUCTIONS = (
    "Also extract concrete values, quantities, dates, roles, and named teams / services / "
    "products / tools as their own entities (e.g. '40 requests per minute', 'March 14', "
    "'security lead', 'billing service', 'SQLite', 'Northwind'), so that relations BETWEEN "
    "them can be formed. Prefer creating an entity for the object of a statement over omitting it."
)


# A session marker like "Session 1:", "Turn 12 :", "day 3:" — used as a robust
# fallback when a benchmark packs multiple sessions inline on a single line.
_SESSION_MARKER = re.compile(r"(?=\b(?:Session|Turn|Day|Message|Msg)\s+\d+\s*:)", re.IGNORECASE)
# The same marker anchored at the start of a segment, so we can strip it from the
# episode body — otherwise the extractor coins junk entities like "Session 2: Marketing".
_SESSION_PREFIX = re.compile(r"^(?:Session|Turn|Day|Message|Msg)\s+\d+\s*:\s*", re.IGNORECASE)


def _split_sessions(context: str, settings: dict[str, Any]) -> list[str]:
    """Split a context into ordered session texts.

    Order of preference: an explicit ``session_regex`` setting, then the literal
    ``session_delimiter``, then — if that still yields a single block but the text
    clearly contains ``Session N:`` markers — an automatic split on those markers.
    Cross-episode benchmarks (e.g. LoCoMo) sometimes pack sessions inline on one
    line, so the automatic fallback keeps each session a distinct episode.
    """
    regex = settings.get("session_regex")
    if regex:
        segments = [seg.strip() for seg in re.split(str(regex), context) if seg.strip()]
    else:
        delimiter = str(settings.get("session_delimiter", "\n\n"))
        segments = [seg.strip() for seg in context.split(delimiter) if seg.strip()]
    if len(segments) <= 1:
        auto = [seg.strip() for seg in _SESSION_MARKER.split(context) if seg.strip()]
        if len(auto) > 1:
            segments = auto
    # Strip the leading "Session N:" / "Turn N:" marker from each episode body so
    # the extractor sees clean prose, not "Session 2: Marketing ...".
    segments = [cleaned for seg in segments if (cleaned := _SESSION_PREFIX.sub("", seg).strip())]
    if not segments:
        segments = [context.strip()]
    return segments


def _episodes(sample: BenchmarkSample, settings: dict[str, Any]) -> list[tuple[str, str]]:
    """Split a sample's context into ordered episodes (sessions).

    Splitting on session boundaries makes each session one episode, so the
    long-term tier must recall *across* sessions rather than within one window.
    """
    segments = _split_sessions(sample.context, settings)
    return [(f"{sample.sample_id}-s{index:03d}", text) for index, text in enumerate(segments)]


def _build_graphiti(settings: dict[str, Any]) -> "Graphiti":
    """Construct Graphiti with an OpenAI-compatible LLM + embedder (Together.ai default).

    Everything is env-overridable so the same baseline runs against Together, a
    local LM Studio/Ollama server, or OpenAI without code changes:

    - ``LLM_BASE_URL`` / ``LLM_API_KEY`` (or ``TOGETHER_API_KEY``)
    - ``LLM_MODEL`` (heavy extraction) / ``SMALL_MODEL`` (lighter passes)
    - ``EMB_MODEL`` / ``EMB_DIM`` (must match the embedder's true dimension)
    - ``GRAPHITI_MAX_COROUTINES`` (serialize LLM calls to respect rate limits)
    - graph connection via the ``*_env`` setting names (default GRAPH_URI/USER/PASSWORD)
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
        cross_encoder=OpenAIRerankerClient(config=llm_config),
        max_coroutines=max_coroutines,
    )


async def _ingest_and_search(
    *,
    sample: BenchmarkSample,
    settings: dict[str, Any],
    group_id: str,
    num_results: int,
) -> list[dict[str, Any]]:
    assert Graphiti is not None and EpisodeType is not None  # guaranteed by _availability()
    graphiti = _build_graphiti(settings)
    try:
        await graphiti.build_indices_and_constraints()
        # Stamp sessions with an increasing valid-time so the graph carries the
        # episode order — later sessions can supersede earlier facts.
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
        # Combined edge + node retrieval. Edges carry entity-entity facts with
        # bi-temporal validity; nodes carry the name/summary/attributes where a
        # scalar or attribute fact (e.g. a rate limit, a status) lives when it
        # never became an edge between two entities. `search_mode="edges"` falls
        # back to edge-only. Episodes/communities are intentionally ignored so this
        # stays graph retrieval, not raw-text RAG over the sessions.
        recipe = EDGE_HYBRID_SEARCH_RRF if str(settings.get("search_mode", "combined")) == "edges" else COMBINED_HYBRID_SEARCH_RRF
        config = recipe.model_copy(update={"limit": num_results})
        results = await graphiti.search_(sample.question, config=config, group_ids=[group_id])
        edges = list(results.edges)
        nodes = list(getattr(results, "nodes", []))
    finally:
        await graphiti.close()

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
    # Node retrieval, two modes (``node_text_mode``):
    #   "summary" (default): name + regional summary + attributes — maximizes recall, but
    #     the summary aggregates history (can mention a since-superseded value) and so can
    #     reintroduce a stale leak on supersession questions.
    #   "current_edges": name + the node's *current* incident edge facts only, dropping the
    #     history-aggregating summary. Entity-name answers still resolve (the name is in the
    #     text) and current-state answers stay clean — best-of-both for bi-temporal
    #     correctness without losing identity recall.
    node_text_mode = str(settings.get("node_text_mode", "summary"))
    incident_current: dict[str, list[str]] = {}
    if node_text_mode == "current_edges":
        for edge in edges:
            if getattr(edge, "invalid_at", None) is not None:
                continue  # current edges only
            for endpoint in (getattr(edge, "source_node_uuid", None), getattr(edge, "target_node_uuid", None)):
                if endpoint:
                    incident_current.setdefault(endpoint, []).append(edge.fact)
    for node in nodes:
        name = (getattr(node, "name", "") or "").strip()
        if node_text_mode == "current_edges":
            edge_facts = incident_current.get(getattr(node, "uuid", None), [])
            text = f"{name}: {' '.join(edge_facts)}".strip() if edge_facts else name
        else:
            summary = (getattr(node, "summary", "") or "").strip()
            attributes = getattr(node, "attributes", {}) or {}
            attr_text = "; ".join(
                f"{key}: {value}" for key, value in attributes.items() if value not in (None, "", [], {})
            )
            text = f"{name} — {summary}" if (name and summary) else (summary or name)
            if attr_text:
                text = f"{text} ({attr_text})" if text else attr_text
        text = text.strip()
        if not text:
            continue
        facts.append(
            {
                "fact": text,
                "kind": "node",
                "uuid": getattr(node, "uuid", None),
                "valid_at": None,
                "invalid_at": None,
                "current": True,
            }
        )
    return facts


def _order_facts(facts: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply the long-term tier's temporal preference to retrieved facts.

    The whole point of a bi-temporal store is that *current* facts answer
    "what is true now". ``current_facts_only`` drops superseded edges entirely
    (falling back to all facts if nothing is current); otherwise ``prefer_current``
    orders current facts first. This is what gives the temporal graph an edge over
    a local window that has no notion of supersession — without it, the extractive
    predictor happily quotes a stale value alongside the current one.
    """
    if not facts:
        return facts
    # Edges carry bi-temporal validity (invalid_at), so a *current edge* is the
    # authoritative source for "what is true now". Node name/summary/attributes
    # supplement recall but a node's regional summary can aggregate history (it may
    # mention a since-superseded value), so it ranks AFTER current edges.
    current_edges = [f for f in facts if f.get("current") and f.get("kind") != "node"]
    current_nodes = [f for f in facts if f.get("current") and f.get("kind") == "node"]
    superseded = [f for f in facts if not f.get("current")]
    if bool(settings.get("current_facts_only", False)):
        ordered = current_edges + current_nodes
        return ordered or facts
    if bool(settings.get("prefer_current", True)):
        return current_edges + current_nodes + superseded
    return facts


def _availability(settings: dict[str, Any]) -> tuple[bool, str | None]:
    if Graphiti is None:
        return False, f"graphiti-core not installed ({_GRAPHITI_IMPORT_ERROR})"
    if bool(settings.get("require_graph_uri", True)):
        uri_env = str(settings.get("graph_uri_env", "GRAPH_URI"))
        if not os.environ.get(uri_env):
            return False, f"{uri_env} not set (no graph backend configured)"
    if bool(settings.get("require_llm_key", True)):
        if not (os.environ.get("LLM_API_KEY") or os.environ.get("TOGETHER_API_KEY")):
            return False, "no LLM API key (set TOGETHER_API_KEY or LLM_API_KEY)"
    return True, None


def run_graphiti_cross_episode_baseline(
    *,
    baseline: BaselineSpec,
    benchmark: BenchmarkSpec,
    samples: list[BenchmarkSample],
    runtime_profile: RuntimeProfile,
    output_dir: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    settings = baseline.settings
    num_results = int(settings.get("num_results", baseline.max_chunks_to_retrieve or 8))
    available, unavailable_reason = _availability(settings)
    can_run = available and not dry_run

    artifacts: list[dict[str, Any]] = []
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        group_id = f"{benchmark.id}__{sample.sample_id}"
        facts: list[dict[str, Any]] = []
        status = "ok"
        error: str | None = None

        if can_run:
            try:
                facts = asyncio.run(
                    _ingest_and_search(
                        sample=sample,
                        settings=settings,
                        group_id=group_id,
                        num_results=num_results,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - record the failure in the artifact
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
        else:
            status = "not_executed"
            error = "dry-run" if dry_run else unavailable_reason

        ordered_facts = _order_facts(facts, settings)
        retrieval_texts = [str(item["fact"]) for item in ordered_facts]
        prediction = build_prediction_block(
            question=sample.question,
            retrieval_texts=retrieval_texts,
            expected_answer_type=sample.expected_answer_type,
            reference_answer=sample.reference_answer,
            reference_answers=sample.reference_answers,
        )

        episodes = _episodes(sample, settings)
        artifact: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "status": status,
            "error": error,
            "baseline": {
                "id": baseline.id,
                "name": baseline.name,
                "type": baseline.type,
                "retrieval_strategy": baseline.retrieval_strategy,
                "settings": baseline.settings,
            },
            "benchmark": {
                "id": benchmark.id,
                "name": benchmark.name,
                "family": benchmark.family,
            },
            "runtime_profile": runtime_profile.to_dict(),
            "fixture": {
                "sample_id": sample.sample_id,
                "source": sample.source,
                "task_type": sample.task_type,
                "question": sample.question,
                "expected_answer_type": sample.expected_answer_type,
                "reference_answers": list(sample.reference_answers),
                "metadata": sample.metadata,
            },
            "memory_state": {
                "cross_episode": {
                    "group_id": group_id,
                    "episode_count": len(episodes),
                    "retrieved_fact_count": len(facts),
                    "current_fact_count": sum(1 for item in facts if item.get("current")),
                    "superseded_fact_count": sum(1 for item in facts if not item.get("current")),
                    "facts": facts,
                }
            },
            "retrieval_preview": {"cross_episode": retrieval_texts},
            "prediction": prediction,
            "limitations": [
                "Long-term tier only: ingests this sample's sessions into a fresh per-sample graph; it does not yet share one graph across samples.",
                "Episodes are stamped with synthetic increasing valid-times, not the benchmark's real session timestamps.",
                "Extraction quality and cost depend on the configured LLM; emits a not_executed artifact when no graph backend is configured.",
            ],
            "budget_ledger": BudgetLedger(
                proxy_token_ops=None,
                active_memory_units=len(facts),
                notes=[
                    f"Runtime profile: {runtime_profile.profile_id}",
                    "Cross-episode symbolic memory via Graphiti (bi-temporal knowledge graph).",
                    f"Status: {status}" + (f" ({error})" if error else ""),
                ],
            ).to_dict(),
        }

        suffix = f"__{sample.sample_id}" if len(samples) > 1 or sample.source == "local" else ""
        artifact_path = output_dir / f"{baseline.id}__{benchmark.id}{suffix}.json"
        artifact["artifact_path"] = str(artifact_path)
        if not dry_run:
            artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        artifacts.append(artifact)

    return artifacts
