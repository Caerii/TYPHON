"""Serverless listwise reranker — a RankGPT-style cross-encoder that runs on any
OpenAI-compatible chat API (Together, local LM Studio/Ollama), not just OpenAI.

Why this exists: Graphiti's stock ``OpenAIRerankerClient`` ranks by reading the
per-token ``logprobs`` of a forced "True"/"False" answer, nudged with
``logit_bias={'6432': 1, '7983': 1}`` — but those are **OpenAI tokenizer IDs**, and
Together/Llama returns an *empty* ``logprobs.content``. With no logprobs the stock
client's ``zip(passages, scores, strict=True)`` raises, so the cross-encoder recipe
crashes on every non-OpenAI backend (verified against Together's Llama-3.3-70B).

Instead of N per-passage logprob calls, this issues **one** strict-structured-output
call that scores *all* passages together (listwise). That is both cheaper (1 call vs N)
and a stronger signal — the model compares candidates against each other, RankGPT-style.
Scores are Pydantic-validated; on any failure it falls back to the input order with
descending neutral scores, so retrieval degrades to the pre-rerank ranking rather than
crashing. Token usage is accumulated on ``self.usage`` so the run can price the rerank
cost alongside extraction.

The scoring/ordering logic lives in module-level pure helpers (``_fallback_ranking``,
``_apply_scores``, ``_build_messages``) so it is unit-testable without a graph backend or
an LLM; the ``ListwiseReranker`` class (defined only when graphiti-core is importable)
is a thin async wrapper that calls the model and delegates to them.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import BaseModel, Field

from ._deps import CrossEncoderClient, openai

logger = logging.getLogger(__name__)


class _PassageScore(BaseModel):
    """One passage's relevance to the query, keyed by its position in the input list."""

    index: int = Field(description="0-based index of the passage in the provided list.")
    relevance: float = Field(
        description="Relevance to the query from 0.0 (unrelated) to 1.0 (directly answers it)."
    )


class _RerankResult(BaseModel):
    """Listwise scores for every passage (one entry per passage, any order)."""

    scores: list[_PassageScore]


_SYSTEM_PROMPT = (
    "You are a precise relevance ranker. Given a QUERY and a numbered list of PASSAGES, "
    "score how relevant each passage is to answering the query, from 0.0 (unrelated) to "
    "1.0 (directly contains the answer). Judge each passage on whether it helps answer the "
    "specific question — a passage that merely shares a word with the query but does not "
    "answer it is NOT relevant and should score low. Return a score for every passage by its "
    "index."
)


# --- pure helpers (no graphiti / no network — unit-tested directly) ------------


def _fallback_ranking(passages: list[str]) -> list[tuple[str, float]]:
    """Input order, descending neutral scores — used when the model call fails.

    Preserving input order means a reranker failure degrades retrieval to the
    pre-rerank ranking (RRF/cosine order) rather than crashing or shuffling.
    """
    n = len(passages)
    return [(text, 1.0 - i / n if n else 0.0) for i, text in enumerate(passages)]


def _apply_scores(
    passages: list[str], scored: list[tuple[int, float]]
) -> list[tuple[str, float]]:
    """Map model (index, relevance) pairs onto passages and sort by descending relevance.

    Indices out of range are ignored; passages the model omitted score 0.0 (not relevant
    enough to mention). Relevance is clamped to [0, 1] so a stray value cannot reorder
    wildly. Ties keep input order (Python sort is stable), so the upstream ranking breaks
    ties — a sensible prior.
    """
    by_index: dict[int, float] = {}
    for index, relevance in scored:
        if 0 <= index < len(passages):
            by_index[index] = max(0.0, min(1.0, float(relevance)))
    ranked = [(text, by_index.get(i, 0.0)) for i, text in enumerate(passages)]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def _build_messages(query: str, passages: list[str]) -> list[dict[str, str]]:
    """Compose the system+user messages for a listwise rerank over ``passages``."""
    numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(passages))
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"QUERY:\n{query}\n\nPASSAGES:\n{numbered}"},
    ]


# --- async client (only when graphiti-core is importable) ----------------------


if CrossEncoderClient is not None and openai is not None:

    class ListwiseReranker(CrossEncoderClient):  # type: ignore[misc,valid-type]
        """Rank passages by relevance to a query in a single structured-output call.

        Implements graphiti's ``CrossEncoderClient`` interface (``rank`` returns
        ``[(passage, score), ...]`` sorted by descending relevance), so it drops in
        wherever ``OpenAIRerankerClient`` is used — but works on any OpenAI-compatible
        backend because it parses a validated JSON object, not token logprobs.
        """

        def __init__(self, config: Any, max_passages: int = 32) -> None:
            self.config = config
            self.model = getattr(config, "model", None) or "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            self.max_passages = max_passages
            self.client = openai.AsyncOpenAI(
                api_key=getattr(config, "api_key", "") or "",
                base_url=getattr(config, "base_url", None),
            )
            self.usage: dict[str, int] = {
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        def _record_usage(self, completion: Any) -> None:
            usage = getattr(completion, "usage", None)
            self.usage["llm_calls"] += 1
            if usage is None:
                return
            self.usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            self.usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)

        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
            if not passages:
                return []
            if len(passages) == 1:
                return [(passages[0], 1.0)]

            # Bound the prompt: rank at most ``max_passages`` (graphiti already pre-limits,
            # but a runaway candidate set would blow the context and the cost).
            capped = passages[: self.max_passages]
            try:
                completion = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=cast("Any", _build_messages(query, capped)),
                    temperature=0.0,
                    response_format=_RerankResult,
                )
                self._record_usage(completion)
                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    return _fallback_ranking(capped)
            except Exception as exc:  # noqa: BLE001 - any failure must not crash search
                logger.warning("ListwiseReranker failed (%s); falling back to input order", exc)
                return _fallback_ranking(capped)

            return _apply_scores(capped, [(s.index, s.relevance) for s in parsed.scores])

else:  # graphiti-core unavailable

    ListwiseReranker = None  # type: ignore[assignment]
