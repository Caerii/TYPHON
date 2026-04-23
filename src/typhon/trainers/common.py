from __future__ import annotations

import re

from typhon.benchmarks.base import BenchmarkSpec, SmokeFixture
from typhon.runtime.base import RuntimeProfile
from typhon.utils.text import significant_terms


def chunk_context(text: str, chunk_size: int) -> list[tuple[int, list[str]]]:
    words = text.split()
    return [
        (start // chunk_size, words[start : start + chunk_size])
        for start in range(0, len(words), chunk_size)
    ]


def normalize_score(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def question_term_set(question: str) -> set[str]:
    return set(significant_terms(question))


def estimate_chunk_features(
    chunk_id: int,
    chunk_words: list[str],
    question_terms: set[str],
    family: str,
    fixture: SmokeFixture,
) -> dict[str, object]:
    chunk_text = " ".join(chunk_words)
    chunk_terms = significant_terms(chunk_text)
    term_set = set(chunk_terms)
    overlap_count = len(question_terms.intersection(term_set))
    normalized_overlap = min(1.0, overlap_count / max(1, min(6, len(question_terms) or 1)))
    novelty_ratio = len(term_set) / max(1, len(chunk_terms))
    numeric_signal = 1.0 if re.search(r"\d", chunk_text) else 0.0
    rare_hint = 1.0 if len(term_set) > 0 and len(term_set) == len(chunk_terms) else 0.0
    latent_constraint = family == "conversational_memory" and bool(
        fixture.metadata.get("latent_constraint")
    )
    family_persistence_bias = 1.0 if family in {
        "conversational_memory",
        "streaming_agentic_memory",
        "continual_learning",
    } else 0.0

    surprise = normalize_score(0.35 * numeric_signal + 0.35 * rare_hint + 0.3 * novelty_ratio)
    gradient_norm = normalize_score(
        0.55 * normalized_overlap
        + 0.25 * numeric_signal
        + 0.25 * family_persistence_bias
    )
    predicted_utility = normalize_score(
        0.5 * normalized_overlap
        + 0.25 * family_persistence_bias
        + 0.15 * numeric_signal
        + 0.15 * (1.0 if latent_constraint else 0.0)
    )
    novelty = normalize_score(novelty_ratio)

    return {
        "chunk_id": chunk_id,
        "text": chunk_text,
        "question_overlap_count": overlap_count,
        "normalized_overlap": normalized_overlap,
        "question_overlap_terms": sorted(question_terms.intersection(term_set)),
        "surprise": surprise,
        "gradient_norm": gradient_norm,
        "predicted_utility": predicted_utility,
        "novelty": novelty,
        "has_numeric_signal": bool(numeric_signal),
        "latent_constraint": latent_constraint,
        "token_count_estimate": len(chunk_words),
    }


def runtime_aware_chunk_size(
    spec: BenchmarkSpec,
    runtime_profile: RuntimeProfile,
    chunk_size_override: int | None = None,
) -> int:
    if chunk_size_override is not None:
        return max(1, chunk_size_override)
    preferred = int(runtime_profile.recommendations["preferred_chunk_size_tokens"])
    return min(spec.default_chunk_size, preferred)


def effective_local_window_tokens(
    spec: BenchmarkSpec,
    runtime_profile: RuntimeProfile,
    local_window_tokens_override: int | None = None,
) -> int:
    if local_window_tokens_override is not None:
        return max(1, local_window_tokens_override)
    return int(runtime_profile.recommendations["preferred_local_window_tokens"])


def proxy_token_ops(
    token_count: int,
    runtime_profile: RuntimeProfile,
    chunk_size: int,
    *,
    local_window_tokens_override: int | None = None,
    spec: BenchmarkSpec | None = None,
) -> int:
    if spec is None:
        local_window_tokens = (
            max(1, local_window_tokens_override)
            if local_window_tokens_override is not None
            else int(runtime_profile.recommendations["preferred_local_window_tokens"])
        )
    else:
        local_window_tokens = effective_local_window_tokens(
            spec=spec,
            runtime_profile=runtime_profile,
            local_window_tokens_override=local_window_tokens_override,
        )
    return token_count * max(local_window_tokens, chunk_size)
