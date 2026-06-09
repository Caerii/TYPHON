"""Turn a sample's context into ordered episode texts (pure, no graph/LLM).

Cross-episode benchmarks pack a multi-session history into one ``context`` string. To
exercise *cross*-episode recall, each session must become its own episode. This module
splits the context into ordered sessions and strips the session marker so the extractor
sees clean prose. Pure functions — unit-testable without a backend.
"""

from __future__ import annotations

import re
from typing import Any

from typhon.benchmarks.base import BenchmarkSample

# A session marker like "Session 1:", "Turn 12 :", "day 3:" — used as a robust fallback
# when a benchmark packs multiple sessions inline on a single line.
_SESSION_MARKER = re.compile(r"(?=\b(?:Session|Turn|Day|Message|Msg)\s+\d+\s*:)", re.IGNORECASE)
# The same marker anchored at the start of a segment, so we can strip it from the episode
# body — otherwise the extractor coins junk entities like "Session 2: Marketing".
_SESSION_PREFIX = re.compile(r"^(?:Session|Turn|Day|Message|Msg)\s+\d+\s*:\s*", re.IGNORECASE)
# Sentence boundary — used to chunk a long session under the extraction token budget
# without splitting mid-sentence (so a relation stays inside one chunk).
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Greedily pack sentences into chunks of at most ``max_chars`` characters.

    Long multi-turn sessions (e.g. real LoCoMo) make a single episode's extraction overflow
    the model's output budget, which drops edges. Chunking keeps each extraction call small
    while never splitting a sentence (unless one sentence alone exceeds the budget, which is
    then hard-split). ``max_chars <= 0`` disables chunking.
    """
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:  # a single oversized sentence — hard-split it
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:].strip()
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _split_sessions(context: str, settings: dict[str, Any]) -> list[str]:
    """Split a context into ordered session texts.

    Order of preference: an explicit ``session_regex`` setting, then the literal
    ``session_delimiter``, then — if that still yields a single block but the text clearly
    contains ``Session N:`` markers — an automatic split on those markers. Each segment has
    its leading marker stripped so the extractor sees clean prose.
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
    segments = [cleaned for seg in segments if (cleaned := _SESSION_PREFIX.sub("", seg).strip())]
    if not segments:
        segments = [context.strip()]
    return segments


def _episodes(sample: BenchmarkSample, settings: dict[str, Any]) -> list[tuple[str, str]]:
    """Split a sample's context into ordered ``(episode_name, episode_body)`` pairs.

    Splitting on session boundaries makes each session one episode, so the long-term tier
    must recall *across* sessions rather than within one window. When ``max_episode_chars``
    is set (> 0), each session is further chunked under that budget (on sentence boundaries)
    so extraction on long multi-turn sessions doesn't overflow the model's output budget;
    chunk episodes are named ``…-sNNN-cMM`` and keep increasing valid-times. Default (0)
    leaves one episode per session, unchanged.
    """
    segments = _split_sessions(sample.context, settings)
    max_chars = int(settings.get("max_episode_chars", 0) or 0)
    episodes: list[tuple[str, str]] = []
    for s_index, segment in enumerate(segments):
        chunks = _chunk_text(segment, max_chars) if max_chars > 0 else [segment]
        if len(chunks) <= 1:
            episodes.append((f"{sample.sample_id}-s{s_index:03d}", chunks[0] if chunks else segment))
        else:
            for c_index, chunk in enumerate(chunks):
                episodes.append((f"{sample.sample_id}-s{s_index:03d}-c{c_index:02d}", chunk))
    return episodes
