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
    """Split a sample's context into ``(episode_name, episode_body)`` pairs in order.

    Splitting on session boundaries makes each session one episode, so the long-term tier
    must recall *across* sessions rather than within one window.
    """
    segments = _split_sessions(sample.context, settings)
    return [(f"{sample.sample_id}-s{index:03d}", text) for index, text in enumerate(segments)]
