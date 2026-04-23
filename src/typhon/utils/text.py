from __future__ import annotations

import re

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "which",
    "with",
}


def normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", token.lower())


def significant_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in text.split():
        token = normalize_token(raw)
        if not token or token in STOPWORDS:
            continue
        terms.append(token)
    return terms
