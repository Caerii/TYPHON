"""Importer: snap-research LoCoMo (``locomo10.json``) -> TYPHON benchmark samples.

LoCoMo is a long, multi-session dialogue dataset with QA over the whole history.
Each conversation becomes a shared context (sessions in order, with their real
timestamps); each QA pair becomes one sample. Session text is prefixed
``Session N:`` so the graphiti baseline's session splitter treats each session as
one episode. Category 5 is adversarial (no ``answer``; the expected graceful
response lives in ``adversarial_answer``).

Pure transform — no I/O — so it is unit-testable. See
``scripts/build_locomo_benchmark.py`` for the CLI that reads the JSON and writes
``data/benchmarks/<id>/samples.jsonl``.
"""

from __future__ import annotations

import re
from typing import Any

_SESSION_KEY = re.compile(r"^session_(\d+)$")


def _ordered_sessions(conversation: dict[str, Any]) -> list[tuple[int, list, str]]:
    """Return (n, turns, date_time) sorted by numeric session index."""
    sessions: list[tuple[int, list, str]] = []
    for key, turns in conversation.items():
        match = _SESSION_KEY.match(key)
        if not match or not isinstance(turns, list):
            continue
        index = int(match.group(1))
        date = str(conversation.get(f"session_{index}_date_time", "") or "")
        sessions.append((index, turns, date))
    sessions.sort(key=lambda item: item[0])
    return sessions


def _session_text(index: int, turns: list, date: str) -> str:
    body = " ".join(
        f"{turn.get('speaker', '')}: {turn.get('text', '')}".strip()
        for turn in turns
        if isinstance(turn, dict)
    )
    prefix = f"Session {index}: ({date})" if date else f"Session {index}:"
    return f"{prefix} {body}".strip()


def conversation_context(conversation: dict[str, Any]) -> str:
    """Flatten a conversation's sessions into one ordered, marker-prefixed context."""
    return " ".join(_session_text(n, turns, date) for n, turns, date in _ordered_sessions(conversation))


def build_locomo_samples(
    raw: list[dict[str, Any]],
    *,
    conversations: int | None = None,
    max_per_conversation: int | None = None,
    include_adversarial: bool = True,
) -> list[dict[str, Any]]:
    """Map raw LoCoMo conversations into TYPHON sample dicts.

    Args:
        raw: parsed ``locomo10.json`` (a list of conversations).
        conversations: cap the number of conversations (None = all).
        max_per_conversation: cap QA pairs per conversation (None = all).
        include_adversarial: include category-5 (adversarial) QA.
    """
    samples: list[dict[str, Any]] = []
    selected = raw[:conversations] if conversations is not None else raw
    for conv_index, conv in enumerate(selected):
        conv_id = str(conv.get("sample_id", f"conv{conv_index}"))
        context = conversation_context(conv.get("conversation", {}))
        kept = 0
        for qa_index, qa in enumerate(conv.get("qa", []) or []):
            adversarial = "answer" not in qa and "adversarial_answer" in qa
            if adversarial and not include_adversarial:
                continue
            answer = qa.get("answer", qa.get("adversarial_answer"))
            question = qa.get("question")
            if answer is None or not question:
                continue
            samples.append(
                {
                    "sample_id": f"{conv_id}_q{qa_index:03d}",
                    "split": "test",
                    "task_type": "conversation_qa",
                    "question": question,
                    "context": context,
                    "expected_answer_type": "short_text",
                    "reference_answer": str(answer),
                    "metadata": {
                        "conversation": conv_id,
                        "category": qa.get("category"),
                        "adversarial": adversarial,
                        "evidence": qa.get("evidence", []),
                    },
                }
            )
            kept += 1
            if max_per_conversation is not None and kept >= max_per_conversation:
                break
    return samples
