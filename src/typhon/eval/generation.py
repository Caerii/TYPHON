"""Generative reading + LLM-as-judge — climb onto the SOTA evaluation axis.

Every headline agent-memory SOTA number (Mem0, Zep, MemMachine, …) is an **LLM-as-judge
binary-correctness score over a *generated* answer**, not extractive token-recall. This module
adds both halves so TYPHON can report a comparable number:

- a **generative reader** that composes an answer from a baseline's retrieved facts, and
- an **LLM-as-judge** that grades that answer against the reference(s) (the LoCoMo/Mem0 protocol).

It is applied **post-hoc over a baseline artifact's ``retrieval_preview``** (the same retrieved
texts the extractive path saw), so (a) it works on frozen runs without re-paying ingestion, and
(b) the *identical* reader+judge runs over both baselines — the head-to-head still isolates
retrieval. The extractive ``prediction`` is preserved; we add ``prediction.generative`` and
``prediction.llm_judge`` alongside it.

Prompt builders and the verdict parser are pure (unit-testable); only ``OpenAIChat`` calls a model,
and it is constructed only when an API key is present (``make_chat_from_env`` returns ``None``
otherwise, so callers degrade gracefully).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol, cast

# --- pure prompt construction + parsing (no I/O, unit-testable) ---------------------

_READER_SYSTEM = (
    "You answer questions using ONLY the provided memory facts. "
    "The facts are retrieved from a long multi-session conversation. "
    "Answer concisely — a few words for a factual question, a short phrase otherwise. "
    "Resolve relative dates to absolute ones when the facts allow it. "
    "If the facts do not contain the answer, reply exactly: I don't know."
)

_JUDGE_SYSTEM = (
    "You are an impartial grader. Decide whether a PREDICTED answer is correct for the QUESTION, "
    "given the REFERENCE answer(s). Grade CORRECT if the prediction conveys the same key "
    "information as ANY reference — paraphrase, extra context, or different formatting are fine. "
    "Grade WRONG if it contradicts the reference, omits the key fact, or says it does not know. "
    'Respond with a single JSON object: {"correct": true|false, "reason": "<short>"}.'
)


def _format_facts(retrieval_texts: list[str]) -> str:
    if not retrieval_texts:
        return "(no facts retrieved)"
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(retrieval_texts))


def build_reader_messages(
    question: str, retrieval_texts: list[str], expected_answer_type: str | None = None
) -> list[dict[str, str]]:
    """Reader prompt: compose an answer from retrieved facts (pure)."""
    hint = ""
    if (expected_answer_type or "").lower() == "classification":
        hint = " Answer with just the label."
    user = (
        f"Memory facts:\n{_format_facts(retrieval_texts)}\n\n"
        f"Question: {question}\n\nAnswer concisely:{hint}"
    )
    return [{"role": "system", "content": _READER_SYSTEM}, {"role": "user", "content": user}]


def build_judge_messages(
    question: str, predicted: str, references: list[str]
) -> list[dict[str, str]]:
    """Judge prompt: grade predicted vs reference(s) (pure)."""
    refs = "\n".join(f"- {r}" for r in references if r) or "(none provided)"
    user = (
        f"QUESTION: {question}\n\nREFERENCE answer(s):\n{refs}\n\n"
        f"PREDICTED answer: {predicted}\n\nGrade:"
    )
    return [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_judge_verdict(raw: str) -> bool:
    """Parse a judge response into a correctness bool. Robust to non-JSON output (pure).

    Prefers the JSON ``correct`` field; falls back to a CORRECT/WRONG keyword scan so a model
    that ignores the format still grades. Defaults to False (a malformed/empty verdict is not a
    pass — fail closed so the J-score never over-counts).
    """
    if not raw:
        return False
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj.get("correct"), bool):
                return obj["correct"]
            if isinstance(obj.get("correct"), str):
                return obj["correct"].strip().lower() in {"true", "yes", "correct"}
        except (json.JSONDecodeError, AttributeError):
            pass
    low = raw.lower()
    if re.search(r"\bwrong\b|\bincorrect\b|\bfalse\b", low):
        return False
    return bool(re.search(r"\bcorrect\b|\btrue\b|\byes\b", low))


def retrieved_texts(artifact: dict[str, Any]) -> list[str]:
    """Pull a baseline artifact's retrieved texts from ``retrieval_preview`` (any key) (pure).

    Both baselines store retrieval under ``retrieval_preview`` keyed by layer
    (``local_exact`` / ``cross_episode``); take the first non-empty list so the reader sees
    exactly what the extractive path saw.
    """
    preview = artifact.get("retrieval_preview") or {}
    for value in preview.values():
        if isinstance(value, list) and value:
            return [str(v) for v in value]
    return []


def references_of(artifact: dict[str, Any]) -> list[str]:
    """Reference answers for the judge (fixture first, then the prediction block) (pure)."""
    fixture = artifact.get("fixture") or {}
    prediction = artifact.get("prediction") or {}
    refs = list(fixture.get("reference_answers") or prediction.get("reference_answers") or [])
    single = prediction.get("reference_answer")
    if single and single not in refs:
        refs.insert(0, single)
    return [str(r) for r in refs if r]


# --- the model-calling layer (gated; constructed only with an API key) ---------------


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAIChat:
    """Minimal OpenAI-compatible chat wrapper (Together default), with usage accumulation.

    Lazy-imports ``openai`` so this module imports even where the SDK is absent (the pure
    helpers above stay usable, and tests inject a fake ``ChatModel``). Temperature defaults to
    0.0 for deterministic reading + judging.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        import openai  # lazy: only needed for a real run

        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage: dict[str, int] = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def complete(self, messages: list[dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=cast("Any", messages),  # plain dicts; openai's typed params are over-strict
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = getattr(resp, "usage", None)
        self.usage["llm_calls"] += 1
        if usage is not None:
            self.usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            self.usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
        return resp.choices[0].message.content or ""


def make_chat_from_env(settings: dict[str, Any] | None = None) -> OpenAIChat | None:
    """Build an ``OpenAIChat`` from env (Together default), or ``None`` if no key is set.

    Mirrors the graphiti baseline's env contract (``LLM_BASE_URL`` / ``LLM_API_KEY`` |
    ``TOGETHER_API_KEY`` / ``LLM_MODEL``) so reader, judge, and extractor share one configuration.
    Returning ``None`` lets the driver skip generative scoring cleanly (e.g. CI without a key).
    """
    settings = settings or {}
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("LLM_BASE_URL", str(settings.get("llm_base_url", "https://api.together.xyz/v1")))
    model = os.environ.get("LLM_MODEL", str(settings.get("llm_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo")))
    return OpenAIChat(base_url=base_url, api_key=api_key, model=model)


def read_answer(
    chat: ChatModel, question: str, retrieval_texts: list[str], expected_answer_type: str | None = None
) -> str:
    return chat.complete(build_reader_messages(question, retrieval_texts, expected_answer_type)).strip()


def judge_answer(
    chat: ChatModel, question: str, predicted: str, references: list[str]
) -> dict[str, Any]:
    raw = chat.complete(build_judge_messages(question, predicted, references))
    return {"correct": parse_judge_verdict(raw), "raw": raw.strip()[:500]}


def decorate_artifact(
    artifact: dict[str, Any],
    chat: ChatModel,
    *,
    generate: bool = True,
    judge: bool = True,
    judge_model: str | None = None,
    reader_model: str | None = None,
) -> dict[str, Any]:
    """Add a generative answer and/or an LLM-judge verdict to one artifact, in place.

    The reader composes an answer from the artifact's retrieved facts; the judge grades that
    generated answer (or the extractive ``predicted_answer`` when ``generate=False``) against the
    references. Writes ``prediction.generative = {predicted_answer, answer_mode, …}`` and
    ``prediction.llm_judge = {correct, target, model, raw}``. Pure-data otherwise — the caller
    decides whether to persist.
    """
    prediction = artifact.setdefault("prediction", {})
    question = (artifact.get("fixture") or {}).get("question", "")
    facts = retrieved_texts(artifact)
    refs = references_of(artifact)
    expected = (artifact.get("fixture") or {}).get("expected_answer_type")

    graded_answer = prediction.get("predicted_answer", "")
    target = "extractive"
    if generate:
        answer = read_answer(chat, question, facts, expected)
        prediction["generative"] = {
            "predicted_answer": answer,
            "answer_mode": "generative",
            "model": reader_model,
            "n_facts": len(facts),
        }
        graded_answer = answer
        target = "generative"

    if judge:
        verdict = judge_answer(chat, question, graded_answer, refs)
        prediction["llm_judge"] = {
            "correct": verdict["correct"],
            "target": target,
            "model": judge_model,
            "raw": verdict["raw"],
        }
    return artifact
