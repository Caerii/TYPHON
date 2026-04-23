from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    id: str
    name: str
    family: str
    paper_url: str
    official_repo: str | None
    dataset_ref: str | None
    tasks: tuple[str, ...]
    default_chunk_size: int
    default_local_window: int
    smoke_fixture: str
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkSpec":
        return cls(
            id=payload["id"],
            name=payload["name"],
            family=payload["family"],
            paper_url=payload["paper_url"],
            official_repo=payload.get("official_repo") or None,
            dataset_ref=payload.get("dataset_ref") or None,
            tasks=tuple(payload.get("tasks", [])),
            default_chunk_size=int(payload["default_chunk_size"]),
            default_local_window=int(payload["default_local_window"]),
            smoke_fixture=payload["smoke_fixture"],
            notes=payload.get("notes", ""),
        )


@dataclass(frozen=True)
class SmokeFixture:
    task_type: str
    question: str
    context: str
    expected_answer_type: str
    reference_answer: str | None = None
    reference_answers: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SmokeFixture":
        raw_reference_answers = payload.get("reference_answers")
        if raw_reference_answers is None and isinstance(payload.get("answers"), list):
            raw_reference_answers = payload.get("answers")
        reference_answers = tuple(
            str(item).strip()
            for item in (raw_reference_answers or [])
            if str(item).strip()
        )
        reference_answer = payload.get("reference_answer")
        if reference_answer is None and reference_answers:
            reference_answer = reference_answers[0]
        return cls(
            task_type=payload["task_type"],
            question=payload["question"],
            context=payload["context"],
            expected_answer_type=payload["expected_answer_type"],
            reference_answer=None if reference_answer is None else str(reference_answer),
            reference_answers=reference_answers,
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    split: str
    task_type: str
    question: str
    context: str
    expected_answer_type: str
    reference_answer: str | None = None
    reference_answers: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "fixture"
    source_path: str | None = None

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        default_sample_id: str,
        default_split: str,
        source: str,
        source_path: str | None,
    ) -> "BenchmarkSample":
        raw_reference_answers = payload.get("reference_answers")
        if raw_reference_answers is None and isinstance(payload.get("answers"), list):
            raw_reference_answers = payload.get("answers")
        reference_answers = tuple(
            str(item).strip()
            for item in (raw_reference_answers or [])
            if str(item).strip()
        )
        reference_answer = payload.get("reference_answer")
        if reference_answer is None and reference_answers:
            reference_answer = reference_answers[0]
        return cls(
            sample_id=payload.get("sample_id", default_sample_id),
            split=payload.get("split", default_split),
            task_type=payload["task_type"],
            question=payload["question"],
            context=payload["context"],
            expected_answer_type=payload["expected_answer_type"],
            reference_answer=None if reference_answer is None else str(reference_answer),
            reference_answers=reference_answers,
            metadata=dict(payload.get("metadata", {})),
            source=source,
            source_path=source_path,
        )
