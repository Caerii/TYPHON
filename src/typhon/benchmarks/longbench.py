from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typhon.benchmarks.base import BenchmarkSpec
from typhon.benchmarks.importer import write_normalized_benchmark_pack

LONG_BENCH_ENGLISH_TASKS = {
    "narrativeqa": {"task_type": "long_context_qa", "expected_answer_type": "long_text"},
    "qasper": {"task_type": "long_context_qa", "expected_answer_type": "long_text"},
    "multifieldqa_en": {"task_type": "long_context_qa", "expected_answer_type": "short_text"},
    "hotpotqa": {"task_type": "long_context_qa", "expected_answer_type": "short_text"},
    "2wikimqa": {"task_type": "long_context_qa", "expected_answer_type": "short_text"},
    "musique": {"task_type": "long_context_qa", "expected_answer_type": "short_text"},
    "gov_report": {"task_type": "summarization", "expected_answer_type": "summary"},
    "qmsum": {"task_type": "summarization", "expected_answer_type": "summary"},
    "multi_news": {"task_type": "summarization", "expected_answer_type": "summary"},
    "trec": {"task_type": "classification", "expected_answer_type": "classification"},
    "triviaqa": {"task_type": "few_shot_qa", "expected_answer_type": "short_text"},
    "samsum": {"task_type": "summarization", "expected_answer_type": "summary"},
    "passage_count": {"task_type": "synthetic_retrieval", "expected_answer_type": "short_text"},
    "passage_retrieval_en": {"task_type": "synthetic_retrieval", "expected_answer_type": "short_text"},
}


@dataclass(frozen=True)
class LongBenchTaskConfig:
    source_name: str
    task_type: str
    expected_answer_type: str
    include: bool = True
    max_samples: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LongBenchTaskConfig":
        source_name = str(payload["source_name"])
        defaults = LONG_BENCH_ENGLISH_TASKS.get(source_name, {})
        return cls(
            source_name=source_name,
            task_type=str(payload.get("task_type", defaults.get("task_type", "long_context_task"))),
            expected_answer_type=str(
                payload.get(
                    "expected_answer_type",
                    defaults.get("expected_answer_type", "short_text"),
                )
            ),
            include=bool(payload.get("include", True)),
            max_samples=None if payload.get("max_samples") is None else int(payload["max_samples"]),
        )


@dataclass(frozen=True)
class LongBenchLengthBucket:
    name: str
    min_length: int | None = None
    max_length: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LongBenchLengthBucket":
        return cls(
            name=str(payload["name"]),
            min_length=None if payload.get("min_length") is None else int(payload["min_length"]),
            max_length=None if payload.get("max_length") is None else int(payload["max_length"]),
        )

    def matches(self, value: int | None) -> bool:
        if value is None:
            return self.min_length is None and self.max_length is None
        if self.min_length is not None and value < self.min_length:
            return False
        if self.max_length is not None and value > self.max_length:
            return False
        return True


@dataclass(frozen=True)
class LongBenchImportConfig:
    benchmark_id: str
    dataset_id: str
    split: str
    pack_id: str
    description: str
    language: str
    max_samples_per_task: int | None
    tasks: tuple[LongBenchTaskConfig, ...]
    length_buckets: tuple[LongBenchLengthBucket, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LongBenchImportConfig":
        return cls(
            benchmark_id=str(payload.get("benchmark_id", "longbench")),
            dataset_id=str(payload.get("dataset_id", "THUDM/LongBench")),
            split=str(payload.get("split", "test")),
            pack_id=str(payload["pack_id"]),
            description=str(payload.get("description", "Imported LongBench pack")),
            language=str(payload.get("language", "en")),
            max_samples_per_task=None
            if payload.get("max_samples_per_task") is None
            else int(payload["max_samples_per_task"]),
            tasks=tuple(LongBenchTaskConfig.from_dict(item) for item in payload.get("tasks", [])),
            length_buckets=tuple(
                LongBenchLengthBucket.from_dict(item) for item in payload.get("length_buckets", [])
            ),
        )


def load_longbench_import_config(path: Path) -> LongBenchImportConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LongBenchImportConfig.from_dict(payload)


def _load_hf_rows(dataset_id: str, source_name: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "LongBench import requires the optional 'datasets' dependency. "
            "Use `uv run --with datasets typhon import-longbench --config ...` "
            "or add the adapter dependency before running this command."
        ) from exc

    dataset = load_dataset(
        dataset_id,
        source_name,
        split=split,
        trust_remote_code=True,
    )
    return [dict(row) for row in dataset]


def _bucket_name(length: int | None, buckets: tuple[LongBenchLengthBucket, ...]) -> str | None:
    if not buckets:
        return None
    for bucket in buckets:
        if bucket.matches(length):
            return bucket.name
    return None


def _normalize_question(raw_input: str, all_classes: list[str] | None) -> str:
    question = raw_input.strip()
    if all_classes:
        options = "\n".join(f"- {item}" for item in all_classes)
        return f"{question}\n\nOptions:\n{options}"
    return question


def _normalize_row(
    *,
    row: dict[str, Any],
    task: LongBenchTaskConfig,
    split: str,
    buckets: tuple[LongBenchLengthBucket, ...],
    sample_index: int,
) -> dict[str, Any]:
    answers = [str(item).strip() for item in row.get("answers", []) if str(item).strip()]
    all_classes = [str(item).strip() for item in (row.get("all_classes") or []) if str(item).strip()]
    length_value = row.get("length")
    normalized_length = None if length_value is None else int(length_value)
    source_name = str(row.get("dataset") or task.source_name)
    sample_id = str(row.get("_id") or f"{task.source_name}_{sample_index:05d}")
    bucket_name = _bucket_name(normalized_length, buckets)

    metadata = {
        "dataset": source_name,
        "language": row.get("language"),
        "length": normalized_length,
        "all_classes": all_classes,
        "longbench_source_name": task.source_name,
        "longbench_split": split,
        "length_bucket": bucket_name,
    }

    return {
        "sample_id": sample_id,
        "split": split,
        "task_type": task.task_type,
        "question": _normalize_question(str(row["input"]), all_classes or None),
        "context": str(row["context"]),
        "expected_answer_type": task.expected_answer_type,
        "reference_answer": answers[0] if answers else None,
        "reference_answers": answers,
        "metadata": metadata,
    }


def import_longbench_pack(
    *,
    spec: BenchmarkSpec,
    config: LongBenchImportConfig,
    replace: bool,
) -> dict[str, Any]:
    if spec.id != config.benchmark_id:
        raise ValueError(
            f"LongBench import config benchmark_id={config.benchmark_id} does not match spec id={spec.id}."
        )

    selected_tasks = [task for task in config.tasks if task.include]
    if not selected_tasks:
        raise ValueError("LongBench import config does not include any enabled tasks.")

    normalized_records: list[dict[str, Any]] = []
    imported_task_counts: dict[str, int] = {}
    for task in selected_tasks:
        rows = _load_hf_rows(config.dataset_id, task.source_name, config.split)
        count = 0
        for index, row in enumerate(rows):
            if str(row.get("language", "")).lower() != config.language.lower():
                continue
            normalized = _normalize_row(
                row=row,
                task=task,
                split=config.split,
                buckets=config.length_buckets,
                sample_index=index,
            )
            if normalized["metadata"]["length_bucket"] is None and config.length_buckets:
                continue
            normalized_records.append(normalized)
            count += 1
            per_task_limit = task.max_samples if task.max_samples is not None else config.max_samples_per_task
            if per_task_limit is not None and count >= per_task_limit:
                break
        imported_task_counts[task.source_name] = count

    artifact = write_normalized_benchmark_pack(
        spec=spec,
        pack_id=config.pack_id,
        description=config.description,
        normalized_records=normalized_records,
        default_split=config.split,
        source_label=f"{config.dataset_id}:{config.split}",
        default_task_type="long_context_task",
        default_expected_answer_type="short_text",
        replace=replace,
    )
    artifact["imported_tasks"] = imported_task_counts
    artifact["length_buckets"] = [
        {
            "name": bucket.name,
            "min_length": bucket.min_length,
            "max_length": bucket.max_length,
        }
        for bucket in config.length_buckets
    ]
    return artifact
