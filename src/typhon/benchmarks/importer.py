from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from typhon.benchmarks.base import BenchmarkSpec
from typhon.benchmarks.packs import upsert_pack_entry
from typhon.utils.paths import repo_root


def _sanitize_pack_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return normalized.strip("._-") or "imported_pack"


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Each JSONL row must be an object: {path}")
            rows.append(payload)
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"Every JSON list item must be an object: {path}")
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        rows = payload["samples"]
        if not all(isinstance(item, dict) for item in rows):
            raise ValueError(f"Every samples[] item must be an object: {path}")
        return rows
    raise ValueError(f"Unsupported import format in {path}")


def _optional_value(payload: dict[str, Any], field_name: str | None) -> Any:
    if not field_name:
        return None
    return payload.get(field_name)


def _required_text(payload: dict[str, Any], field_name: str, *, label: str) -> str:
    value = payload.get(field_name)
    if value is None:
        raise KeyError(f"Missing required field '{field_name}' for {label}.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Required field '{field_name}' for {label} is empty.")
    return text


def _metadata_payload(
    payload: dict[str, Any],
    *,
    metadata_field: str | None,
    consumed_fields: set[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if metadata_field:
        value = payload.get(metadata_field)
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(f"Metadata field '{metadata_field}' must contain an object.")
            metadata.update(value)
            consumed_fields.add(metadata_field)
    elif isinstance(payload.get("metadata"), dict):
        metadata.update(payload["metadata"])
        consumed_fields.add("metadata")

    extras = {
        key: value
        for key, value in payload.items()
        if key not in consumed_fields and key != metadata_field
    }
    if extras:
        metadata["_import_extras"] = extras
    return metadata


def _normalize_records(
    rows: list[dict[str, Any]],
    *,
    pack_id: str,
    default_split: str,
    default_task_type: str,
    default_expected_answer_type: str,
    sample_id_field: str | None,
    split_field: str | None,
    task_type_field: str | None,
    question_field: str,
    context_field: str,
    expected_answer_type_field: str | None,
    reference_answer_field: str | None,
    metadata_field: str | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, payload in enumerate(rows):
        consumed_fields = {
            field_name
            for field_name in [
                sample_id_field,
                split_field,
                task_type_field,
                question_field,
                context_field,
                expected_answer_type_field,
                reference_answer_field,
            ]
            if field_name
        }
        sample_id = str(_optional_value(payload, sample_id_field) or f"{pack_id}_{index:04d}")
        split = str(_optional_value(payload, split_field) or default_split)
        task_type = str(_optional_value(payload, task_type_field) or default_task_type)
        expected_answer_type = str(
            _optional_value(payload, expected_answer_type_field) or default_expected_answer_type
        )
        question = _required_text(payload, question_field, label="question")
        context = _required_text(payload, context_field, label="context")
        reference_answer = _optional_value(payload, reference_answer_field)
        metadata = _metadata_payload(
            payload,
            metadata_field=metadata_field,
            consumed_fields=consumed_fields,
        )

        normalized.append(
            {
                "sample_id": sample_id,
                "split": split,
                "task_type": task_type,
                "question": question,
                "context": context,
                "expected_answer_type": expected_answer_type,
                "reference_answer": None if reference_answer is None else str(reference_answer),
                "metadata": metadata,
            }
        )
    return normalized


def write_normalized_benchmark_pack(
    *,
    spec: BenchmarkSpec,
    pack_id: str,
    description: str,
    normalized_records: list[dict[str, Any]],
    default_split: str,
    source_label: str,
    default_task_type: str,
    default_expected_answer_type: str,
    replace: bool,
) -> dict[str, Any]:
    pack_id = _sanitize_pack_id(pack_id)
    benchmark_dir = repo_root() / "data" / "benchmarks" / spec.id
    target_dir = benchmark_dir / "packs" / pack_id
    target_path = target_dir / "samples.jsonl"
    if target_path.exists() and not replace:
        raise FileExistsError(
            f"Pack already exists at {target_path}. Re-run with replace enabled or choose a new pack id."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in normalized_records) + "\n",
        encoding="utf-8",
    )

    manifest_path = upsert_pack_entry(
        spec,
        pack_id=pack_id,
        relative_path=str(target_path.relative_to(benchmark_dir)).replace("\\", "/"),
        sample_count=len(normalized_records),
        default_split=default_split,
        description=description,
        source_label=source_label,
        metadata={
            "importer": "typhon.import_benchmark_pack",
            "default_task_type": default_task_type,
            "default_expected_answer_type": default_expected_answer_type,
        },
    )

    return {
        "benchmark_id": spec.id,
        "pack_id": pack_id,
        "input_path": source_label,
        "output_path": str(target_path),
        "manifest_path": str(manifest_path),
        "sample_count": len(normalized_records),
        "default_split": default_split,
        "description": description,
    }


def import_benchmark_pack(
    *,
    spec: BenchmarkSpec,
    input_path: Path,
    pack_id: str,
    description: str,
    default_split: str,
    default_task_type: str,
    default_expected_answer_type: str,
    sample_id_field: str | None,
    split_field: str | None,
    task_type_field: str | None,
    question_field: str,
    context_field: str,
    expected_answer_type_field: str | None,
    reference_answer_field: str | None,
    metadata_field: str | None,
    replace: bool,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Import source does not exist: {input_path}")

    rows = _load_records(input_path)
    normalized = _normalize_records(
        rows,
        pack_id=pack_id,
        default_split=default_split,
        default_task_type=default_task_type,
        default_expected_answer_type=default_expected_answer_type,
        sample_id_field=sample_id_field,
        split_field=split_field,
        task_type_field=task_type_field,
        question_field=question_field,
        context_field=context_field,
        expected_answer_type_field=expected_answer_type_field,
        reference_answer_field=reference_answer_field,
        metadata_field=metadata_field,
    )
    return write_normalized_benchmark_pack(
        spec=spec,
        pack_id=pack_id,
        description=description,
        normalized_records=normalized,
        default_split=default_split,
        source_label=str(input_path),
        default_task_type=default_task_type,
        default_expected_answer_type=default_expected_answer_type,
        replace=replace,
    )
