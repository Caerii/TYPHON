from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from typhon.benchmarks.base import BenchmarkSample, BenchmarkSpec, SmokeFixture
from typhon.benchmarks.packs import load_pack_manifest, resolve_pack_sample_paths
from typhon.utils.paths import repo_root


@dataclass(frozen=True)
class DatasetStatus:
    benchmark_id: str
    source_mode: str
    has_local_data: bool
    sample_count: int
    paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_id": self.benchmark_id,
            "source_mode": self.source_mode,
            "has_local_data": self.has_local_data,
            "sample_count": self.sample_count,
            "paths": list(self.paths),
        }


def dataset_root() -> Path:
    return repo_root() / "data" / "benchmarks"


def dataset_dir(spec: BenchmarkSpec) -> Path:
    return dataset_root() / spec.id


def _candidate_files(spec: BenchmarkSpec) -> list[Path]:
    base = dataset_dir(spec)
    return [base / "samples.jsonl", base / "samples.json"]


def _load_jsonl_samples(path: Path) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        samples.append(json.loads(stripped))
    return samples


def _load_json_samples(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "samples" in payload and isinstance(payload["samples"], list):
        return payload["samples"]
    raise ValueError(f"Unsupported JSON sample format in {path}")


def _load_raw_samples(path: Path) -> list[dict[str, object]]:
    return _load_jsonl_samples(path) if path.suffix == ".jsonl" else _load_json_samples(path)


def local_data_paths(spec: BenchmarkSpec) -> list[Path]:
    manifest_paths = [path for _, path in resolve_pack_sample_paths(spec) if path.exists()]
    if manifest_paths:
        return manifest_paths
    return [path for path in _candidate_files(spec) if path.exists()]


def load_local_samples(spec: BenchmarkSpec) -> list[BenchmarkSample]:
    packed_paths = resolve_pack_sample_paths(spec)
    if packed_paths:
        samples: list[BenchmarkSample] = []
        for entry, path in packed_paths:
            raw_samples = _load_raw_samples(path)
            samples.extend(
                BenchmarkSample.from_dict(
                    payload=payload,
                    default_sample_id=f"{spec.id}_{entry.pack_id}_{index}",
                    default_split=entry.default_split,
                    source="local",
                    source_path=str(path),
                )
                for index, payload in enumerate(raw_samples)
            )
        return samples

    paths = local_data_paths(spec)
    if not paths:
        return []

    path = paths[0]
    raw_samples = _load_raw_samples(path)
    return [
        BenchmarkSample.from_dict(
            payload=payload,
            default_sample_id=f"{spec.id}_local_{index}",
            default_split="local",
            source="local",
            source_path=str(path),
        )
        for index, payload in enumerate(raw_samples)
    ]


def fixture_to_sample(spec: BenchmarkSpec, fixture: SmokeFixture) -> BenchmarkSample:
    return BenchmarkSample.from_dict(
        payload={
            "task_type": fixture.task_type,
            "question": fixture.question,
            "context": fixture.context,
            "expected_answer_type": fixture.expected_answer_type,
            "reference_answer": fixture.reference_answer,
            "reference_answers": list(fixture.reference_answers),
            "metadata": fixture.metadata,
        },
        default_sample_id=f"{spec.id}_fixture_0",
        default_split="fixture",
        source="fixture",
        source_path=None,
    )


def dataset_status(spec: BenchmarkSpec, fixture: SmokeFixture) -> DatasetStatus:
    manifest = load_pack_manifest(spec)
    paths = local_data_paths(spec)
    if paths:
        return DatasetStatus(
            benchmark_id=spec.id,
            source_mode="local_pack" if manifest is not None else "local",
            has_local_data=True,
            sample_count=len(load_local_samples(spec)),
            paths=tuple(str(path) for path in paths),
        )

    return DatasetStatus(
        benchmark_id=spec.id,
        source_mode="fixture",
        has_local_data=False,
        sample_count=1 if fixture else 0,
        paths=tuple(),
    )


def validate_local_data(spec: BenchmarkSpec) -> dict[str, object]:
    manifest = load_pack_manifest(spec)
    errors: list[str] = []
    warnings: list[str] = []
    paths = local_data_paths(spec)

    if manifest is not None and manifest.benchmark_id != spec.id:
        errors.append(
            f"Manifest benchmark_id {manifest.benchmark_id} does not match spec id {spec.id}."
        )

    if not paths:
        return {
            "benchmark_id": spec.id,
            "source_mode": "none",
            "is_valid": True,
            "sample_count": 0,
            "paths": [],
            "errors": errors,
            "warnings": warnings,
        }

    sample_count = 0
    packed_paths = resolve_pack_sample_paths(spec)
    if packed_paths:
        for entry, path in packed_paths:
            if not path.exists():
                errors.append(f"Missing pack file for {entry.pack_id}: {path}")
                continue
            try:
                raw_samples = _load_raw_samples(path)
            except Exception as exc:
                errors.append(f"Failed to parse {path}: {type(exc).__name__}: {exc}")
                continue
            sample_count += len(raw_samples)
            for index, payload in enumerate(raw_samples):
                try:
                    BenchmarkSample.from_dict(
                        payload=payload,
                        default_sample_id=f"{spec.id}_{entry.pack_id}_{index}",
                        default_split=entry.default_split,
                        source="local",
                        source_path=str(path),
                    )
                except Exception as exc:
                    errors.append(
                        f"Invalid sample in {path} at index {index}: {type(exc).__name__}: {exc}"
                    )
    else:
        path = paths[0]
        try:
            raw_samples = _load_raw_samples(path)
        except Exception as exc:
            errors.append(f"Failed to parse {path}: {type(exc).__name__}: {exc}")
            raw_samples = []
        sample_count = len(raw_samples)
        for index, payload in enumerate(raw_samples):
            try:
                BenchmarkSample.from_dict(
                    payload=payload,
                    default_sample_id=f"{spec.id}_local_{index}",
                    default_split="local",
                    source="local",
                    source_path=str(path),
                )
            except Exception as exc:
                errors.append(
                    f"Invalid sample in {path} at index {index}: {type(exc).__name__}: {exc}"
                )
        warnings.append("Using legacy local sample file without pack manifest.")

    return {
        "benchmark_id": spec.id,
        "source_mode": "local_pack" if manifest is not None else "local",
        "is_valid": not errors,
        "sample_count": sample_count,
        "paths": [str(path) for path in paths],
        "errors": errors,
        "warnings": warnings,
    }
