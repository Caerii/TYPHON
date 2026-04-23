from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from typhon.benchmarks.base import BenchmarkSpec
from typhon.utils.paths import repo_root

PACK_MANIFEST_FILENAME = "pack.json"
PACK_FORMAT_VERSION = 1


@dataclass(frozen=True)
class BenchmarkPackEntry:
    pack_id: str
    path: str
    format: str
    sample_count: int
    default_split: str = "local"
    description: str = ""
    source_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkPackEntry":
        return cls(
            pack_id=str(payload["pack_id"]),
            path=str(payload["path"]),
            format=str(payload.get("format", "jsonl")),
            sample_count=int(payload.get("sample_count", 0)),
            default_split=str(payload.get("default_split", "local")),
            description=str(payload.get("description", "")),
            source_label=payload.get("source_label"),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "path": self.path,
            "format": self.format,
            "sample_count": self.sample_count,
            "default_split": self.default_split,
            "description": self.description,
            "source_label": self.source_label,
            "metadata": self.metadata,
        }

    def resolve_path(self, benchmark_dir: Path) -> Path:
        return benchmark_dir / self.path


@dataclass(frozen=True)
class BenchmarkPackManifest:
    benchmark_id: str
    format_version: int
    preferred_source: str
    packs: tuple[BenchmarkPackEntry, ...]
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkPackManifest":
        return cls(
            benchmark_id=str(payload["benchmark_id"]),
            format_version=int(payload.get("format_version", PACK_FORMAT_VERSION)),
            preferred_source=str(payload.get("preferred_source", "local_pack")),
            packs=tuple(BenchmarkPackEntry.from_dict(item) for item in payload.get("packs", [])),
            notes=str(payload.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "format_version": self.format_version,
            "preferred_source": self.preferred_source,
            "packs": [entry.to_dict() for entry in self.packs],
            "notes": self.notes,
        }


def pack_manifest_path(spec: BenchmarkSpec) -> Path:
    return (repo_root() / "data" / "benchmarks" / spec.id) / PACK_MANIFEST_FILENAME


def load_pack_manifest(spec: BenchmarkSpec) -> BenchmarkPackManifest | None:
    path = pack_manifest_path(spec)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkPackManifest.from_dict(payload)


def save_pack_manifest(spec: BenchmarkSpec, manifest: BenchmarkPackManifest) -> Path:
    path = pack_manifest_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def resolve_pack_sample_paths(spec: BenchmarkSpec) -> list[tuple[BenchmarkPackEntry, Path]]:
    manifest = load_pack_manifest(spec)
    if manifest is None:
        return []
    benchmark_dir = pack_manifest_path(spec).parent
    return [(entry, entry.resolve_path(benchmark_dir)) for entry in manifest.packs]


def upsert_pack_entry(
    spec: BenchmarkSpec,
    *,
    pack_id: str,
    relative_path: str,
    sample_count: int,
    default_split: str,
    description: str,
    source_label: str | None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    manifest = load_pack_manifest(spec)
    entries = list(manifest.packs) if manifest is not None else []
    entries = [entry for entry in entries if entry.pack_id != pack_id]
    entries.append(
        BenchmarkPackEntry(
            pack_id=pack_id,
            path=relative_path,
            format=Path(relative_path).suffix.lstrip(".") or "jsonl",
            sample_count=sample_count,
            default_split=default_split,
            description=description,
            source_label=source_label,
            metadata=dict(metadata or {}),
        )
    )
    entries.sort(key=lambda item: item.pack_id)
    updated = BenchmarkPackManifest(
        benchmark_id=spec.id,
        format_version=PACK_FORMAT_VERSION,
        preferred_source="local_pack",
        packs=tuple(entries),
        notes=manifest.notes if manifest is not None else "Preferred local benchmark-pack manifest.",
    )
    return save_pack_manifest(spec, updated)
