from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from typhon.benchmarks.base import BenchmarkSample, BenchmarkSpec, SmokeFixture
from typhon.benchmarks.datasets import (
    dataset_status,
    fixture_to_sample,
    load_local_samples,
    validate_local_data,
)
from typhon.utils.paths import repo_root


@dataclass(frozen=True)
class BenchmarkRegistry:
    specs: dict[str, BenchmarkSpec]
    config_dir: Path
    fixture_dir: Path

    @classmethod
    def load(cls) -> "BenchmarkRegistry":
        root = repo_root()
        config_dir = root / "configs" / "benchmarks"
        fixture_dir = config_dir / "fixtures"
        specs: dict[str, BenchmarkSpec] = {}
        for path in sorted(config_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            spec = BenchmarkSpec.from_dict(payload)
            specs[spec.id] = spec
        return cls(specs=specs, config_dir=config_dir, fixture_dir=fixture_dir)

    def list_benchmarks(self, family: str | None = None) -> list[BenchmarkSpec]:
        specs = sorted(self.specs.values(), key=lambda item: item.id)
        if family is None:
            return specs
        return [spec for spec in specs if spec.family == family]

    def get(self, benchmark_id: str) -> BenchmarkSpec:
        try:
            return self.specs[benchmark_id]
        except KeyError as exc:
            raise KeyError(f"Unknown benchmark id: {benchmark_id}") from exc

    def load_fixture(self, spec: BenchmarkSpec) -> SmokeFixture:
        fixture_path = self.fixture_dir / spec.smoke_fixture
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        return SmokeFixture.from_dict(payload)

    def get_dataset_status(self, spec: BenchmarkSpec) -> dict[str, object]:
        fixture = self.load_fixture(spec)
        return dataset_status(spec, fixture).to_dict()

    def validate_local_data(self, spec: BenchmarkSpec) -> dict[str, object]:
        return validate_local_data(spec)

    def load_samples(
        self,
        spec: BenchmarkSpec,
        *,
        sample_source: str = "auto",
        limit: int | None = None,
    ) -> list[BenchmarkSample]:
        if sample_source not in {"auto", "local", "fixture"}:
            raise ValueError(f"Unsupported sample_source: {sample_source}")

        if sample_source in {"auto", "local"}:
            local_samples = load_local_samples(spec)
            if local_samples:
                return local_samples[:limit] if limit is not None else local_samples
            if sample_source == "local":
                return []

        fixture = self.load_fixture(spec)
        fallback = [fixture_to_sample(spec, fixture)]
        return fallback[:limit] if limit is not None else fallback
