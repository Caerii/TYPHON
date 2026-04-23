from __future__ import annotations

import json
from pathlib import Path

from typhon.baselines.base import BaselineSpec
from typhon.utils.paths import repo_root


class BaselineRegistry:
    def __init__(self, specs: dict[str, BaselineSpec], config_dir: Path) -> None:
        self.specs = specs
        self.config_dir = config_dir

    @classmethod
    def load(cls) -> "BaselineRegistry":
        config_dir = repo_root() / "configs" / "baselines"
        specs: dict[str, BaselineSpec] = {}
        for path in sorted(config_dir.glob("*.json")):
            if path.name == "reproduction_matrix.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            spec = BaselineSpec.from_dict(payload)
            specs[spec.id] = spec
        return cls(specs=specs, config_dir=config_dir)

    def list_baselines(self) -> list[BaselineSpec]:
        return sorted(self.specs.values(), key=lambda item: item.id)

    def get(self, baseline_id: str) -> BaselineSpec:
        try:
            return self.specs[baseline_id]
        except KeyError as exc:
            raise KeyError(f"Unknown baseline id: {baseline_id}") from exc
