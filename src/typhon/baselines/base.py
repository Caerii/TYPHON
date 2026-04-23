from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BaselineSpec:
    id: str
    name: str
    description: str
    type: str
    retrieval_strategy: str
    uses_runtime_profile: bool
    supports_families: tuple[str, ...]
    local_window_policy: str
    max_chunks_to_retrieve: int
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineSpec":
        return cls(
            id=payload["id"],
            name=payload["name"],
            description=payload["description"],
            type=payload["type"],
            retrieval_strategy=payload["retrieval_strategy"],
            uses_runtime_profile=bool(payload.get("uses_runtime_profile", False)),
            supports_families=tuple(payload.get("supports_families", [])),
            local_window_policy=payload.get("local_window_policy", "fixed"),
            max_chunks_to_retrieve=int(payload.get("max_chunks_to_retrieve", 3)),
            settings=dict(payload.get("settings", {})),
        )
