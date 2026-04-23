from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class MemoryWriteRequest:
    layer: str
    content: str
    utility_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryReadRequest:
    layer: str
    query: str
    top_k: int = 3


class MemoryLayer(Protocol):
    def write(self, request: MemoryWriteRequest) -> None:
        ...

    def read(self, request: MemoryReadRequest) -> list[str]:
        ...

    def reset(self) -> None:
        ...
