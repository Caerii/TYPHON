from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from typhon.memory.interfaces import MemoryReadRequest, MemoryWriteRequest
from typhon.utils.text import significant_terms


@dataclass
class MemoryRecord:
    layer: str
    content: str
    utility_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "content_preview": self.content[:220],
            "utility_score": round(self.utility_score, 4),
            "metadata": self.metadata,
        }


class SimpleMemoryStore:
    def __init__(self, layer_name: str, capacity: int) -> None:
        self.layer_name = layer_name
        self.capacity = capacity
        self._records: list[MemoryRecord] = []

    def write(self, request: MemoryWriteRequest) -> None:
        record = MemoryRecord(
            layer=request.layer,
            content=request.content,
            utility_score=request.utility_score,
            metadata=dict(request.metadata),
        )
        self._records.append(record)
        self._records.sort(key=lambda item: item.utility_score, reverse=True)
        self._records = self._records[: self.capacity]

    def read(self, request: MemoryReadRequest) -> list[str]:
        query_terms = set(significant_terms(request.query))
        ranked = sorted(
            self._records,
            key=lambda item: (
                len(query_terms.intersection(significant_terms(item.content))),
                item.utility_score,
            ),
            reverse=True,
        )
        return [item.content for item in ranked[: request.top_k]]

    def top_records(self, top_k: int = 5) -> list[MemoryRecord]:
        return self._records[:top_k]

    def reset(self) -> None:
        self._records.clear()

    @property
    def record_count(self) -> int:
        return len(self._records)
