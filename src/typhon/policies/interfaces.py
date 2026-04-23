from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class WriteSignal:
    surprise: float
    gradient_norm: float
    novelty: float
    predicted_utility: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WriteDecision:
    action: str
    target_layer: str | None
    score: float
    reason: str


class WritePolicy(Protocol):
    def decide(self, signal: WriteSignal) -> WriteDecision:
        ...
