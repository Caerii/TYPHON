from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BudgetLedger:
    training_flops: int | None = None
    inference_flops: int | None = None
    proxy_token_ops: int | None = None
    active_memory_units: int | None = None
    latency_ms: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_flops": self.training_flops,
            "inference_flops": self.inference_flops,
            "proxy_token_ops": self.proxy_token_ops,
            "active_memory_units": self.active_memory_units,
            "latency_ms": self.latency_ms,
            "notes": self.notes,
        }
