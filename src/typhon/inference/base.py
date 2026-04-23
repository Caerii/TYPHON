from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    system_prompt: str
    user_prompt: str
    benchmark_id: str | None = None
    question: str | None = None
    context: str | None = None
    expected_answer_type: str | None = None
    max_output_tokens: int = 128
    temperature: float = 0.0
    think: str | None = None
    request_timeout_seconds: float = 300.0


@dataclass(frozen=True)
class GenerationResult:
    content: str
    raw_response: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendStatus:
    backend_id: str
    available: bool
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "available": self.available,
            "details": self.details,
            "message": self.message,
        }


class InferenceBackend(Protocol):
    backend_id: str

    def status(self) -> BackendStatus:
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
