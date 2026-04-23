from __future__ import annotations

from typhon.inference.base import InferenceBackend
from typhon.inference.extractive import ExtractiveHeuristicBackend
from typhon.inference.lmstudio_local import LMStudioLocalBackend
from typhon.inference.ollama_local import OllamaLocalBackend
from typhon.inference.openai_compatible import OpenAICompatibleBackend


def available_backend_ids() -> list[str]:
    return ["extractive_heuristic", "ollama_local", "lmstudio_local", "openai_compatible_http"]


def create_backend(
    backend_id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> InferenceBackend:
    if backend_id == "extractive_heuristic":
        return ExtractiveHeuristicBackend()
    if backend_id == "ollama_local":
        return OllamaLocalBackend(base_url=base_url or "http://localhost:11434")
    if backend_id == "lmstudio_local":
        return LMStudioLocalBackend(
            base_url=base_url or "http://localhost:1234/v1",
            api_key=api_key,
        )
    if backend_id == "openai_compatible_http":
        return OpenAICompatibleBackend(
            base_url=base_url or "http://localhost:8000/v1",
            api_key=api_key,
        )
    raise KeyError(f"Unknown inference backend: {backend_id}")
