from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from typing import Any

from typhon.inference.base import BackendStatus, GenerationRequest, GenerationResult


class OllamaLocalBackend:
    backend_id = "ollama_local"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body) if body else {}

    def status(self) -> BackendStatus:
        cli_present = shutil.which("ollama") is not None
        try:
            tags = self._request(method="GET", path="/api/tags", timeout=2.0)
        except urllib.error.URLError as exc:
            return BackendStatus(
                backend_id=self.backend_id,
                available=False,
                details={
                    "cli_present": cli_present,
                    "base_url": self.base_url,
                    "installed_models": [],
                },
                message=f"Ollama server is not reachable: {exc.reason}",
            )

        models = tags.get("models", [])
        return BackendStatus(
            backend_id=self.backend_id,
            available=True,
            details={
                "cli_present": cli_present,
                "base_url": self.base_url,
                "installed_models": [
                    {
                        "name": model.get("name"),
                        "size": model.get("size"),
                        "details": model.get("details", {}),
                    }
                    for model in models
                ],
            },
            message="Ollama server is reachable.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        if request.think:
            payload["think"] = request.think

        response = self._request(
            method="POST",
            path="/api/chat",
            payload=payload,
            timeout=request.request_timeout_seconds,
        )
        message = response.get("message", {})
        return GenerationResult(
            content=message.get("content", ""),
            raw_response=response,
            usage={
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "total_duration": response.get("total_duration"),
                "load_duration": response.get("load_duration"),
                "eval_duration": response.get("eval_duration"),
            },
        )
