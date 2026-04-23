from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from typhon.inference.base import BackendStatus, GenerationRequest, GenerationResult


class OpenAICompatibleBackend:
    backend_id = "openai_compatible_http"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        self.base_url = normalized
        self.api_key = api_key

    def _request(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
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
        try:
            response = self._request(method="GET", path="/models", timeout=3.0)
        except urllib.error.HTTPError as exc:
            return BackendStatus(
                backend_id=self.backend_id,
                available=False,
                details={
                    "base_url": self.base_url,
                    "http_status": exc.code,
                },
                message=f"OpenAI-compatible endpoint rejected the probe with HTTP {exc.code}.",
            )
        except urllib.error.URLError as exc:
            return BackendStatus(
                backend_id=self.backend_id,
                available=False,
                details={"base_url": self.base_url},
                message=f"OpenAI-compatible endpoint is not reachable: {exc.reason}",
            )

        models = response.get("data", [])
        return BackendStatus(
            backend_id=self.backend_id,
            available=True,
            details={
                "base_url": self.base_url,
                "models": [
                    {
                        "id": model.get("id"),
                        "owned_by": model.get("owned_by"),
                    }
                    for model in models
                ],
            },
            message="OpenAI-compatible endpoint is reachable.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "stream": False,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        response = self._request(
            method="POST",
            path="/chat/completions",
            payload=payload,
            timeout=request.request_timeout_seconds,
        )
        choices = response.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        usage = response.get("usage", {})
        return GenerationResult(
            content=message.get("content", "") or "",
            raw_response=response,
            usage={
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        )
