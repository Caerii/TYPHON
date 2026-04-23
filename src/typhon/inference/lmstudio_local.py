from __future__ import annotations

from typhon.inference.openai_compatible import OpenAICompatibleBackend


class LMStudioLocalBackend(OpenAICompatibleBackend):
    backend_id = "lmstudio_local"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:1234/v1",
        api_key: str | None = None,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key)
