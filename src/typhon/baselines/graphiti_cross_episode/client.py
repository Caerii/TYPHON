"""StrictSchemaClient — schema-enforced structured outputs over an OpenAI-compatible API.

Graphiti's stock ``OpenAIGenericClient`` sends a ``json_schema`` *without* ``strict: true``,
so an OpenAI-compatible backend (e.g. Together.ai) treats it as best-effort: the model may
omit fields, which silently drops value/attribute edges and leaves episodes with entities
but no facts. This subclass routes schema'd calls through the OpenAI SDK's structured-output
``beta.chat.completions.parse`` path, which Together honors as strict constrained decoding,
and validates against the Pydantic model. Calls with no ``response_model`` fall back to
plain ``json_object`` mode (graphiti uses those rarely).

Defined only when graphiti-core is importable; otherwise ``StrictSchemaClient is None``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ._deps import OpenAIGenericClient, RateLimitError, RefusalError, openai

logger = logging.getLogger(__name__)


if OpenAIGenericClient is not None:

    class StrictSchemaClient(OpenAIGenericClient):  # type: ignore[misc,valid-type]
        """OpenAIGenericClient that enforces the response schema via strict structured outputs."""

        async def _generate_response(  # type: ignore[override]
            self,
            messages,
            response_model=None,
            max_tokens=None,
            model_size=None,
        ) -> dict[str, Any]:
            openai_messages = []
            for message in messages:
                message.content = self._clean_input(message.content)
                if message.role in ("user", "system"):
                    openai_messages.append({"role": message.role, "content": message.content})
            try:
                if response_model is not None:
                    completion = await self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=openai_messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format=response_model,
                    )
                    parsed_message = completion.choices[0].message
                    refusal = getattr(parsed_message, "refusal", None)
                    if refusal:
                        raise RefusalError(refusal)
                    parsed = getattr(parsed_message, "parsed", None)
                    if parsed is not None:
                        return parsed.model_dump()
                    return json.loads(parsed_message.content or "{}")
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content or "{}")
            except openai.RateLimitError as exc:  # type: ignore[union-attr]
                raise RateLimitError from exc
            except RefusalError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("StrictSchemaClient response error: %s", exc)
                raise

else:  # graphiti-core unavailable

    StrictSchemaClient = None  # type: ignore[assignment]
