"""OpenAI Chat Completions provider."""

from __future__ import annotations

from typing import Any

from ..models import ModelSpec
from .base import CompletionRequest, ProviderError, SuggestionProvider


class OpenAIProvider(SuggestionProvider):
    """Calls OpenAI via an injected ``openai.OpenAI``-compatible client.

    Capability flags on the :class:`ModelSpec` decide the request shape:
    reasoning models omit ``temperature`` and add ``reasoning_effort``; the
    output-token budget uses the model's ``token_param`` key.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(self, spec: ModelSpec, request: CompletionRequest) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": request.system}]
        messages.extend(request.messages)

        params: dict[str, Any] = {"model": spec.id, "messages": messages}
        if request.max_tokens is not None:
            params[spec.token_param] = request.max_tokens
        if spec.supports_temperature:
            params["temperature"] = request.temperature
        if spec.supports_reasoning_effort:
            params["reasoning_effort"] = request.reasoning_effort

        try:
            response = self._client.chat.completions.create(**params)
        except Exception as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        content = response.choices[0].message.content
        return (content or "").strip()
