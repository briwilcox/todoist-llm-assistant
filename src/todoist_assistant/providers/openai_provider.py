"""OpenAI Chat Completions provider."""

from __future__ import annotations

from typing import Any

from ..models import ModelSpec
from .base import CompletionRequest, ProviderError, SuggestionProvider


class OpenAIProvider(SuggestionProvider):
    """Calls OpenAI via an injected ``openai.OpenAI``-compatible client.

    Every registered OpenAI model is a reasoning model: instructions go in a
    ``developer`` message, the effort is always sent, the output budget is
    ``max_completion_tokens`` (it covers reasoning tokens too), and ``temperature``
    is sent only at the one effort level where the model accepts it.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(self, spec: ModelSpec, request: CompletionRequest) -> str:
        messages: list[dict[str, str]] = [{"role": "developer", "content": request.system}]
        messages.extend(request.messages)

        params: dict[str, Any] = {
            "model": spec.id,
            "messages": messages,
            "reasoning_effort": request.reasoning_effort,
        }
        if request.max_tokens is not None:
            params["max_completion_tokens"] = request.max_tokens
        if spec.accepts_temperature(request.reasoning_effort):
            params["temperature"] = request.temperature

        try:
            response = self._client.chat.completions.create(**params)
        except Exception as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        content = response.choices[0].message.content
        return (content or "").strip()
