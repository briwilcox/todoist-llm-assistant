"""Anthropic Messages provider."""

from __future__ import annotations

from typing import Any

from ..models import ModelSpec
from .base import CompletionRequest, ProviderError, SuggestionProvider

# Anthropic requires max_tokens on every request; use this when the caller
# didn't specify a budget (the CLI's "no limit" default).
DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(SuggestionProvider):
    """Calls Anthropic via an injected ``anthropic.Anthropic``-compatible client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(self, spec: ModelSpec, request: CompletionRequest) -> str:
        params: dict[str, Any] = {
            "model": spec.id,
            "system": request.system,
            "messages": request.messages,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        }
        if spec.supports_temperature:
            # Anthropic's temperature range is 0.0-1.0; clamp the shared default.
            params["temperature"] = max(0.0, min(request.temperature, 1.0))

        try:
            response = self._client.messages.create(**params)
        except Exception as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "text") == "text"
        )
        return text.strip()
