"""Anthropic Messages provider."""

from __future__ import annotations

from typing import Any

from ..models import ModelSpec
from .base import CompletionRequest, ProviderError, SuggestionProvider

# Anthropic requires max_tokens on every request, and on the Claude 5 models it caps
# thinking plus reply text together. This default applies when the caller set no
# budget (the CLI's "no limit" default) and leaves room for both.
DEFAULT_MAX_TOKENS = 16000


class AnthropicProvider(SuggestionProvider):
    """Calls Anthropic via an injected ``anthropic.Anthropic``-compatible client.

    Thinking depth is set with ``output_config.effort``. Sampling parameters are
    never sent: API version 2026-10-21 rejects them for every model. Replies can
    open with thinking blocks, so text is read from the blocks of type ``text``.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(self, spec: ModelSpec, request: CompletionRequest) -> str:
        params: dict[str, Any] = {
            "model": spec.id,
            "system": request.system,
            "messages": request.messages,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
            "output_config": {"effort": request.reasoning_effort},
        }

        try:
            response = self._client.messages.create(**params)
        except Exception as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        return text.strip()
