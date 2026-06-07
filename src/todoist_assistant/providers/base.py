"""Provider abstraction: a uniform interface over OpenAI and Anthropic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..models import ModelSpec
from ..types import Message


class ProviderError(Exception):
    """Raised when an underlying provider SDK call fails."""


@dataclass
class CompletionRequest:
    """A provider-agnostic request for a single completion.

    ``system`` is kept separate from ``messages`` so each provider can place it
    where its API expects (OpenAI: a system-role message; Anthropic: the
    top-level ``system`` parameter).
    """

    system: str
    messages: list[Message] = field(default_factory=list)
    max_tokens: int | None = None
    temperature: float = 0.7
    reasoning_effort: str = "medium"


class SuggestionProvider(Protocol):
    """Turns a :class:`CompletionRequest` into a suggestion string."""

    def complete(self, spec: ModelSpec, request: CompletionRequest) -> str: ...
