"""Tests for the OpenAI and Anthropic providers, against client doubles."""

from __future__ import annotations

import pytest

from todoist_assistant.models import get_model_spec
from todoist_assistant.providers.anthropic_provider import DEFAULT_MAX_TOKENS, AnthropicProvider
from todoist_assistant.providers.base import CompletionRequest, ProviderError
from todoist_assistant.providers.openai_provider import OpenAIProvider

from .factories import FakeAnthropicClient, FakeOpenAIClient


def req(
    *, max_tokens: int | None = None, temperature: float = 0.7, effort: str = "medium"
) -> CompletionRequest:
    return CompletionRequest(
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=effort,
    )


# --- OpenAI -----------------------------------------------------------------


def test_openai_sends_effort_budget_and_developer_instructions() -> None:
    client = FakeOpenAIClient(content="Plan it out.")

    out = OpenAIProvider(client).complete(
        get_model_spec("gpt-6-astra"), req(max_tokens=200, effort="high")
    )

    assert out == "Plan it out."
    call = client.last_call
    assert call["model"] == "gpt-6-astra"
    assert call["reasoning_effort"] == "high"
    assert call["max_completion_tokens"] == 200
    assert "temperature" not in call
    assert call["messages"] == [
        {"role": "developer", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_openai_temperature_only_at_effort_none() -> None:
    client = FakeOpenAIClient()
    provider = OpenAIProvider(client)

    provider.complete(get_model_spec("gpt-6-luna"), req(temperature=0.9, effort="none"))
    assert client.last_call["temperature"] == 0.9

    provider.complete(get_model_spec("gpt-6-luna"), req(temperature=0.9, effort="low"))
    assert "temperature" not in client.last_call


def test_openai_omits_token_budget_when_unset() -> None:
    client = FakeOpenAIClient()
    OpenAIProvider(client).complete(get_model_spec("gpt-6.1-sol"), req(max_tokens=None))
    assert "max_completion_tokens" not in client.last_call
    assert "max_tokens" not in client.last_call


def test_openai_strips_and_handles_empty_content() -> None:
    client = FakeOpenAIClient(content="  spaced  ")
    assert OpenAIProvider(client).complete(get_model_spec("gpt-6-luna"), req()) == "spaced"

    empty = FakeOpenAIClient(content=None)
    assert OpenAIProvider(empty).complete(get_model_spec("gpt-6-luna"), req()) == ""


def test_openai_wraps_sdk_errors() -> None:
    client = FakeOpenAIClient(error=RuntimeError("boom"))
    with pytest.raises(ProviderError, match="OpenAI request failed: boom"):
        OpenAIProvider(client).complete(get_model_spec("gpt-6-luna"), req())


# --- Anthropic --------------------------------------------------------------


def test_anthropic_passes_system_effort_and_joins_text_blocks_only() -> None:
    client = FakeAnthropicClient(texts=["Part 1. ", "Part 2."], thinking_first=True)

    out = AnthropicProvider(client).complete(
        get_model_spec("claude-opus-5-5"), req(max_tokens=500, effort="high")
    )

    assert out == "Part 1. Part 2."
    call = client.last_call
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["max_tokens"] == 500
    assert call["output_config"] == {"effort": "high"}


def test_anthropic_defaults_max_tokens_with_room_for_thinking() -> None:
    client = FakeAnthropicClient()
    AnthropicProvider(client).complete(get_model_spec("claude-haiku-5-5"), req(max_tokens=None))
    assert client.last_call["max_tokens"] == DEFAULT_MAX_TOKENS == 16000


def test_anthropic_never_sends_temperature() -> None:
    client = FakeAnthropicClient()
    AnthropicProvider(client).complete(get_model_spec("claude-sonnet-5-5"), req(temperature=0.2))
    assert "temperature" not in client.last_call


def test_anthropic_wraps_sdk_errors() -> None:
    client = FakeAnthropicClient(error=RuntimeError("nope"))
    with pytest.raises(ProviderError, match="Anthropic request failed: nope"):
        AnthropicProvider(client).complete(get_model_spec("claude-opus-5-5"), req())
