"""Tests for the OpenAI and Anthropic providers."""

from __future__ import annotations

import pytest

from todoist_assistant.models import ModelSpec, Provider, get_model_spec
from todoist_assistant.providers.anthropic_provider import DEFAULT_MAX_TOKENS, AnthropicProvider
from todoist_assistant.providers.base import CompletionRequest, ProviderError
from todoist_assistant.providers.openai_provider import OpenAIProvider

from .factories import FakeAnthropicClient, FakeOpenAIClient


def req(**kw: object) -> CompletionRequest:
    base = {"system": "sys", "messages": [{"role": "user", "content": "hi"}]}
    base.update(kw)
    return CompletionRequest(**base)  # type: ignore[arg-type]


# --- OpenAI -----------------------------------------------------------------


def test_openai_reasoning_model_omits_temperature_adds_effort() -> None:
    client = FakeOpenAIClient(content="Plan it out.")
    provider = OpenAIProvider(client)

    out = provider.complete(get_model_spec("gpt-5.5"), req(max_tokens=200, reasoning_effort="high"))

    assert out == "Plan it out."
    call = client.last_call
    assert call["model"] == "gpt-5.5"
    assert "temperature" not in call
    assert call["reasoning_effort"] == "high"
    assert call["max_completion_tokens"] == 200
    assert call["messages"][0] == {"role": "system", "content": "sys"}
    assert call["messages"][1] == {"role": "user", "content": "hi"}


def test_openai_standard_model_includes_temperature_no_effort() -> None:
    client = FakeOpenAIClient()
    provider = OpenAIProvider(client)

    provider.complete(get_model_spec("gpt-5.4-mini"), req(max_tokens=300, temperature=0.9))

    call = client.last_call
    assert call["temperature"] == 0.9
    assert "reasoning_effort" not in call
    assert call["max_completion_tokens"] == 300


def test_openai_omits_token_param_when_no_budget() -> None:
    client = FakeOpenAIClient()
    OpenAIProvider(client).complete(get_model_spec("gpt-5.4"), req(max_tokens=None))
    assert "max_completion_tokens" not in client.last_call


def test_openai_strips_and_handles_empty_content() -> None:
    client = FakeOpenAIClient(content="  spaced  ")
    assert OpenAIProvider(client).complete(get_model_spec("gpt-5.4"), req()) == "spaced"

    empty = FakeOpenAIClient(content=None)
    assert OpenAIProvider(empty).complete(get_model_spec("gpt-5.4"), req()) == ""


def test_openai_wraps_sdk_errors() -> None:
    client = FakeOpenAIClient(error=RuntimeError("boom"))
    with pytest.raises(ProviderError, match="OpenAI request failed: boom"):
        OpenAIProvider(client).complete(get_model_spec("gpt-5.4"), req())


# --- Anthropic --------------------------------------------------------------


def test_anthropic_passes_system_separately_and_joins_text() -> None:
    client = FakeAnthropicClient(texts=["Part 1. ", "Part 2."])
    provider = AnthropicProvider(client)

    out = provider.complete(get_model_spec("claude-opus-4-8"), req(max_tokens=500))

    assert out == "Part 1. Part 2."
    call = client.last_call
    assert call["model"] == "claude-opus-4-8"
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["max_tokens"] == 500


def test_anthropic_defaults_max_tokens_when_unset() -> None:
    client = FakeAnthropicClient()
    AnthropicProvider(client).complete(get_model_spec("claude-haiku-4-5"), req(max_tokens=None))
    assert client.last_call["max_tokens"] == DEFAULT_MAX_TOKENS


def test_anthropic_clamps_temperature_to_one() -> None:
    client = FakeAnthropicClient()
    AnthropicProvider(client).complete(get_model_spec("claude-sonnet-4-6"), req(temperature=1.8))
    assert client.last_call["temperature"] == 1.0


def test_anthropic_omits_temperature_when_model_unsupported() -> None:
    spec = ModelSpec("claude-opus-4-8", Provider.ANTHROPIC, supports_temperature=False)
    client = FakeAnthropicClient()
    AnthropicProvider(client).complete(spec, req())
    assert "temperature" not in client.last_call


def test_anthropic_wraps_sdk_errors() -> None:
    client = FakeAnthropicClient(error=RuntimeError("nope"))
    with pytest.raises(ProviderError, match="Anthropic request failed: nope"):
        AnthropicProvider(client).complete(get_model_spec("claude-opus-4-8"), req())
