"""The providers against the installed OpenAI and Anthropic SDKs, with only HTTP faked.

These tests catch what the client doubles in ``tests.factories`` cannot: a keyword the
installed SDK no longer accepts, and request bodies the current APIs reject.
"""

from __future__ import annotations

from anthropic import Anthropic
from openai import OpenAI

from todoist_assistant.models import get_model_spec
from todoist_assistant.providers.anthropic_provider import AnthropicProvider
from todoist_assistant.providers.base import CompletionRequest
from todoist_assistant.providers.openai_provider import OpenAIProvider

from .http_fakes import FakeLLMServer

FAKE_KEY = "test-key-not-real"


def request(effort: str, *, temperature: float = 0.4) -> CompletionRequest:
    return CompletionRequest(
        system="You plan tasks.",
        messages=[{"role": "user", "content": "Pay the water bill"}],
        temperature=temperature,
        reasoning_effort=effort,
    )


def anthropic_provider(server: FakeLLMServer) -> AnthropicProvider:
    return AnthropicProvider(
        Anthropic(api_key=FAKE_KEY, http_client=server.anthropic_client(), max_retries=0)
    )


def openai_provider(server: FakeLLMServer) -> OpenAIProvider:
    return OpenAIProvider(
        OpenAI(api_key=FAKE_KEY, http_client=server.openai_client(), max_retries=0)
    )


def test_anthropic_request_uses_effort_and_omits_sampling_parameters() -> None:
    server = FakeLLMServer(anthropic_reply="Book the appointment for Tuesday.")

    text = anthropic_provider(server).complete(get_model_spec("claude-haiku-5-5"), request("low"))

    # The reply opens with a thinking block; only the text block is the suggestion.
    assert text == "Book the appointment for Tuesday."
    (body,) = server.bodies("/v1/messages")
    assert body["model"] == "claude-haiku-5-5"
    assert body["output_config"] == {"effort": "low"}
    assert body["system"] == "You plan tasks."
    assert body["messages"] == [{"role": "user", "content": "Pay the water bill"}]
    # API version 2026-10-21 rejects temperature, top_p and top_k with a 400.
    assert not {"temperature", "top_p", "top_k", "thinking"} & body.keys()
    # max_tokens covers thinking plus text, so the default leaves room for both.
    assert body["max_tokens"] >= 16000


def test_openai_reasoning_request_omits_temperature() -> None:
    server = FakeLLMServer(openai_reply="Open the bill and pay it online.")

    text = openai_provider(server).complete(get_model_spec("gpt-6-luna"), request("low"))

    assert text == "Open the bill and pay it online."
    (body,) = server.bodies("/chat/completions")
    assert body["model"] == "gpt-6-luna"
    assert body["reasoning_effort"] == "low"
    assert "temperature" not in body
    assert body["messages"][0] == {"role": "developer", "content": "You plan tasks."}
    assert body["messages"][1] == {"role": "user", "content": "Pay the water bill"}


def test_openai_effort_none_sends_temperature() -> None:
    server = FakeLLMServer()

    openai_provider(server).complete(get_model_spec("gpt-6-luna"), request("none"))

    (body,) = server.bodies("/chat/completions")
    assert body["reasoning_effort"] == "none"
    assert body["temperature"] == 0.4
