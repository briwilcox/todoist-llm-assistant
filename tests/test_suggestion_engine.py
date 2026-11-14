"""Tests for the suggestion engine (prompting, dispatch, fallback, formatting)."""

from __future__ import annotations

import pytest

from todoist_assistant.models import Provider
from todoist_assistant.providers.base import ProviderError
from todoist_assistant.suggestion_engine import Suggestion, SuggestionEngine

from .factories import FakeProvider


def make_engine(
    openai: FakeProvider | None = None,
    anthropic: FakeProvider | None = None,
    fallback_model: str = "gpt-6-luna",
) -> tuple[SuggestionEngine, FakeProvider, FakeProvider]:
    oa = openai or FakeProvider()
    an = anthropic or FakeProvider()
    engine = SuggestionEngine(
        {Provider.OPENAI: oa, Provider.ANTHROPIC: an},
        fallback_model=fallback_model,
    )
    return engine, oa, an


def test_formatted_suggestion_has_model_prefix() -> None:
    engine, _, _ = make_engine(openai=FakeProvider(default="Break it into steps."))
    result = engine.generate(task_content="Plan trip", model="gpt-6-luna")
    assert isinstance(result, Suggestion)
    assert result.text == "Break it into steps."
    assert result.formatted == "GPT-6-LUNA SUGGESTION:\nBreak it into steps."
    assert result.ok


def test_routes_anthropic_models_to_anthropic_provider() -> None:
    engine, oa, an = make_engine(anthropic=FakeProvider(default="Claude says hi"))
    result = engine.generate(task_content="x", model="claude-opus-5-5")
    assert result.text == "Claude says hi"
    assert len(an.requests) == 1
    assert len(oa.requests) == 0


def test_initial_prompt_includes_task_content() -> None:
    engine, oa, _ = make_engine()
    engine.generate(task_content="Renew passport", model="gpt-6.1-sol")
    user_msg = oa.last_request.messages[-1]
    assert user_msg["role"] == "user"
    assert "Renew passport" in user_msg["content"]


def test_conversation_history_is_passed_through() -> None:
    engine, oa, _ = make_engine()
    convo = [
        {"role": "user", "content": "do X"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "make it shorter"},
    ]
    engine.generate(task_content="X", model="gpt-6.1-sol", conversation=convo)
    assert oa.last_request.messages == convo


def test_passes_token_temperature_and_effort() -> None:
    engine, oa, _ = make_engine()
    engine.generate(
        task_content="x",
        model="gpt-6-astra",
        max_tokens=250,
        temperature=0.3,
        reasoning_effort="high",
    )
    r = oa.last_request
    assert r.max_tokens == 250
    assert r.temperature == 0.3
    assert r.reasoning_effort == "high"


def test_empty_response_without_fallback_reports_unable() -> None:
    engine, _, _ = make_engine(openai=FakeProvider(default=""))
    result = engine.generate(task_content="x", model="gpt-6.1-sol")
    assert result.text == ""
    assert not result.ok
    assert result.formatted == "Unable to generate suggestion using GPT-6.1-SOL"


def test_empty_response_with_fallback_uses_fallback_model() -> None:
    oa = FakeProvider(responses={"gpt-6-astra": "", "gpt-6-luna": "fallback wins"})
    engine, _, _ = make_engine(openai=oa)
    result = engine.generate(task_content="x", model="gpt-6-astra", enable_fallback=True)
    assert result.model == "gpt-6-luna"
    assert result.text == "fallback wins"


def test_provider_error_without_fallback_returns_error_suggestion() -> None:
    oa = FakeProvider(error=ProviderError("kaboom"))
    engine, _, _ = make_engine(openai=oa)
    result = engine.generate(task_content="x", model="gpt-6.1-sol")
    assert result.error == "kaboom"
    assert result.formatted == "Error: kaboom"
    assert not result.ok


def test_provider_error_with_fallback_recovers() -> None:
    # gpt-6-astra errors; fallback gpt-6-luna succeeds.
    class SelectiveProvider(FakeProvider):
        def complete(self, spec: object, request: object) -> str:
            self.requests.append((spec, request))
            if getattr(spec, "id", "") == "gpt-6-astra":
                raise ProviderError("primary down")
            return "recovered"

    engine, _, _ = make_engine(openai=SelectiveProvider())
    result = engine.generate(task_content="x", model="gpt-6-astra", enable_fallback=True)
    assert result.text == "recovered"
    assert result.model == "gpt-6-luna"


def test_unknown_model_raises() -> None:
    engine, _, _ = make_engine()
    with pytest.raises(Exception, match="o1"):
        engine.generate(task_content="x", model="o1")


def test_model_without_configured_provider_returns_error_not_crash() -> None:
    # Only OpenAI is configured; a Claude model chosen interactively or by a project
    # mapping must surface a readable error instead of a KeyError traceback.
    engine = SuggestionEngine({Provider.OPENAI: FakeProvider()}, fallback_model="gpt-6-luna")
    result = engine.generate(task_content="x", model="claude-sonnet-5-5")
    assert not result.ok
    assert result.formatted == (
        "Error: No API key configured for the 'anthropic' provider, "
        "which model 'claude-sonnet-5-5' requires."
    )


def test_model_without_configured_provider_falls_back_when_enabled() -> None:
    oa = FakeProvider(default="from the fallback")
    engine = SuggestionEngine({Provider.OPENAI: oa}, fallback_model="gpt-6-luna")
    result = engine.generate(task_content="x", model="claude-opus-5-5", enable_fallback=True)
    assert result.model == "gpt-6-luna"
    assert result.text == "from the fallback"


def test_unsupported_effort_is_reported_without_calling_the_api() -> None:
    engine, oa, _ = make_engine()
    result = engine.generate(task_content="x", model="gpt-6-astra", reasoning_effort="none")
    assert not result.ok
    assert result.error is not None
    assert "gpt-6-astra" in result.error
    assert "'none'" in result.error
    assert oa.requests == []
