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
    fallback_model: str = "gpt-5.4-mini",
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
    result = engine.generate(task_content="Plan trip", model="gpt-5.4-mini")
    assert isinstance(result, Suggestion)
    assert result.text == "Break it into steps."
    assert result.formatted == "GPT-5.4-MINI SUGGESTION:\nBreak it into steps."
    assert result.ok


def test_routes_anthropic_models_to_anthropic_provider() -> None:
    engine, oa, an = make_engine(anthropic=FakeProvider(default="Claude says hi"))
    result = engine.generate(task_content="x", model="claude-opus-4-8")
    assert result.text == "Claude says hi"
    assert len(an.requests) == 1
    assert len(oa.requests) == 0


def test_reasoning_and_standard_models_get_different_system_prompts() -> None:
    engine, oa, _ = make_engine()
    engine.generate(task_content="x", model="gpt-5.5")  # reasoning
    reasoning_system = oa.last_request.system
    engine.generate(task_content="x", model="gpt-5.4")  # standard
    standard_system = oa.last_request.system
    assert "step-by-step" in reasoning_system.lower()
    assert reasoning_system != standard_system


def test_initial_prompt_includes_task_content() -> None:
    engine, oa, _ = make_engine()
    engine.generate(task_content="Renew passport", model="gpt-5.4")
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
    engine.generate(task_content="X", model="gpt-5.4", conversation=convo)
    assert oa.last_request.messages == convo


def test_passes_token_temperature_and_effort() -> None:
    engine, oa, _ = make_engine()
    engine.generate(
        task_content="x",
        model="gpt-5.5",
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
    result = engine.generate(task_content="x", model="gpt-5.4")
    assert result.text == ""
    assert not result.ok
    assert result.formatted == "Unable to generate suggestion using GPT-5.4"


def test_empty_response_with_fallback_uses_fallback_model() -> None:
    oa = FakeProvider(responses={"gpt-5.5": "", "gpt-5.4-mini": "fallback wins"})
    engine, _, _ = make_engine(openai=oa)
    result = engine.generate(task_content="x", model="gpt-5.5", enable_fallback=True)
    assert result.model == "gpt-5.4-mini"
    assert result.text == "fallback wins"


def test_provider_error_without_fallback_returns_error_suggestion() -> None:
    oa = FakeProvider(error=ProviderError("kaboom"))
    engine, _, _ = make_engine(openai=oa)
    result = engine.generate(task_content="x", model="gpt-5.4")
    assert result.error == "kaboom"
    assert result.formatted == "Error: kaboom"
    assert not result.ok


def test_provider_error_with_fallback_recovers() -> None:
    # gpt-5.5 errors; fallback gpt-5.4-mini succeeds.
    class SelectiveProvider(FakeProvider):
        def complete(self, spec: object, request: object) -> str:
            self.requests.append((spec, request))
            if getattr(spec, "id", "") == "gpt-5.5":
                raise ProviderError("primary down")
            return "recovered"

    engine, _, _ = make_engine(openai=SelectiveProvider())
    result = engine.generate(task_content="x", model="gpt-5.5", enable_fallback=True)
    assert result.text == "recovered"
    assert result.model == "gpt-5.4-mini"


def test_unknown_model_raises() -> None:
    engine, _, _ = make_engine()
    with pytest.raises(Exception, match="o1"):
        engine.generate(task_content="x", model="o1")
