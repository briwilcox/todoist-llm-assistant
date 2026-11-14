"""Tests for the model registry and per-task model resolution."""

from __future__ import annotations

import pytest

from todoist_assistant import models as m


def test_openai_reasoning_model_capabilities() -> None:
    spec = m.get_model_spec("gpt-5.5")
    assert spec.provider is m.Provider.OPENAI
    assert spec.is_reasoning is True
    assert spec.supports_temperature is False
    assert spec.supports_reasoning_effort is True
    assert spec.token_param == "max_completion_tokens"


def test_openai_standard_model_capabilities() -> None:
    spec = m.get_model_spec("gpt-5.4-mini")
    assert spec.provider is m.Provider.OPENAI
    assert spec.is_reasoning is False
    assert spec.supports_temperature is True
    assert spec.supports_reasoning_effort is False


def test_anthropic_model_capabilities() -> None:
    spec = m.get_model_spec("claude-opus-4-8")
    assert spec.provider is m.Provider.ANTHROPIC
    assert spec.token_param == "max_tokens"
    assert spec.supports_temperature is True
    assert spec.supports_reasoning_effort is False


def test_haiku_alias_resolves_to_canonical_id() -> None:
    spec = m.get_model_spec("claude-haiku-4-5")
    assert spec.id == "claude-haiku-4-5-20251001"
    # canonical id resolves to the same spec
    assert m.get_model_spec("claude-haiku-4-5-20251001") is spec


def test_unknown_model_raises() -> None:
    with pytest.raises(m.UnknownModelError, match="o1"):
        m.get_model_spec("o1")


def test_is_registered() -> None:
    assert m.is_registered("gpt-5.5")
    assert m.is_registered("claude-haiku-4-5")  # alias
    assert not m.is_registered("gpt-4o")


def test_available_models_lists_selectable_names() -> None:
    available = m.available_models()
    assert "gpt-5.5" in available
    assert "claude-opus-4-8" in available
    assert "claude-haiku-4-5" in available  # friendly alias is selectable


def test_resolve_cli_takes_precedence() -> None:
    chosen = m.resolve_model(
        cli_model="gpt-5.5",
        task_project_id="p1",
        project_models={"p1": "claude-opus-4-8"},
        default_model="gpt-5.4-mini",
    )
    assert chosen == "gpt-5.5"


def test_resolve_uses_project_model() -> None:
    chosen = m.resolve_model(
        cli_model=None,
        task_project_id="p1",
        project_models={"p1": "claude-opus-4-8"},
        default_model="gpt-5.4-mini",
    )
    assert chosen == "claude-opus-4-8"


def test_resolve_falls_back_to_default() -> None:
    chosen = m.resolve_model(
        cli_model=None,
        task_project_id="p2",
        project_models={"p1": "claude-opus-4-8"},
        default_model="gpt-5.4-mini",
    )
    assert chosen == "gpt-5.4-mini"


def test_resolve_with_no_cli_and_no_project_returns_default() -> None:
    chosen = m.resolve_model(
        cli_model=None,
        task_project_id=None,
        project_models={},
        default_model="gpt-5.4-mini",
    )
    assert chosen == "gpt-5.4-mini"


def test_resolve_ignores_invalid_project_model() -> None:
    chosen = m.resolve_model(
        cli_model=None,
        task_project_id="p1",
        project_models={"p1": "totally-bogus"},
        default_model="gpt-5.4-mini",
    )
    assert chosen == "gpt-5.4-mini"
