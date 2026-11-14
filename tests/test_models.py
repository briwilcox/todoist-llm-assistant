"""Tests for the model registry and per-task model resolution."""

from __future__ import annotations

import pytest

from todoist_assistant import models as m

OPENAI_IDS = {"gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna"}
ANTHROPIC_IDS = {"claude-fable-5-2", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"}


def test_registry_lists_current_models_by_provider() -> None:
    assert set(m.available_models()) == OPENAI_IDS | ANTHROPIC_IDS
    for model_id in OPENAI_IDS:
        assert m.get_model_spec(model_id).provider is m.Provider.OPENAI
    for model_id in ANTHROPIC_IDS:
        assert m.get_model_spec(model_id).provider is m.Provider.ANTHROPIC


def test_retired_generation_ids_are_rejected() -> None:
    for old in ("gpt-5.4-mini", "gpt-5.5", "claude-opus-4-8", "claude-haiku-4-5"):
        assert not m.is_registered(old)


@pytest.mark.parametrize("model_id", sorted(OPENAI_IDS | ANTHROPIC_IDS))
def test_every_model_accepts_low_through_max(model_id: str) -> None:
    spec = m.get_model_spec(model_id)
    for effort in ("low", "medium", "high", "xhigh", "max"):
        assert spec.effort_error(effort) is None


@pytest.mark.parametrize("model_id", ["gpt-6-luna", "gpt-6.1-terra"])
def test_small_openai_models_accept_effort_none_with_temperature(model_id: str) -> None:
    spec = m.get_model_spec(model_id)
    assert spec.effort_error("none") is None
    assert spec.accepts_temperature("none")
    assert not spec.accepts_temperature("low")


@pytest.mark.parametrize("model_id", ["gpt-6-astra", "gpt-6.1-sol", *sorted(ANTHROPIC_IDS)])
def test_models_without_effort_none_reject_it_and_never_take_temperature(model_id: str) -> None:
    spec = m.get_model_spec(model_id)
    error = spec.effort_error("none")
    assert error is not None
    assert model_id in error
    assert "low, medium, high, xhigh, max" in error
    for effort in m.EFFORT_LEVELS:
        assert not spec.accepts_temperature(effort)


def test_effort_levels_are_ordered_for_the_cli() -> None:
    assert m.EFFORT_LEVELS == ("none", "low", "medium", "high", "xhigh", "max")
    assert m.DEFAULT_EFFORT == "medium"


def test_unknown_model_raises_with_known_list() -> None:
    with pytest.raises(m.UnknownModelError, match=r"'o1'.*gpt-6-luna"):
        m.get_model_spec("o1")


def test_is_registered() -> None:
    assert m.is_registered("claude-sonnet-5-5")
    assert not m.is_registered("gpt-4o")


def test_resolve_cli_takes_precedence() -> None:
    chosen = m.resolve_model(
        cli_model="gpt-6-astra",
        task_project_id="p1",
        project_models={"p1": "claude-opus-5-5"},
        default_model="gpt-6-luna",
    )
    assert chosen == "gpt-6-astra"


def test_resolve_uses_project_model() -> None:
    chosen = m.resolve_model(
        cli_model=None,
        task_project_id="p1",
        project_models={"p1": "claude-opus-5-5"},
        default_model="gpt-6-luna",
    )
    assert chosen == "claude-opus-5-5"


def test_resolve_falls_back_to_default() -> None:
    chosen = m.resolve_model(
        cli_model=None,
        task_project_id="p2",
        project_models={"p1": "claude-opus-5-5"},
        default_model="gpt-6-luna",
    )
    assert chosen == "gpt-6-luna"


def test_resolve_with_no_cli_and_no_project_returns_default() -> None:
    chosen = m.resolve_model(
        cli_model=None,
        task_project_id=None,
        project_models={},
        default_model="gpt-6-luna",
    )
    assert chosen == "gpt-6-luna"


def test_resolve_ignores_invalid_project_model() -> None:
    chosen = m.resolve_model(
        cli_model=None,
        task_project_id="p1",
        project_models={"p1": "claude-opus-4-8"},
        default_model="gpt-6-luna",
    )
    assert chosen == "gpt-6-luna"
