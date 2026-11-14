"""Registry of supported AI models and their API capabilities.

This is the single source of truth for which models exist and how each one must
be called. It replaces the original ``model.startswith('o')`` heuristic (which
broke when the o-series was retired) with explicit per-model capability flags,
and is consumed by the CLI (choices/validation), provider routing, and the
suggestion engine.

To add a model, add one :class:`ModelSpec` to ``_SPECS`` — nothing else needs to
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class UnknownModelError(Exception):
    """Raised when a model id is not in the registry."""


@dataclass(frozen=True)
class ModelSpec:
    """How to call a particular model.

    Attributes:
        id: The canonical model id passed to the provider's API.
        provider: Which provider serves the model.
        is_reasoning: Whether this is a reasoning model.
        supports_temperature: Whether the ``temperature`` parameter is accepted.
        supports_reasoning_effort: Whether a ``reasoning_effort`` parameter applies.
        token_param: The request key for the output-token budget
            (``max_tokens`` vs ``max_completion_tokens``).
    """

    id: str
    provider: Provider
    is_reasoning: bool = False
    supports_temperature: bool = True
    supports_reasoning_effort: bool = False
    token_param: str = "max_tokens"


_SPECS: tuple[ModelSpec, ...] = (
    # OpenAI GPT-5 family (Chat Completions; max_completion_tokens is the modern key).
    ModelSpec(
        "gpt-5.5",
        Provider.OPENAI,
        is_reasoning=True,
        supports_temperature=False,
        supports_reasoning_effort=True,
        token_param="max_completion_tokens",
    ),
    ModelSpec("gpt-5.4", Provider.OPENAI, token_param="max_completion_tokens"),
    ModelSpec("gpt-5.4-mini", Provider.OPENAI, token_param="max_completion_tokens"),
    ModelSpec("gpt-5.4-nano", Provider.OPENAI, token_param="max_completion_tokens"),
    # Anthropic Claude 4.x family (Messages API; max_tokens is required).
    ModelSpec("claude-opus-4-8", Provider.ANTHROPIC),
    ModelSpec("claude-sonnet-4-6", Provider.ANTHROPIC),
    ModelSpec("claude-haiku-4-5-20251001", Provider.ANTHROPIC),
)

_REGISTRY: dict[str, ModelSpec] = {spec.id: spec for spec in _SPECS}

# Friendly aliases users can type that map to a canonical model id.
_ALIASES: dict[str, str] = {
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
}


def get_model_spec(model: str) -> ModelSpec:
    """Resolve a model id (or alias) to its :class:`ModelSpec`."""
    canonical = _ALIASES.get(model, model)
    try:
        return _REGISTRY[canonical]
    except KeyError:
        raise UnknownModelError(
            f"Unknown model {model!r}. Known models: {', '.join(available_models())}"
        ) from None


def is_registered(model: str) -> bool:
    """True if ``model`` is a known model id or alias."""
    return model in _ALIASES or model in _REGISTRY


def available_models() -> list[str]:
    """Selectable model names (canonical ids plus friendly aliases), sorted."""
    return sorted([*_REGISTRY.keys(), *_ALIASES.keys()])


def resolve_model(
    *,
    cli_model: str | None,
    task_project_id: str | None,
    project_models: dict[str, str],
    default_model: str,
) -> str:
    """Pick the model for a task. Precedence: CLI > project mapping > default.

    A project mapping pointing at an unregistered model is ignored in favour of
    the default, mirroring the original lenient behaviour.
    """
    if cli_model:
        return cli_model
    if task_project_id is not None:
        project_choice = project_models.get(task_project_id)
        if project_choice and is_registered(project_choice):
            return project_choice
    return default_model
