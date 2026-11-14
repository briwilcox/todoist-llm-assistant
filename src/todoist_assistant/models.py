"""Registry of supported AI models and how each one must be called.

This is the single source of truth for which models exist. It is consumed by the CLI
(choices and validation), provider routing, and the suggestion engine. To add a model,
add one :class:`ModelSpec` to ``_SPECS``.

Every current OpenAI and Anthropic model reasons before it answers, and the effort
parameter is the control for how much. What still differs by model is which effort
levels it accepts and whether sampling parameters such as ``temperature`` are allowed:

* OpenAI GPT-6 family (developers.openai.com/api/docs/models): ``gpt-6-luna`` and
  ``gpt-6.1-terra`` accept effort ``none``, and only at ``none`` do they take
  ``temperature``. ``gpt-6-astra`` and ``gpt-6.1-sol`` reject ``none``.
* Anthropic Claude 5 family (platform.claude.com/docs/en/about-claude/models/overview):
  all four accept ``low`` through ``max``. API version 2026-10-21 removed
  ``temperature``, ``top_p`` and ``top_k`` for every model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class UnknownModelError(Exception):
    """Raised when a model id is not in the registry."""


# Every effort level any registered model accepts, in the order the CLI offers them.
EFFORT_LEVELS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "medium"

_LOW_TO_MAX: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
_NONE_TO_MAX: tuple[str, ...] = ("none", *_LOW_TO_MAX)


@dataclass(frozen=True)
class ModelSpec:
    """How to call a particular model.

    Attributes:
        id: The model id passed to the provider's API.
        provider: Which provider serves the model.
        effort_levels: The reasoning effort levels the model accepts.
        sampling_effort: The one effort level at which the model accepts
            ``temperature``, or None if it never does.
    """

    id: str
    provider: Provider
    effort_levels: tuple[str, ...] = _LOW_TO_MAX
    sampling_effort: str | None = None

    def effort_error(self, effort: str) -> str | None:
        """A user-facing message if ``effort`` is not accepted, else None."""
        if effort in self.effort_levels:
            return None
        return (
            f"Model '{self.id}' does not support reasoning effort '{effort}' "
            f"(supported: {', '.join(self.effort_levels)})."
        )

    def accepts_temperature(self, effort: str) -> bool:
        """Whether a request at ``effort`` may include ``temperature``."""
        return self.sampling_effort is not None and effort == self.sampling_effort


_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("gpt-6-astra", Provider.OPENAI),
    ModelSpec("gpt-6.1-sol", Provider.OPENAI),
    ModelSpec("gpt-6.1-terra", Provider.OPENAI, _NONE_TO_MAX, sampling_effort="none"),
    ModelSpec("gpt-6-luna", Provider.OPENAI, _NONE_TO_MAX, sampling_effort="none"),
    ModelSpec("claude-fable-5-2", Provider.ANTHROPIC),
    ModelSpec("claude-opus-5-5", Provider.ANTHROPIC),
    ModelSpec("claude-sonnet-5-5", Provider.ANTHROPIC),
    ModelSpec("claude-haiku-5-5", Provider.ANTHROPIC),
)

_REGISTRY: dict[str, ModelSpec] = {spec.id: spec for spec in _SPECS}


def get_model_spec(model: str) -> ModelSpec:
    """Resolve a model id to its :class:`ModelSpec`."""
    try:
        return _REGISTRY[model]
    except KeyError:
        raise UnknownModelError(
            f"Unknown model {model!r}. Known models: {', '.join(available_models())}"
        ) from None


def is_registered(model: str) -> bool:
    """True if ``model`` is a known model id."""
    return model in _REGISTRY


def available_models() -> list[str]:
    """Selectable model ids, sorted."""
    return sorted(_REGISTRY)


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
