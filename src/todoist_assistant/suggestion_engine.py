"""Builds prompts, dispatches to the right provider, and formats suggestions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .config import DEFAULT_MODEL
from .models import DEFAULT_EFFORT, Provider, get_model_spec
from .providers.base import (
    CompletionRequest,
    ProviderError,
    SuggestionProvider,
    missing_key_message,
)
from .types import Message

# Every registered model reasons before answering, so one system prompt serves all.
_SYSTEM = (
    "You are a task planning assistant that uses step-by-step reasoning to provide "
    "actionable suggestions. Your goal is to help users accomplish their tasks "
    "effectively. Adapt your response length to the task's complexity - be concise for "
    "simple tasks but provide more detail for complex ones. Focus on being specific and "
    "actionable."
)


def _initial_user_prompt(task_content: str) -> str:
    return (
        f"For the task: '{task_content}'\n\n"
        "Please provide a suggestion that matches the task's complexity:\n"
        "- For simple tasks: A concise 1-2 sentence suggestion\n"
        "- For complex tasks: A more detailed response with specific steps\n"
        "Always be specific and actionable."
    )


@dataclass
class Suggestion:
    """The outcome of a generation attempt."""

    model: str
    text: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)

    @property
    def formatted(self) -> str:
        """Display/storage string, including the ``MODEL SUGGESTION:`` marker."""
        if self.error is not None:
            return f"Error: {self.error}"
        if not self.text:
            return f"Unable to generate suggestion using {self.model.upper()}"
        return f"{self.model.upper()} SUGGESTION:\n{self.text}"


class SuggestionEngine:
    """Generates task suggestions, routing each model to its provider.

    Providers are injected (one per :class:`Provider`), which is what makes the
    engine unit-testable without real network calls. The original code's bug —
    building a fresh OpenAI client inside the generate function and ignoring the
    one passed in — is gone: dispatch goes through the injected provider map.
    """

    def __init__(
        self,
        providers: Mapping[Provider, SuggestionProvider],
        *,
        fallback_model: str = DEFAULT_MODEL,
    ) -> None:
        self._providers = providers
        self._fallback_model = fallback_model

    def generate(
        self,
        *,
        task_content: str,
        model: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str = DEFAULT_EFFORT,
        enable_fallback: bool = False,
        conversation: list[Message] | None = None,
    ) -> Suggestion:
        spec = get_model_spec(model)
        effort_error = spec.effort_error(reasoning_effort)
        if effort_error is not None:
            return Suggestion(model=model, text="", error=effort_error)
        messages: list[Message] = (
            list(conversation)
            if conversation
            else [{"role": "user", "content": _initial_user_prompt(task_content)}]
        )

        request = CompletionRequest(
            system=_SYSTEM,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        try:
            text = self._provider_for(model, spec.provider).complete(spec, request)
        except ProviderError as exc:
            if self._can_fallback(model, enable_fallback):
                return self._fallback(
                    task_content, max_tokens, temperature, reasoning_effort, conversation
                )
            return Suggestion(model=model, text="", error=str(exc))

        if not text and self._can_fallback(model, enable_fallback):
            return self._fallback(
                task_content, max_tokens, temperature, reasoning_effort, conversation
            )

        return Suggestion(model=model, text=text)

    def _provider_for(self, model: str, provider: Provider) -> SuggestionProvider:
        """The configured provider for ``model``, or a ProviderError naming the missing key."""
        try:
            return self._providers[provider]
        except KeyError:
            raise ProviderError(missing_key_message(provider, model)) from None

    def _can_fallback(self, model: str, enable_fallback: bool) -> bool:
        return enable_fallback and model != self._fallback_model

    def _fallback(
        self,
        task_content: str,
        max_tokens: int | None,
        temperature: float,
        reasoning_effort: str,
        conversation: list[Message] | None,
    ) -> Suggestion:
        return self.generate(
            task_content=task_content,
            model=self._fallback_model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            enable_fallback=False,
            conversation=conversation,
        )
