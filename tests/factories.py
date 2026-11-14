"""Lightweight test doubles that structurally satisfy the app's protocols."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass
class FakeDue:
    date: date | datetime
    string: str = ""
    is_recurring: bool = False


@dataclass
class FakeTask:
    id: str = "1"
    content: str = "Write the docs"
    description: str = ""
    project_id: str = "proj-1"
    due: FakeDue | None = None


def make_task(
    *,
    id: str = "1",
    content: str = "Write the docs",
    description: str = "",
    project_id: str = "proj-1",
    due: date | datetime | None = None,
    due_string: str = "",
    is_recurring: bool = False,
) -> FakeTask:
    """Build a FakeTask that structurally satisfies ``TaskLike``."""
    due_obj = (
        FakeDue(date=due, string=due_string, is_recurring=is_recurring) if due is not None else None
    )
    return FakeTask(
        id=id,
        content=content,
        description=description,
        project_id=project_id,
        due=due_obj,
    )


# --- Fake OpenAI SDK client -------------------------------------------------


@dataclass
class _OAMessage:
    content: str | None


@dataclass
class _OAChoice:
    message: _OAMessage


@dataclass
class _OAResponse:
    choices: list[_OAChoice]


class _OACompletions:
    def __init__(self, owner: FakeOpenAIClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> _OAResponse:
        self._owner.calls.append(kwargs)
        if self._owner.error is not None:
            raise self._owner.error
        return _OAResponse(choices=[_OAChoice(_OAMessage(self._owner.content))])


class _OAChat:
    def __init__(self, owner: FakeOpenAIClient) -> None:
        self.completions = _OACompletions(owner)


class FakeOpenAIClient:
    """Mimics ``openai.OpenAI`` enough to exercise OpenAIProvider."""

    def __init__(
        self, content: str | None = "Do the thing.", error: Exception | None = None
    ) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.chat = _OAChat(self)

    @property
    def last_call(self) -> dict[str, Any]:
        return self.calls[-1]


# --- Fake Anthropic SDK client ----------------------------------------------


@dataclass
class _AnthropicTextBlock:
    text: str
    type: str = "text"


@dataclass
class _AnthropicThinkingBlock:
    thinking: str = ""
    type: str = "thinking"


@dataclass
class _AnthropicResponse:
    content: list[_AnthropicTextBlock | _AnthropicThinkingBlock]


class _AnthropicMessages:
    def __init__(self, owner: FakeAnthropicClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> _AnthropicResponse:
        self._owner.calls.append(kwargs)
        if self._owner.error is not None:
            raise self._owner.error
        blocks: list[_AnthropicTextBlock | _AnthropicThinkingBlock] = [
            _AnthropicTextBlock(t) for t in self._owner.texts
        ]
        if self._owner.thinking_first:
            blocks.insert(0, _AnthropicThinkingBlock())
        return _AnthropicResponse(content=blocks)


class FakeAnthropicClient:
    """Mimics ``anthropic.Anthropic`` enough to exercise AnthropicProvider."""

    def __init__(
        self,
        texts: list[str] | None = None,
        error: Exception | None = None,
        *,
        thinking_first: bool = False,
    ) -> None:
        self.texts = ["Do the thing."] if texts is None else texts
        self.error = error
        self.thinking_first = thinking_first
        self.calls: list[dict[str, Any]] = []
        self.messages = _AnthropicMessages(self)

    @property
    def last_call(self) -> dict[str, Any]:
        return self.calls[-1]


# --- Fake suggestion provider (for engine/app tests) ------------------------


class FakeProvider:
    """Records requests and returns canned text (or raises) per model id."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        default: str = "a suggestion",
        error: Exception | None = None,
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.error = error
        self.requests: list[tuple[Any, Any]] = []

    def complete(self, spec: Any, request: Any) -> str:
        self.requests.append((spec, request))
        if self.error is not None:
            raise self.error
        return self.responses.get(spec.id, self.default)

    @property
    def last_request(self) -> Any:
        return self.requests[-1][1]


# --- Fake Todoist API (v4 paginator shape) ----------------------------------


@dataclass
class FakeProject:
    id: str
    name: str = "Project"
    is_inbox_project: bool = False


class FakeTodoistAPI:
    """Mimics todoist_api_python TodoistAPI v4: get_* return iterators of pages."""

    def __init__(
        self,
        task_pages: list[list[Any]] | None = None,
        project_pages: list[list[Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.task_pages = task_pages if task_pages is not None else []
        self.project_pages = project_pages if project_pages is not None else []
        self.error = error
        self.updates: list[dict[str, Any]] = []
        self.get_tasks_calls: list[dict[str, Any]] = []

    def get_tasks(self, **kwargs: Any) -> Any:
        self.get_tasks_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return iter(self.task_pages)

    def get_projects(self, **kwargs: Any) -> Any:
        return iter(self.project_pages)

    def update_task(self, task_id: str, **kwargs: Any) -> None:
        self.updates.append({"task_id": task_id, **kwargs})
