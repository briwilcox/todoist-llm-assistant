"""Structural protocols for the parts of Todoist objects this app reads.

Using protocols (rather than importing the concrete ``todoist_api_python`` models)
keeps the pure logic layer decoupled from the SDK and trivial to test with
lightweight doubles. The real ``Task``/``Due`` dataclasses satisfy these
structurally.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, runtime_checkable

# A single chat message, e.g. {"role": "user", "content": "..."}.
Message = dict[str, str]


@runtime_checkable
class DueLike(Protocol):
    date: date | datetime
    string: str
    is_recurring: bool


@runtime_checkable
class TaskLike(Protocol):
    id: str
    content: str
    description: str
    project_id: str
    due: DueLike | None
