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


# Members are read-only properties: the app only reads tasks, and read-only members let
# any object with matching attributes (the SDK's dataclasses, test doubles) satisfy them.


@runtime_checkable
class DueLike(Protocol):
    @property
    def date(self) -> date | datetime: ...
    @property
    def string(self) -> str: ...
    @property
    def is_recurring(self) -> bool: ...


@runtime_checkable
class TaskLike(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def content(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def project_id(self) -> str: ...
    @property
    def due(self) -> DueLike | None: ...
