"""A thin, testable wrapper over the Todoist v4 API.

v4 changed the read methods to return paginators (``Iterator[list[T]]``) rather
than flat lists, which silently broke the original code's ``tasks.sort(...)``.
This wrapper flattens that pagination and exposes only the handful of operations
the app needs, against an injected API object.
"""

from __future__ import annotations

from typing import Any

from .types import TaskLike


class TodoistClient:
    def __init__(self, api: Any) -> None:
        self._api = api

    def check_connection(self) -> None:
        """Fire a single small request to validate credentials/connectivity."""
        next(iter(self._api.get_tasks(limit=1)), None)

    def iter_all_tasks(self) -> list[TaskLike]:
        """Return every active task, flattening the v4 paginator."""
        tasks: list[TaskLike] = []
        for page in self._api.get_tasks():
            tasks.extend(page)
        return tasks

    def find_inbox_id(self) -> str | None:
        """Return the Inbox project's id via its ``is_inbox_project`` flag."""
        for page in self._api.get_projects():
            for project in page:
                if project.is_inbox_project:
                    return str(project.id)
        return None

    def update_description(self, task_id: str, description: str) -> None:
        """Replace a task's description."""
        self._api.update_task(task_id, description=description)
