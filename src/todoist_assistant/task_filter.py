"""Pure functions for selecting and ordering Todoist tasks.

No I/O, no globals — every input is passed in, which makes this the most
heavily unit-tested layer of the app.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime

from .types import TaskLike

SUGGESTION_MARKER = "SUGGESTION"


def has_suggestion(task: TaskLike) -> bool:
    """True if the task description already contains a generated suggestion."""
    return SUGGESTION_MARKER in (task.description or "").upper()


def due_on(task: TaskLike) -> date | None:
    """The task's due date as a plain ``date`` (dropping any time), or None."""
    if task.due is None:
        return None
    value = task.due.date
    if isinstance(value, datetime):
        return value.date()
    return value


def is_recurring(task: TaskLike) -> bool:
    """True for recurring tasks, using the structured ``is_recurring`` flag."""
    return task.due is not None and task.due.is_recurring


def select_tasks(
    tasks: Iterable[TaskLike],
    *,
    today: date,
    include_inbox: bool = False,
    include_no_date: bool = False,
    inbox_id: str | None = None,
    skip_existing: bool = True,
) -> list[TaskLike]:
    """Filter and sort tasks for suggestion generation.

    A task is included when any of these hold:
      * it is due today or earlier, or
      * ``include_inbox`` and it lives in the inbox project, or
      * ``include_no_date`` and it has no due date.

    Tasks that already carry a suggestion (when ``skip_existing``) or that are
    recurring are always excluded. Results are sorted by due date descending,
    with undated tasks last.
    """
    selected: list[TaskLike] = []
    for task in tasks:
        if skip_existing and has_suggestion(task):
            continue
        if is_recurring(task):
            continue

        due = due_on(task)
        due_ok = due is not None and due <= today
        inbox_ok = include_inbox and inbox_id is not None and task.project_id == inbox_id
        no_date_ok = include_no_date and task.due is None

        if due_ok or inbox_ok or no_date_ok:
            selected.append(task)

    selected.sort(key=_due_sort_key, reverse=True)
    return selected


def window[T](tasks: Sequence[T], *, offset: int, limit: int) -> list[T]:
    """Return a slice of ``tasks`` starting at ``offset`` of size ``limit``."""
    return list(tasks[offset : offset + limit])


def _due_sort_key(task: TaskLike) -> date:
    """Sort key: real due dates sort by value; undated tasks sort lowest.

    Used with ``reverse=True`` so the newest due date comes first and undated
    tasks fall to the end.
    """
    return due_on(task) or date.min
