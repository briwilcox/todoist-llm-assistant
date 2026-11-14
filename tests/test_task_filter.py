"""Tests for the pure task selection/filtering logic."""

from __future__ import annotations

from datetime import date, datetime

from todoist_assistant import task_filter as tf

from .factories import make_task

TODAY = date(2026, 6, 4)


def test_has_suggestion_is_case_insensitive() -> None:
    assert tf.has_suggestion(make_task(description="gpt SUGGESTION: do it"))
    assert tf.has_suggestion(make_task(description="model suggestion: do it"))
    assert not tf.has_suggestion(make_task(description="just a note"))
    assert not tf.has_suggestion(make_task(description=""))


def test_due_on_returns_date_for_date_and_datetime() -> None:
    assert tf.due_on(make_task(due=date(2026, 6, 1))) == date(2026, 6, 1)
    assert tf.due_on(make_task(due=datetime(2026, 6, 1, 9, 30))) == date(2026, 6, 1)
    assert tf.due_on(make_task(due=None)) is None


def test_is_recurring_uses_due_flag_not_string() -> None:
    assert tf.is_recurring(make_task(due=date(2026, 6, 1), is_recurring=True))
    # "every" in the string must NOT trigger recurring detection anymore
    assert not tf.is_recurring(
        make_task(due=date(2026, 6, 1), due_string="every detail counts", is_recurring=False)
    )
    assert not tf.is_recurring(make_task(due=None))


def test_select_defaults_to_due_today_or_earlier() -> None:
    overdue = make_task(id="a", due=date(2026, 6, 1))
    today = make_task(id="b", due=date(2026, 6, 4))
    future = make_task(id="c", due=date(2026, 7, 1))
    no_date = make_task(id="d", due=None)

    selected = tf.select_tasks([future, today, overdue, no_date], today=TODAY)
    ids = [t.id for t in selected]

    assert "c" not in ids  # future excluded
    assert "d" not in ids  # no-date excluded by default
    assert set(ids) == {"a", "b"}


def test_select_skips_tasks_that_already_have_suggestions() -> None:
    done = make_task(id="a", due=date(2026, 6, 1), description="GPT SUGGESTION: x")
    todo = make_task(id="b", due=date(2026, 6, 1))
    assert [t.id for t in tf.select_tasks([done, todo], today=TODAY)] == ["b"]


def test_select_skips_recurring_tasks() -> None:
    recurring = make_task(id="a", due=date(2026, 6, 1), is_recurring=True)
    once = make_task(id="b", due=date(2026, 6, 1))
    assert [t.id for t in tf.select_tasks([recurring, once], today=TODAY)] == ["b"]


def test_select_include_no_date() -> None:
    no_date = make_task(id="a", due=None)
    future = make_task(id="b", due=date(2026, 7, 1))
    selected = tf.select_tasks([no_date, future], today=TODAY, include_no_date=True)
    assert [t.id for t in selected] == ["a"]


def test_select_include_inbox_by_project_id() -> None:
    inbox_task = make_task(id="a", project_id="inbox", due=None)
    other_task = make_task(id="b", project_id="work", due=None)
    selected = tf.select_tasks(
        [inbox_task, other_task],
        today=TODAY,
        include_inbox=True,
        inbox_id="inbox",
    )
    assert [t.id for t in selected] == ["a"]


def test_select_include_inbox_noop_without_inbox_id() -> None:
    inbox_task = make_task(id="a", project_id="inbox", due=None)
    selected = tf.select_tasks([inbox_task], today=TODAY, include_inbox=True, inbox_id=None)
    assert selected == []


def test_select_sorts_by_due_descending_no_date_last() -> None:
    t1 = make_task(id="early", due=date(2026, 6, 1))
    t2 = make_task(id="late", due=date(2026, 6, 4))
    t3 = make_task(id="none", due=None)
    selected = tf.select_tasks([t1, t3, t2], today=TODAY, include_no_date=True)
    assert [t.id for t in selected] == ["late", "early", "none"]


def test_window_applies_offset_and_limit() -> None:
    tasks = [make_task(id=str(i)) for i in range(20)]
    assert [t.id for t in tf.window(tasks, offset=0, limit=7)] == [str(i) for i in range(7)]
    assert [t.id for t in tf.window(tasks, offset=7, limit=7)] == [str(i) for i in range(7, 14)]
    assert [t.id for t in tf.window(tasks, offset=14, limit=7)] == [str(i) for i in range(14, 20)]
