"""Tests for the Todoist v4 API wrapper."""

from __future__ import annotations

import pytest

from todoist_assistant.todoist_client import TodoistClient

from .factories import FakeProject, FakeTodoistAPI, make_task


def test_iter_all_tasks_flattens_paginator() -> None:
    pages = [
        [make_task(id="1"), make_task(id="2")],
        [make_task(id="3")],
    ]
    client = TodoistClient(FakeTodoistAPI(task_pages=pages))
    assert [t.id for t in client.iter_all_tasks()] == ["1", "2", "3"]


def test_iter_all_tasks_handles_no_pages() -> None:
    client = TodoistClient(FakeTodoistAPI(task_pages=[]))
    assert client.iter_all_tasks() == []


def test_find_inbox_id_returns_inbox_project() -> None:
    pages = [
        [FakeProject(id="work"), FakeProject(id="home")],
        [FakeProject(id="inbox", is_inbox_project=True)],
    ]
    client = TodoistClient(FakeTodoistAPI(project_pages=pages))
    assert client.find_inbox_id() == "inbox"


def test_find_inbox_id_returns_none_when_absent() -> None:
    pages = [[FakeProject(id="work")]]
    client = TodoistClient(FakeTodoistAPI(project_pages=pages))
    assert client.find_inbox_id() is None


def test_update_description_calls_update_task() -> None:
    api = FakeTodoistAPI()
    TodoistClient(api).update_description("task-9", "GPT SUGGESTION:\nDo it")
    assert api.updates == [{"task_id": "task-9", "description": "GPT SUGGESTION:\nDo it"}]


def test_check_connection_consumes_one_page() -> None:
    api = FakeTodoistAPI(task_pages=[[make_task(id="1")]])
    TodoistClient(api).check_connection()
    assert api.get_tasks_calls == [{"limit": 1}]


def test_check_connection_propagates_errors() -> None:
    api = FakeTodoistAPI(error=RuntimeError("bad token"))
    with pytest.raises(RuntimeError, match="bad token"):
        TodoistClient(api).check_connection()
