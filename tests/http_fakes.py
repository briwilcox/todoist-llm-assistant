"""In-memory HTTP fakes for the real OpenAI, Anthropic and Todoist SDK clients.

The unit-test doubles in ``tests.factories`` replace the SDK clients outright, so they
accept any keyword argument. These fakes sit one layer lower: the real SDK builds and
sends the request, and only the socket is replaced. A parameter the installed SDK no
longer accepts, or a request body the API would reject, fails here as it would live.
Nothing in this module opens a network connection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import httpx2

JsonObject = dict[str, Any]


@dataclass
class Exchange:
    """One request an SDK sent, as the fake server saw it."""

    method: str
    path: str
    body: JsonObject | None
    query: dict[str, str]


@dataclass
class FakeLLMServer:
    """Answers OpenAI Chat Completions and Anthropic Messages requests."""

    openai_reply: str = "Open the bill and pay it online."
    anthropic_reply: str = "Book the appointment for Tuesday."
    exchanges: list[Exchange] = field(default_factory=list)

    def openai_client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self._handle_openai))

    def anthropic_client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self._handle_anthropic))

    def bodies(self, path_suffix: str) -> list[JsonObject]:
        return [e.body for e in self.exchanges if e.path.endswith(path_suffix) and e.body]

    def _record(self, request: httpx2.Request) -> JsonObject:
        body: JsonObject = json.loads(request.content) if request.content else {}
        self.exchanges.append(
            Exchange(request.method, request.url.path, body, dict(request.url.params))
        )
        return body

    def _handle_openai(self, request: httpx2.Request) -> httpx2.Response:
        body = self._record(request)
        assert request.url.path == "/v1/chat/completions", request.url.path
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.openai_reply},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def _handle_anthropic(self, request: httpx2.Request) -> httpx2.Response:
        body = self._record(request)
        assert request.url.path == "/v1/messages", request.url.path
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                # Adaptive thinking is always on for the Claude 5.5 models, so a real
                # reply can open with a thinking block before the text.
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "text", "text": self.anthropic_reply},
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        )


def todoist_task(
    task_id: str,
    content: str,
    *,
    due: str | None = None,
    description: str = "",
    project_id: str = "proj-1",
    is_recurring: bool = False,
) -> JsonObject:
    """A Todoist API v1 task object with the fields the SDK requires."""
    return {
        "id": task_id,
        "content": content,
        "description": description,
        "project_id": project_id,
        "section_id": None,
        "parent_id": None,
        "labels": [],
        "priority": 1,
        "due": (
            {"date": due, "string": due, "lang": "en", "is_recurring": is_recurring}
            if due
            else None
        ),
        "deadline": None,
        "duration": None,
        "collapsed": False,
        "child_order": 1,
        "responsible_uid": None,
        "assigned_by_uid": None,
        "completed_at": None,
        "added_by_uid": "user-1",
        "added_at": "2026-11-01T09:00:00Z",
        "updated_at": "2026-11-01T09:00:00Z",
    }


@dataclass
class FakeTodoistServer:
    """Serves Todoist API v1 task pages and records description updates."""

    tasks: list[JsonObject] = field(default_factory=list)
    page_size: int = 2
    exchanges: list[Exchange] = field(default_factory=list)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    @property
    def updates(self) -> dict[str, str]:
        """Task id -> description written, in request order."""
        out: dict[str, str] = {}
        for e in self.exchanges:
            if e.method == "POST" and e.path.startswith("/api/v1/tasks/") and e.body:
                out[e.path.rsplit("/", 1)[-1]] = e.body["description"]
        return out

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body: JsonObject | None = json.loads(request.content) if request.content else None
        self.exchanges.append(
            Exchange(request.method, request.url.path, body, dict(request.url.params))
        )
        path = request.url.path
        if request.method == "GET" and path == "/api/v1/tasks":
            return httpx.Response(200, json=self._page(request))
        if request.method == "POST" and path.startswith("/api/v1/tasks/"):
            task_id = path.rsplit("/", 1)[-1]
            task = next(t for t in self.tasks if t["id"] == task_id)
            assert body is not None
            task["description"] = body["description"]
            return httpx.Response(200, json=task)
        return httpx.Response(404, json={"error": f"unexpected {request.method} {path}"})

    def _page(self, request: httpx.Request) -> JsonObject:
        start = int(request.url.params.get("cursor") or 0)
        limit = int(request.url.params.get("limit") or self.page_size)
        end = start + limit
        return {
            "results": self.tasks[start:end],
            "next_cursor": str(end) if end < len(self.tasks) else None,
        }
