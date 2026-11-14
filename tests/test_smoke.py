"""End-to-end smoke tests of the real entry point.

``cli.main`` runs with the real Todoist, OpenAI and Anthropic SDK clients; only their
HTTP transports are swapped for the in-memory servers in ``tests.http_fakes``. No
network, no real keys.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from todoist_assistant import cli

from .http_fakes import FakeLLMServer, FakeTodoistServer, todoist_task

PAST = "2026-01-15"
FUTURE = "2999-01-01"


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.ini"
    path.write_text(
        "[todoist]\napi_key = fake-todoist\n"
        "[openai]\napi_key = fake-openai\n"
        "[anthropic]\napi_key = fake-anthropic\n"
        "[models]\npreferred_model = gpt-6-luna\n"
        "[project_models]\n6Xc7mPq2Wv9RhJ4f = claude-haiku-5-5\n"
    )
    return path


def test_update_all_run_writes_suggestions_through_real_sdks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    todoist = FakeTodoistServer(
        tasks=[
            todoist_task("t1", "Pay the water bill", due=PAST),
            todoist_task("t2", "Book a dentist visit", due=PAST, project_id="6Xc7mPq2Wv9RhJ4f"),
            todoist_task("t3", "Already done", due=PAST, description="GPT SUGGESTION: x"),
            todoist_task("t4", "Plan next year", due=FUTURE),
            todoist_task("t5", "Water plants", due=PAST, is_recurring=True),
        ],
        page_size=2,  # three pages, so pagination is exercised
    )
    llm = FakeLLMServer(
        openai_reply="Open the bill and pay it online.",
        anthropic_reply="Book the appointment for Tuesday.",
    )
    http = cli.HttpClients(
        todoist=todoist.client(),
        openai=llm.openai_client(),
        anthropic=llm.anthropic_client(),
    )

    rc = cli.main(
        ["--config", str(write_config(tmp_path)), "--update-all", "--reasoning", "low"],
        env={},
        http=http,
    )

    assert rc == 0
    assert todoist.updates == {
        "t2": "CLAUDE-HAIKU-5-5 SUGGESTION:\nBook the appointment for Tuesday.",
        "t1": "GPT-6-LUNA SUGGESTION:\nOpen the bill and pay it online.",
    }
    (openai_body,) = llm.bodies("/chat/completions")
    assert openai_body["model"] == "gpt-6-luna"
    assert openai_body["reasoning_effort"] == "low"
    assert "Pay the water bill" in openai_body["messages"][-1]["content"]
    (anthropic_body,) = llm.bodies("/v1/messages")
    assert anthropic_body["model"] == "claude-haiku-5-5"
    assert anthropic_body["output_config"] == {"effort": "low"}
    assert "Book a dentist visit" in anthropic_body["messages"][-1]["content"]
    out = capsys.readouterr().out
    assert "Updated task 'Pay the water bill'" in out
    assert "Updated task 'Book a dentist visit'" in out


def test_no_update_run_reads_but_never_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    todoist = FakeTodoistServer(tasks=[todoist_task("t1", "Pay the water bill", due=PAST)])
    llm = FakeLLMServer()
    http = cli.HttpClients(
        todoist=todoist.client(), openai=llm.openai_client(), anthropic=llm.anthropic_client()
    )

    rc = cli.main(["--config", str(write_config(tmp_path)), "--no-update"], env={}, http=http)

    assert rc == 0
    assert todoist.updates == {}
    assert all(e.method == "GET" for e in todoist.exchanges)
    assert "Would update task 'Pay the water bill' (demo mode, skipped)" in capsys.readouterr().out


def test_installed_module_entry_point_lists_models() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "todoist_assistant", "--list-models"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "gpt-6-luna" in result.stdout
    assert "claude-haiku-5-5" in result.stdout
