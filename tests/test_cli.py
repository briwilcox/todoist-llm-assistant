"""Tests for the CLI: parsing, validation, and wiring into the app."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from todoist_assistant import cli
from todoist_assistant.config import Config
from todoist_assistant.models import Provider
from todoist_assistant.todoist_client import TodoistClient

from .factories import FakeProvider, FakeTodoistAPI, make_task


def test_model_choices_come_from_registry() -> None:
    parser = cli.build_parser()
    action = next(a for a in parser._actions if a.dest == "model")
    assert "gpt-5.5" in action.choices
    assert "claude-opus-4-8" in action.choices


def test_parse_defaults() -> None:
    args = cli.parse_args([])
    assert args.interactive is False
    assert args.update_all is False
    assert args.model is None
    assert args.tokens is None
    assert args.temperature == 0.7
    assert args.reasoning == "medium"


def test_parse_flags() -> None:
    args = cli.parse_args(
        ["-i", "-u", "--inbox", "--no-due-date", "-m", "gpt-5.5", "-t", "200", "--fallback"]
    )
    assert args.interactive and args.update_all
    assert args.inbox and args.no_due_date and args.fallback
    assert args.model == "gpt-5.5"
    assert args.tokens == 200


def test_run_options_from_args() -> None:
    args = cli.parse_args(["-i", "--inbox", "-m", "gpt-5.4", "-t", "150", "--reasoning", "high"])
    opts = cli.run_options_from_args(args)
    assert opts.interactive is True
    assert opts.include_inbox is True
    assert opts.model == "gpt-5.4"
    assert opts.max_tokens == 150
    assert opts.reasoning_effort == "high"


def test_validate_temperature_out_of_range_resets_to_default() -> None:
    assert cli.validate_temperature(5.0) == 0.7
    assert cli.validate_temperature(-1.0) == 0.7
    assert cli.validate_temperature(1.2) == 1.2


def test_list_models_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["--list-models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gpt-5.5" in out
    assert "claude-opus-4-8" in out


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.ini"
    path.write_text(body)
    return path


def test_main_happy_path_wires_and_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = write_config(
        tmp_path,
        "[todoist]\napi_key = td\n[openai]\napi_key = oa\n",
    )
    api = FakeTodoistAPI(task_pages=[[make_task(id="a", due=date(2026, 6, 1))]])

    monkeypatch.setattr(
        cli, "_build_providers", lambda config: {Provider.OPENAI: FakeProvider(default="Go")}
    )
    monkeypatch.setattr(cli, "_build_todoist", lambda config: TodoistClient(api))
    # freeze "today" so the due task is selected
    monkeypatch.setattr(cli, "_today", lambda: date(2026, 6, 4))

    rc = cli.main(["--config", str(cfg)])

    assert rc == 0
    assert api.updates[0]["task_id"] == "a"


def test_main_config_error_returns_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "none.ini"
    rc = cli.main(["--config", str(missing)], env={})
    assert rc == 1
    assert "Todoist" in capsys.readouterr().out


def test_main_errors_when_model_provider_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Only a Todoist key: default model gpt-5.4-mini needs OpenAI, which is absent.
    cfg = write_config(tmp_path, "[todoist]\napi_key = td\n")
    monkeypatch.setattr(cli, "_build_todoist", lambda config: TodoistClient(FakeTodoistAPI()))
    rc = cli.main(["--config", str(cfg)], env={})
    assert rc == 1
    assert "openai" in capsys.readouterr().out.lower()


def test_main_connection_failure_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = write_config(tmp_path, "[todoist]\napi_key = td\n[openai]\napi_key = oa\n")
    failing = FakeTodoistAPI(error=RuntimeError("401 unauthorized"))
    monkeypatch.setattr(cli, "_build_providers", lambda config: {Provider.OPENAI: FakeProvider()})
    monkeypatch.setattr(cli, "_build_todoist", lambda config: TodoistClient(failing))
    rc = cli.main(["--config", str(cfg)], env={})
    assert rc == 1
    assert "Todoist" in capsys.readouterr().out


def test_build_providers_constructs_for_configured_keys() -> None:
    providers = cli._build_providers(
        Config(todoist_api_key="t", openai_api_key="o", anthropic_api_key="a")
    )
    assert set(providers) == {Provider.OPENAI, Provider.ANTHROPIC}


def test_build_providers_empty_without_keys() -> None:
    assert cli._build_providers(Config(todoist_api_key="t")) == {}


def test_build_todoist_returns_client() -> None:
    assert isinstance(cli._build_todoist(Config(todoist_api_key="t")), TodoistClient)


def test_today_returns_a_date() -> None:
    assert isinstance(cli._today(), date)


def test_main_warns_on_out_of_range_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = write_config(tmp_path, "[todoist]\napi_key = td\n[openai]\napi_key = oa\n")
    api = FakeTodoistAPI(task_pages=[])  # no tasks -> run returns immediately
    monkeypatch.setattr(cli, "_build_providers", lambda c: {Provider.OPENAI: FakeProvider()})
    monkeypatch.setattr(cli, "_build_todoist", lambda c: TodoistClient(api))
    monkeypatch.setattr(cli, "_today", lambda: date(2026, 6, 4))

    rc = cli.main(["--config", str(cfg), "-temp", "9"])
    assert rc == 0
    assert "Temperature must be between" in capsys.readouterr().out


def test_main_unknown_preferred_model_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = write_config(
        tmp_path,
        "[todoist]\napi_key = td\n[openai]\napi_key = oa\n[models]\npreferred_model = bogus\n",
    )
    rc = cli.main(["--config", str(cfg)], env={})
    assert rc == 1
    assert "bogus" in capsys.readouterr().out
