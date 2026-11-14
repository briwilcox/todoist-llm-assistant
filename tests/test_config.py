"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from todoist_assistant.config import DEFAULT_MODEL, ConfigError, load_config


def write_ini(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.ini"
    path.write_text(body)
    return path


def test_loads_all_sections(tmp_path: Path) -> None:
    path = write_ini(
        tmp_path,
        """
        [todoist]
        api_key = td-key

        [openai]
        api_key = oa-key

        [anthropic]
        api_key = an-key

        [models]
        preferred_model = gpt-6-astra

        [project_models]
        2203306141 = claude-opus-5-5
        """,
    )
    cfg = load_config(path, env={})
    assert cfg.todoist_api_key == "td-key"
    assert cfg.openai_api_key == "oa-key"
    assert cfg.anthropic_api_key == "an-key"
    assert cfg.preferred_model == "gpt-6-astra"
    assert cfg.project_models == {"2203306141": "claude-opus-5-5"}


def test_preferred_model_defaults_when_absent(tmp_path: Path) -> None:
    path = write_ini(tmp_path, "[todoist]\napi_key = x\n")
    cfg = load_config(path, env={})
    assert cfg.preferred_model == DEFAULT_MODEL == "gpt-6-luna"


def test_preferred_model_backward_compat_under_openai(tmp_path: Path) -> None:
    path = write_ini(
        tmp_path,
        "[todoist]\napi_key = x\n[openai]\napi_key = y\npreferred_model = gpt-6.1-sol\n",
    )
    cfg = load_config(path, env={})
    assert cfg.preferred_model == "gpt-6.1-sol"


def test_env_overrides_file_keys(tmp_path: Path) -> None:
    path = write_ini(
        tmp_path,
        "[todoist]\napi_key = file-td\n[openai]\napi_key = file-oa\n",
    )
    cfg = load_config(
        path,
        env={
            "TODOIST_API_KEY": "env-td",
            "OPENAI_API_KEY": "env-oa",
            "ANTHROPIC_API_KEY": "env-an",
        },
    )
    assert cfg.todoist_api_key == "env-td"
    assert cfg.openai_api_key == "env-oa"
    assert cfg.anthropic_api_key == "env-an"


def test_env_only_without_config_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.ini"
    cfg = load_config(missing, env={"TODOIST_API_KEY": "env-td"})
    assert cfg.todoist_api_key == "env-td"
    assert cfg.openai_api_key is None
    assert cfg.preferred_model == DEFAULT_MODEL


def test_missing_todoist_key_raises(tmp_path: Path) -> None:
    path = write_ini(tmp_path, "[openai]\napi_key = y\n")
    with pytest.raises(ConfigError, match="Todoist"):
        load_config(path, env={})


def test_project_ids_keep_their_case(tmp_path: Path) -> None:
    # Todoist API v1 project ids are mixed-case strings; matching is exact, so the
    # mapping must not lowercase them the way configparser does by default.
    path = write_ini(
        tmp_path,
        "[todoist]\napi_key = x\n[project_models]\n6Xc7mPq2Wv9RhJ4f = claude-opus-5-5\n",
    )
    cfg = load_config(path, env={})
    assert cfg.project_models == {"6Xc7mPq2Wv9RhJ4f": "claude-opus-5-5"}
