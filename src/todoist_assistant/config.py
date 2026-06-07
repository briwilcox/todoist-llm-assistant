"""Configuration loading from an INI file and/or environment variables.

Unlike the original script (which read ``config.ini`` at import time into module
globals), config is loaded explicitly via :func:`load_config` with an injectable
path and environment mapping, so it can be exercised in tests without touching
the real filesystem or process environment.
"""

from __future__ import annotations

import configparser
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "gpt-5.4-mini"

_ENV_KEYS = {
    "todoist": "TODOIST_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass
class Config:
    todoist_api_key: str
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    preferred_model: str = DEFAULT_MODEL
    project_models: dict[str, str] = field(default_factory=dict)


def load_config(
    path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Load configuration, with environment variables overriding file values.

    Args:
        path: Path to the INI file. Defaults to ``config.ini`` in the CWD. A
            missing file is tolerated as long as required values come from env.
        env: Environment mapping (defaults to ``os.environ``). Injectable for tests.

    Raises:
        ConfigError: If the Todoist API key is not provided by file or env.
    """
    env = os.environ if env is None else env
    path = Path("config.ini") if path is None else Path(path)

    parser = configparser.ConfigParser(interpolation=None)
    if path.exists():
        parser.read(path)

    todoist_key = _resolve(parser, env, "todoist", "api_key")
    if not todoist_key:
        raise ConfigError(
            "Missing Todoist API key. Set it under [todoist] api_key in the config "
            "file or via the TODOIST_API_KEY environment variable."
        )

    preferred_model = (
        parser.get("models", "preferred_model", fallback=None)
        or parser.get("openai", "preferred_model", fallback=None)
        or DEFAULT_MODEL
    )

    project_models: dict[str, str] = (
        dict(parser["project_models"]) if parser.has_section("project_models") else {}
    )

    return Config(
        todoist_api_key=todoist_key,
        openai_api_key=_resolve(parser, env, "openai", "api_key"),
        anthropic_api_key=_resolve(parser, env, "anthropic", "api_key"),
        preferred_model=preferred_model,
        project_models=project_models,
    )


def _resolve(
    parser: configparser.ConfigParser,
    env: Mapping[str, str],
    section: str,
    option: str,
) -> str | None:
    """Return the env value if set, else the file value, else None."""
    env_value = env.get(_ENV_KEYS[section])
    if env_value:
        return env_value
    return parser.get(section, option, fallback=None)
