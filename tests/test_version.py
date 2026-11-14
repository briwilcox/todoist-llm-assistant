"""The package version is declared twice; the copies must agree."""

from __future__ import annotations

from importlib.metadata import version

import todoist_assistant


def test_module_version_matches_installed_metadata() -> None:
    assert todoist_assistant.__version__ == version("todoist-assistant")
