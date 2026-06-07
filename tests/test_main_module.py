"""Cover the ``python -m todoist_assistant`` entry point guard."""

from __future__ import annotations

import runpy
import sys

import pytest


def test_module_entrypoint_runs_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["todoist_assistant", "--list-models"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("todoist_assistant", run_name="__main__")
    assert excinfo.value.code == 0
