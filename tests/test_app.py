"""Tests for the orchestration layer (batch + interactive flows)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from todoist_assistant.app import App, RunOptions
from todoist_assistant.config import Config
from todoist_assistant.models import Provider
from todoist_assistant.suggestion_engine import SuggestionEngine
from todoist_assistant.todoist_client import TodoistClient

from .factories import FakeProvider, FakeTodoistAPI, make_task

TODAY = date(2026, 6, 4)


class ScriptedInput:
    """Returns queued answers in order; raises if exhausted."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        return self._answers.pop(0)


def build_app(
    task_pages: Sequence[Sequence[object]],
    *,
    config: Config | None = None,
    openai: FakeProvider | None = None,
    anthropic: FakeProvider | None = None,
    input_answers: list[str] | None = None,
    input_fn: ScriptedInput | None = None,
    project_pages: Sequence[Sequence[object]] | None = None,
) -> tuple[App, FakeTodoistAPI, FakeProvider, FakeProvider, list[str]]:
    api = FakeTodoistAPI(
        task_pages=[list(p) for p in task_pages],
        project_pages=[list(p) for p in project_pages or []],
    )
    todoist = TodoistClient(api)
    oa = openai or FakeProvider(default="Step one.")
    an = anthropic or FakeProvider(default="Claude step.")
    engine = SuggestionEngine({Provider.OPENAI: oa, Provider.ANTHROPIC: an})
    cfg = config or Config(todoist_api_key="x", preferred_model="gpt-6-luna")
    out: list[str] = []
    app = App(
        todoist,
        engine,
        cfg,
        today=TODAY,
        input_fn=input_fn or ScriptedInput(input_answers or []),
        output_fn=out.append,
    )
    return app, api, oa, an, out


def test_batch_generates_and_updates_due_tasks() -> None:
    pages = [[make_task(id="a", content="Pay bill", due=date(2026, 6, 1))]]
    app, api, _, _, _ = build_app(pages)
    app.run(RunOptions())
    assert api.updates == [{"task_id": "a", "description": "GPT-6-LUNA SUGGESTION:\nStep one."}]


def test_batch_skips_tasks_that_already_have_suggestions() -> None:
    pages = [
        [
            make_task(id="a", due=date(2026, 6, 1), description="GPT SUGGESTION: done"),
            make_task(id="b", due=date(2026, 6, 1)),
        ]
    ]
    app, api, _, _, _ = build_app(pages)
    app.run(RunOptions())
    assert [u["task_id"] for u in api.updates] == ["b"]


def test_demo_mode_does_not_update() -> None:
    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    app, api, _, _, out = build_app(pages)
    app.run(RunOptions(no_update=True))
    assert api.updates == []
    assert any("Would update" in line for line in out)


def test_no_matching_tasks_emits_guidance() -> None:
    pages = [[make_task(id="a", due=date(2026, 7, 1))]]  # future only
    app, api, _, _, out = build_app(pages)
    app.run(RunOptions())
    assert api.updates == []
    assert any("No tasks" in line for line in out)


def test_cli_model_overrides_and_routes_to_anthropic() -> None:
    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    app, api, oa, an, _ = build_app(pages)
    app.run(RunOptions(model="claude-opus-5-5"))
    assert len(an.requests) == 1
    assert len(oa.requests) == 0
    assert api.updates[0]["description"].startswith("CLAUDE-OPUS-5-5 SUGGESTION:")


def test_project_specific_model_is_used() -> None:
    cfg = Config(
        todoist_api_key="x",
        preferred_model="gpt-6-luna",
        project_models={"special": "gpt-6-astra"},
    )
    pages = [[make_task(id="a", project_id="special", due=date(2026, 6, 1))]]
    app, _, oa, _, _ = build_app(pages, config=cfg)
    app.run(RunOptions())
    assert oa.requests[0][0].id == "gpt-6-astra"


def test_update_all_processes_every_match_without_paging() -> None:
    # 9 due tasks: update_all should do all 9 (batch window of 7 does not apply).
    pages = [[make_task(id=str(i), due=date(2026, 6, 1)) for i in range(9)]]
    app, api, _, _, _ = build_app(pages)
    app.run(RunOptions(update_all=True))
    assert len(api.updates) == 9


def test_non_interactive_batch_limited_to_batch_size() -> None:
    pages = [[make_task(id=str(i), due=date(2026, 6, 1)) for i in range(9)]]
    app, api, _, _, _ = build_app(pages)
    app.run(RunOptions())  # not update_all, not interactive -> one window of 7
    assert len(api.updates) == 7


def test_include_inbox_uses_inbox_project_id() -> None:
    projects = [[FakeInbox("inbox")]]
    pages = [[make_task(id="a", project_id="inbox", due=None)]]
    app, api, _, _, _ = build_app(pages, project_pages=projects)
    app.run(RunOptions(include_inbox=True))
    assert [u["task_id"] for u in api.updates] == ["a"]


def test_error_suggestion_is_not_stored() -> None:
    from todoist_assistant.providers.base import ProviderError

    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    app, api, _, _, _ = build_app(pages, openai=FakeProvider(error=ProviderError("down")))
    app.run(RunOptions())
    assert api.updates == []


# --- interactive flows ------------------------------------------------------


def test_interactive_accept_updates_task() -> None:
    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    # accept this task, then quit the paging prompt
    app, api, _, _, _ = build_app(pages, input_answers=["y", "q"])
    app.run(RunOptions(interactive=True))
    assert [u["task_id"] for u in api.updates] == ["a"]


def test_interactive_skip_does_not_update() -> None:
    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    app, api, _, _, _ = build_app(pages, input_answers=["n", "q"])
    app.run(RunOptions(interactive=True))
    assert api.updates == []


def test_interactive_feedback_then_accept_regenerates() -> None:
    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    oa = FakeProvider(default="answer")
    # 'f' -> feedback prompt -> regenerate -> 'y' accept -> 'q' quit
    app, api, used_oa, _, _ = build_app(pages, openai=oa, input_answers=["f", "be terse", "y", "q"])
    app.run(RunOptions(interactive=True))
    assert len(used_oa.requests) == 2  # initial + after feedback
    # feedback message present in the second request's conversation
    second_convo = used_oa.requests[1][1].messages
    assert {"role": "user", "content": "be terse"} in second_convo
    assert [u["task_id"] for u in api.updates] == ["a"]


def test_interactive_switch_model_then_accept() -> None:
    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    oa = FakeProvider(default="oa")
    an = FakeProvider(default="claude")
    # 'o' switches to claude-opus, then 'y' accept, then 'q'
    app, api, _, used_an, _ = build_app(
        pages, openai=oa, anthropic=an, input_answers=["o", "y", "q"]
    )
    app.run(RunOptions(interactive=True))
    assert len(used_an.requests) == 1
    assert api.updates[0]["description"].startswith("CLAUDE-OPUS-5-5 SUGGESTION:")


def test_interactive_menu_names_the_models_its_keys_switch_to() -> None:
    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    scripted = ScriptedInput(["n", "q"])
    app, _, _, _, _ = build_app(pages, input_fn=scripted)
    app.run(RunOptions(interactive=True))
    menu = scripted.prompts[0].splitlines()
    assert (
        "  m/g/o/s  Switch model (m = gpt-6-luna, g = gpt-6-astra, "
        "o = claude-opus-5-5, s = claude-sonnet-5-5)"
    ) in menu
    assert "  fm/fg/fo/fs  Feedback AND switch model" in menu


def test_interactive_paging_advances_offset() -> None:
    pages = [[make_task(id=str(i), due=date(2026, 6, 1)) for i in range(10)]]
    # accept 7 in first window, 'next', accept remaining 3, 'next' -> window empties -> stop
    answers = ["y"] * 7 + ["1"] + ["y"] * 3 + ["1"]
    app, api, _, _, _ = build_app(pages, input_answers=answers)
    app.run(RunOptions(interactive=True))
    assert len(api.updates) == 10


def test_interactive_error_result_is_not_appended_then_skip() -> None:
    from todoist_assistant.providers.base import ProviderError

    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    app, api, _, _, _ = build_app(
        pages,
        openai=FakeProvider(error=ProviderError("down")),
        input_answers=["n", "q"],
    )
    app.run(RunOptions(interactive=True))
    assert api.updates == []


def test_interactive_feedback_plus_switch_model() -> None:
    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    oa = FakeProvider(default="oa")
    an = FakeProvider(default="claude")
    # 'fo' -> feedback prompt -> switch to opus -> regenerate via anthropic -> accept -> quit
    app, api, _, used_an, _ = build_app(
        pages, openai=oa, anthropic=an, input_answers=["fo", "tighten it", "y", "q"]
    )
    app.run(RunOptions(interactive=True))
    assert len(used_an.requests) == 1
    assert {"role": "user", "content": "tighten it"} in used_an.requests[0][1].messages
    assert api.updates[0]["description"].startswith("CLAUDE-OPUS-5-5 SUGGESTION:")


def test_interactive_invalid_choice_then_skip() -> None:
    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    app, api, _, _, out = build_app(pages, input_answers=["zzz", "n", "q"])
    app.run(RunOptions(interactive=True))
    assert api.updates == []
    assert any("Invalid choice" in line for line in out)


def test_interactive_paging_invalid_choice_then_quit() -> None:
    pages = [[make_task(id="a", due=date(2026, 6, 1))]]
    # accept the task, then an invalid paging choice, then quit
    app, api, _, _, out = build_app(pages, input_answers=["y", "bogus", "q"])
    app.run(RunOptions(interactive=True))
    assert [u["task_id"] for u in api.updates] == ["a"]
    assert any("Invalid choice" in line for line in out)


def test_update_all_with_no_matches_emits_its_own_message() -> None:
    pages = [[make_task(id="a", due=date(2026, 7, 1))]]  # future -> no match
    app, api, _, _, out = build_app(pages)
    app.run(RunOptions(update_all=True))
    assert api.updates == []
    assert any("No tasks to update" in line for line in out)


# helper double for inbox project
class FakeInbox:
    def __init__(self, id: str) -> None:
        self.id = id
        self.is_inbox_project = True
