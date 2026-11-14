"""Application orchestration: select tasks, generate suggestions, update Todoist.

All side effects flow through injected collaborators (the Todoist client, the
suggestion engine, and ``input``/``output`` callables), so the entire control
flow — including the interactive menu — is unit-testable without real I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from .config import Config
from .models import resolve_model
from .suggestion_engine import Suggestion, SuggestionEngine
from .task_filter import select_tasks, window
from .todoist_client import TodoistClient
from .types import Message, TaskLike

BATCH_SIZE = 7

# Interactive single-key shortcuts -> model id.
_MODEL_KEYS = {
    "m": "gpt-5.4-mini",
    "g": "gpt-5.5",
    "o": "claude-opus-4-8",
    "s": "claude-sonnet-4-6",
}

_MENU = (
    "\nChoose an option:\n"
    "  y  Accept and update task\n"
    "  f  Provide feedback (same model)\n"
    "  m/g/o/s  Switch model (gpt-5.4-mini / gpt-5.5 / claude-opus-4-8 / claude-sonnet-4-6)\n"
    "  fm/fg/fo/fs  Feedback AND switch model\n"
    "  n  Skip this task\n"
    "Enter your choice: "
)


@dataclass
class RunOptions:
    interactive: bool = False
    update_all: bool = False
    include_inbox: bool = False
    include_no_date: bool = False
    model: str | None = None
    max_tokens: int | None = None
    temperature: float = 0.7
    reasoning_effort: str = "medium"
    enable_fallback: bool = False
    no_update: bool = False


class App:
    def __init__(
        self,
        todoist: TodoistClient,
        engine: SuggestionEngine,
        config: Config,
        *,
        today: date | None = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._todoist = todoist
        self._engine = engine
        self._config = config
        self._today = today or date.today()
        self._input = input_fn
        self._output = output_fn

    def run(self, opts: RunOptions) -> None:
        inbox_id = self._todoist.find_inbox_id() if opts.include_inbox else None
        offset = 0

        while True:
            selected = select_tasks(
                self._todoist.iter_all_tasks(),
                today=self._today,
                include_inbox=opts.include_inbox,
                include_no_date=opts.include_no_date,
                inbox_id=inbox_id,
            )
            batch = (
                selected if opts.update_all else window(selected, offset=offset, limit=BATCH_SIZE)
            )

            if not batch:
                self._emit_no_tasks(opts)
                return

            for task in batch:
                self._process_task(task, opts)

            if opts.update_all or not opts.interactive:
                return
            if not self._prompt_next():
                return
            offset += BATCH_SIZE

    def _process_task(self, task: TaskLike, opts: RunOptions) -> None:
        model = resolve_model(
            cli_model=opts.model,
            task_project_id=task.project_id,
            project_models=self._config.project_models,
            default_model=self._config.preferred_model,
        )
        if opts.interactive and not opts.update_all:
            self._interactive(task, model, opts)
        else:
            result = self._generate(task, model, opts)
            self._output(f"TASK: {task.content}")
            self._output(result.formatted)
            self._apply(task, result, opts)

    def _interactive(self, task: TaskLike, model: str, opts: RunOptions) -> None:
        conversation: list[Message] = [
            {
                "role": "user",
                "content": (
                    f"Task: {task.content}\n\n"
                    "Please provide a suggestion for how to accomplish this task."
                ),
            }
        ]
        current = model

        while True:
            result = self._generate(task, current, opts, conversation=conversation)
            self._output(f"TASK: {task.content}")
            self._output(result.formatted)
            if result.ok:
                conversation.append({"role": "assistant", "content": result.text})

            choice = self._input(_MENU).strip().lower()

            if choice in {"y", "yes", "1"}:
                self._apply(task, result, opts)
                return
            if choice in {"n", "no", "skip", "5"}:
                self._output("Skipping this task.")
                return
            if choice in _MODEL_KEYS:
                current = _MODEL_KEYS[choice]
                continue
            if choice.startswith("f"):
                switch = _MODEL_KEYS.get(choice[1:])
                feedback = self._input("Feedback: ")
                conversation.append({"role": "user", "content": feedback})
                if switch:
                    current = switch
                continue
            self._output("Invalid choice. Please try again.")

    def _generate(
        self,
        task: TaskLike,
        model: str,
        opts: RunOptions,
        conversation: list[Message] | None = None,
    ) -> Suggestion:
        return self._engine.generate(
            task_content=task.content,
            model=model,
            max_tokens=opts.max_tokens,
            temperature=opts.temperature,
            reasoning_effort=opts.reasoning_effort,
            enable_fallback=opts.enable_fallback,
            conversation=conversation,
        )

    def _apply(self, task: TaskLike, result: Suggestion, opts: RunOptions) -> None:
        if not result.ok:
            return
        if opts.no_update:
            self._output(f"Would update task '{task.content}' (demo mode, skipped)")
            return
        self._todoist.update_description(task.id, result.formatted)
        self._output(f"Updated task '{task.content}'")

    def _prompt_next(self) -> bool:
        while True:
            choice = (
                self._input("\n1) Advice on the next batch  2) Quit\nEnter your choice: ")
                .strip()
                .lower()
            )
            if choice in {"1", "n", "next"}:
                return True
            if choice in {"2", "q", "quit"}:
                self._output("Goodbye!")
                return False
            self._output("Invalid choice. Please try again.")

    def _emit_no_tasks(self, opts: RunOptions) -> None:
        if opts.update_all:
            self._output("No tasks to update — everything already has a suggestion.")
        else:
            self._output(
                "No tasks found that match your current filters. Try:\n"
                "  --inbox         include inbox tasks\n"
                "  --no-due-date   include tasks without a due date\n"
                "  or add tasks due today or earlier."
            )
