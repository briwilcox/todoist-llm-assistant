"""Command-line entry point: parse args, wire dependencies, run the app."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import date

from .app import App, RunOptions
from .config import DEFAULT_MODEL, Config, ConfigError, load_config
from .models import Provider, UnknownModelError, available_models, get_model_spec
from .providers.anthropic_provider import AnthropicProvider
from .providers.base import SuggestionProvider
from .providers.openai_provider import OpenAIProvider
from .suggestion_engine import SuggestionEngine
from .todoist_client import TodoistClient

DEFAULT_TEMPERATURE = 0.7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="todoist-assistant",
        description="Generate actionable AI suggestions for your Todoist tasks.",
    )
    parser.add_argument("--config", help="Path to the config INI (default: config.ini)")
    parser.add_argument("--list-models", action="store_true", help="List supported models and exit")
    parser.add_argument(
        "-u", "--update-all", action="store_true", help="Process all matching tasks (no batching)"
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Review each suggestion")
    parser.add_argument("--inbox", action="store_true", help="Include inbox tasks")
    parser.add_argument(
        "--no-due-date", action="store_true", help="Include tasks without a due date"
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=available_models(),
        default=None,
        help="Model to use (overrides config and project mappings)",
    )
    parser.add_argument(
        "-t", "--tokens", type=int, default=None, help="Max output tokens (default: no limit)"
    )
    parser.add_argument(
        "-temp",
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature 0.0-2.0 (ignored by reasoning models)",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Demo mode: show suggestions without writing to Todoist",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Fall back to the default model on empty responses or errors",
    )
    parser.add_argument(
        "--reasoning",
        choices=["low", "medium", "high"],
        default="medium",
        help="Reasoning effort for reasoning models",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_temperature(value: float) -> float:
    """Clamp out-of-range temperatures back to the default."""
    if value < 0.0 or value > 2.0:
        return DEFAULT_TEMPERATURE
    return value


def run_options_from_args(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        interactive=args.interactive,
        update_all=args.update_all,
        include_inbox=args.inbox,
        include_no_date=args.no_due_date,
        model=args.model,
        max_tokens=args.tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning,
        enable_fallback=args.fallback,
        no_update=args.no_update,
    )


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_models:
        print("Supported models:")
        for name in available_models():
            print(f"  {name}")
        return 0

    try:
        config = load_config(args.config, env=env)
    except ConfigError as exc:
        print(str(exc))
        return 1

    if validate_temperature(args.temperature) != args.temperature:
        print(f"Temperature must be between 0.0 and 2.0; using default {DEFAULT_TEMPERATURE}.")
        args.temperature = DEFAULT_TEMPERATURE

    providers = _build_providers(config)

    primary_model = args.model or config.preferred_model
    try:
        spec = get_model_spec(primary_model)
    except UnknownModelError as exc:
        print(str(exc))
        return 1
    if spec.provider not in providers:
        print(
            f"No API key configured for the '{spec.provider.value}' provider, "
            f"which model '{primary_model}' requires. Add it to your config or environment."
        )
        return 1

    todoist = _build_todoist(config)
    try:
        todoist.check_connection()
    except Exception as exc:
        print(f"Error connecting to Todoist API: {exc}\nCheck your Todoist API key.")
        return 1

    engine = SuggestionEngine(providers, fallback_model=DEFAULT_MODEL)
    app = App(todoist, engine, config, today=_today())
    app.run(run_options_from_args(args))
    return 0


def _build_providers(config: Config) -> dict[Provider, SuggestionProvider]:
    """Construct providers for whichever API keys are configured."""
    providers: dict[Provider, SuggestionProvider] = {}
    if config.openai_api_key:
        from openai import OpenAI

        providers[Provider.OPENAI] = OpenAIProvider(OpenAI(api_key=config.openai_api_key))
    if config.anthropic_api_key:
        from anthropic import Anthropic

        providers[Provider.ANTHROPIC] = AnthropicProvider(
            Anthropic(api_key=config.anthropic_api_key)
        )
    return providers


def _build_todoist(config: Config) -> TodoistClient:
    from todoist_api_python.api import TodoistAPI

    return TodoistClient(TodoistAPI(config.todoist_api_key))


def _today() -> date:
    return date.today()
