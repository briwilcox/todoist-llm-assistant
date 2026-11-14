"""Command-line entry point: parse args, wire dependencies, run the app."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from .app import App, RunOptions
from .config import DEFAULT_MODEL, Config, ConfigError, load_config
from .models import (
    DEFAULT_EFFORT,
    EFFORT_LEVELS,
    Provider,
    UnknownModelError,
    available_models,
    get_model_spec,
)
from .providers.anthropic_provider import AnthropicProvider
from .providers.base import SuggestionProvider, missing_key_message
from .providers.openai_provider import OpenAIProvider
from .suggestion_engine import SuggestionEngine
from .todoist_client import TodoistClient

if TYPE_CHECKING:
    import httpx
    import httpx2

DEFAULT_TEMPERATURE = 0.7


@dataclass(frozen=True)
class HttpClients:
    """HTTP clients handed to the SDKs instead of their defaults.

    For proxies, custom TLS, or tests that replace the network. The OpenAI and
    Anthropic SDKs take an ``httpx2.Client``; the Todoist SDK takes an ``httpx.Client``.
    """

    openai: httpx2.Client | None = None
    anthropic: httpx2.Client | None = None
    todoist: httpx.Client | None = None


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
        help=(
            "Sampling temperature 0.0-2.0. Sent only with --reasoning none on models "
            "that accept it (gpt-6-luna, gpt-6.1-terra); ignored otherwise"
        ),
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
        choices=EFFORT_LEVELS,
        default=DEFAULT_EFFORT,
        help="Reasoning effort (default: %(default)s). 'none' works only on gpt-6-luna "
        "and gpt-6.1-terra",
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


def main(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    http: HttpClients | None = None,
) -> int:
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

    providers = _build_providers(config, http)

    primary_model = args.model or config.preferred_model
    try:
        spec = get_model_spec(primary_model)
    except UnknownModelError as exc:
        print(str(exc))
        return 1
    if spec.provider not in providers:
        print(
            f"{missing_key_message(spec.provider, primary_model)} "
            "Add it to your config or environment."
        )
        return 1
    effort_error = spec.effort_error(args.reasoning)
    if effort_error is not None:
        print(effort_error)
        return 1

    todoist = _build_todoist(config, http)
    try:
        todoist.check_connection()
    except Exception as exc:
        print(f"Error connecting to Todoist API: {exc}\nCheck your Todoist API key.")
        return 1

    engine = SuggestionEngine(providers, fallback_model=DEFAULT_MODEL)
    app = App(todoist, engine, config, today=_today())
    app.run(run_options_from_args(args))
    return 0


def _build_providers(
    config: Config, http: HttpClients | None = None
) -> dict[Provider, SuggestionProvider]:
    """Construct providers for whichever API keys are configured."""
    http = http or HttpClients()
    providers: dict[Provider, SuggestionProvider] = {}
    if config.openai_api_key:
        from openai import OpenAI

        providers[Provider.OPENAI] = OpenAIProvider(
            OpenAI(api_key=config.openai_api_key, http_client=http.openai)
        )
    if config.anthropic_api_key:
        from anthropic import Anthropic

        providers[Provider.ANTHROPIC] = AnthropicProvider(
            Anthropic(api_key=config.anthropic_api_key, http_client=http.anthropic)
        )
    return providers


def _build_todoist(config: Config, http: HttpClients | None = None) -> TodoistClient:
    from todoist_api_python.api import TodoistAPI

    client = http.todoist if http is not None else None
    return TodoistClient(TodoistAPI(config.todoist_api_key, client=client))


def _today() -> date:
    return date.today()
