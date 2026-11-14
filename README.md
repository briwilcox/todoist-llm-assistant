# Todoist Assistant

An AI assistant for your Todoist tasks. It reads your tasks, asks an LLM for an
actionable suggestion on how to accomplish each one, and writes that suggestion
into the task's description.

Supports both **OpenAI (GPT-6 family)** and **Anthropic (Claude 5 family)** models,
selectable globally, per project, or interactively per task.

## Important notes

- **⚠️ This tool modifies your Todoist tasks** by writing AI-generated suggestions
  into task descriptions. Try it on a throwaway task first, or use `--no-update`.
- **💰 It calls paid LLM APIs.** You are billed by OpenAI / Anthropic per usage.
  Every supported model reasons before it answers, and reasoning tokens are billed
  as output tokens; `--reasoning low` keeps that small.
- **Not affiliated** with Todoist or Doist Inc. Independent, community project.

## Requirements

- Python 3.13 or newer (CI runs 3.13, 3.14 and 3.15)
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Todoist API key, plus an OpenAI and/or Anthropic API key

## Install

```bash
git clone https://github.com/briwilcox/todoist-llm-assistant.git
cd todoist-llm-assistant
scripts/bootstrap.sh   # uv sync --locked, and a config.ini from the template
```

`scripts/bootstrap.sh` installs the exact locked dependencies and, if you have no
`config.ini` yet, copies `config.ini.template` to `config.ini` (readable only by
you). It never overwrites an existing `config.ini`.

## Configure

Edit `config.ini`:

```ini
[todoist]
api_key = YOUR_TODOIST_API_KEY

[openai]
api_key = YOUR_OPENAI_API_KEY

[anthropic]
api_key = YOUR_ANTHROPIC_API_KEY

[models]
preferred_model = gpt-6-luna

[project_models]
# Optional per-project overrides: project_id = model (ids are case-sensitive)
# 6Xc7mPq2Wv9RhJ4f = gpt-6-astra
# 6Xc7mPq2Wv9RhJ4g = claude-opus-5-5
```

- Todoist key: https://app.todoist.com/app/settings/integrations (Developer tab)
- OpenAI key: https://platform.openai.com/api-keys
- Anthropic key: https://platform.claude.com/settings/keys

You only need a key for the provider(s) whose models you actually use. Keys may
also be supplied via environment variables (`TODOIST_API_KEY`, `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`), which take precedence over the file. `config.ini` is in
`.gitignore`; keep it out of commits.

## Usage

```bash
# Suggest for the next batch of tasks due today or earlier
uv run todoist-assistant

# List supported models
uv run todoist-assistant --list-models

# Use a specific model
uv run todoist-assistant -m gpt-6-astra
uv run todoist-assistant -m claude-opus-5-5

# Review each suggestion interactively (accept / feedback / switch model / skip)
uv run todoist-assistant -i

# Process every matching task at once
uv run todoist-assistant --update-all

# Include inbox and/or undated tasks
uv run todoist-assistant --inbox --no-due-date

# Dry run: show suggestions without writing to Todoist
uv run todoist-assistant --no-update
```

Key options:

- `-m/--model`: any model from `--list-models`.
- `--reasoning {none,low,medium,high,xhigh,max}` (default `medium`): how much the
  model thinks before answering. `none` works only on `gpt-6-luna` and
  `gpt-6.1-terra`; other models reject it with a clear error before any API call.
- `-temp/--temperature`: sent only with `--reasoning none` on the two models that
  accept it. The current Claude API rejects sampling parameters, so Claude models
  never receive it.
- `-t/--tokens`: output budget. It covers thinking as well as the reply. With no
  value, OpenAI requests carry no limit and Anthropic requests use 16,000 tokens
  (the Anthropic API requires one).
- `--fallback`: retry with `gpt-6-luna` on an empty reply, an API error, or a model
  whose provider has no key configured.
- `--config PATH`: read a config file other than `./config.ini`.

Run `--help` for the full list.

### Model selection priority

1. `--model` on the command line
2. The task's project in `[project_models]`
3. `preferred_model` from `[models]` (default `gpt-6-luna`)

### Supported models

| Provider  | Models | Reasoning effort |
|-----------|--------|------------------|
| OpenAI    | `gpt-6-luna` (default, smallest), `gpt-6.1-terra` | `none` to `max`; `temperature` only at `none` |
| OpenAI    | `gpt-6-astra` (flagship), `gpt-6.1-sol` | `low` to `max` |
| Anthropic | `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-2` | `low` to `max` |

Model IDs and capabilities come from the providers' model pages
([OpenAI](https://developers.openai.com/api/docs/models),
[Anthropic](https://platform.claude.com/docs/en/about-claude/models/overview)) and
are declared in `src/todoist_assistant/models.py`. Adding a model is a one-line
registry entry.

In interactive mode, `m`, `g`, `o` and `s` switch to `gpt-6-luna`, `gpt-6-astra`,
`claude-opus-5-5` and `claude-sonnet-5-5`; the menu lists the mapping.

## Development

```bash
uv run pytest            # unit, SDK-contract and smoke tests; fails under 95% coverage
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check src and tests (strict)
uv run pip-audit         # dependency vulnerability scan
uv run detect-secrets-hook --baseline .secrets.baseline $(git ls-files -- ':!uv.lock')  # secret scan
```

No test touches the network or needs a real key. Three layers:

- **Unit tests** use small doubles for the SDK clients and the Todoist API.
- **SDK-contract tests** (`tests/test_sdk_contract.py`) run the providers against
  the installed OpenAI and Anthropic SDKs with only the HTTP transport faked, so a
  parameter the SDK or API no longer accepts fails here, not in production.
- **Smoke tests** (`tests/test_smoke.py`) run the real `cli.main` end to end with
  the real Todoist, OpenAI and Anthropic SDK clients over in-memory HTTP servers.

The codebase is structured for testability with dependency injection:

```
src/todoist_assistant/
  cli.py               # argument parsing + wiring
  config.py            # config loading (file + env)
  models.py            # model registry & capabilities
  providers/           # OpenAI + Anthropic provider implementations
  todoist_client.py    # Todoist API v1 wrapper (flattens pagination)
  task_filter.py       # pure task selection/sorting logic
  suggestion_engine.py # prompting, dispatch, fallback, formatting
  app.py               # batch + interactive orchestration
```

### Supply-chain hardening

- All dependencies are pinned exactly and locked in `uv.lock` (with hashes).
- `[tool.uv] exclude-newer = "14 days"` enforces a rolling **14-day dependency
  cooldown**, so a freshly published (potentially compromised) release is never
  resolved until it has had time to be vetted.
- CI runs `uv lock --check`, `pip-audit`, and pins GitHub Actions by commit SHA.
- **Secret scanning** (`detect-secrets`) runs in pre-commit and CI, blocking any
  commit that introduces a credential. Known false positives (template
  placeholders, fake test keys) are allowlisted in `.secrets.baseline`.
- The pre-commit hooks are local and run through `uv run`, so they use the locked
  tool versions and download no hook code.

## License

MIT
