# Todoist Assistant

An AI assistant for your Todoist tasks. It reads your tasks, asks an LLM for an
actionable suggestion on how to accomplish each one, and writes that suggestion
into the task's description.

Supports both **OpenAI (GPT-5 family)** and **Anthropic (Claude 4.x)** models,
selectable globally, per-project, or interactively per task.

## Important notes

- **⚠️ This tool modifies your Todoist tasks** by writing AI-generated suggestions
  into task descriptions. Try it on a throwaway task first.
- **💰 It calls paid LLM APIs.** You are billed by OpenAI / Anthropic per usage.
- **Not affiliated** with Todoist or Doist Inc. Independent, community project.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Todoist API key, plus an OpenAI and/or Anthropic API key

## Install

```bash
git clone https://github.com/briwilcox/todoist-llm-assistant.git
cd todoist-llm-assistant
uv sync            # creates .venv and installs pinned dependencies
```

## Configure

```bash
cp config.ini.template config.ini
```

Then edit `config.ini`:

```ini
[todoist]
api_key = YOUR_TODOIST_API_KEY

[openai]
api_key = YOUR_OPENAI_API_KEY

[anthropic]
api_key = YOUR_ANTHROPIC_API_KEY

[models]
preferred_model = gpt-5.4-mini

[project_models]
# Optional per-project overrides: project_id = model
# 2203306141 = gpt-5.5
# 2203306142 = claude-opus-4-8
```

- Todoist key: https://todoist.com/app/settings/integrations
- OpenAI key: https://platform.openai.com/api-keys
- Anthropic key: https://console.anthropic.com/

You only need a key for the provider(s) whose models you actually use. Keys may
also be supplied via environment variables (`TODOIST_API_KEY`, `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`), which take precedence over the file.

## Usage

```bash
# Suggest for the next batch of tasks due today or earlier
uv run todoist-assistant

# List supported models
uv run todoist-assistant --list-models

# Use a specific model
uv run todoist-assistant -m gpt-5.5
uv run todoist-assistant -m claude-opus-4-8

# Review each suggestion interactively (accept / feedback / switch model / skip)
uv run todoist-assistant -i

# Process every matching task at once
uv run todoist-assistant --update-all

# Include inbox and/or undated tasks
uv run todoist-assistant --inbox --no-due-date

# Dry run — show suggestions without writing to Todoist
uv run todoist-assistant --no-update
```

Key options: `-m/--model`, `-t/--tokens`, `-temp/--temperature` (ignored by
reasoning models), `--reasoning {low,medium,high}`, `--fallback` (fall back to
`gpt-5.4-mini` on empty/error responses), `--config PATH`. Run `--help` for the
full list.

### Model selection priority

1. `--model` on the command line
2. The task's project in `[project_models]`
3. `preferred_model` from `[models]`

### Supported models

| Provider  | Models                                                        |
|-----------|---------------------------------------------------------------|
| OpenAI    | `gpt-5.5` (reasoning), `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` |
| Anthropic | `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5`     |

Capabilities (which token parameter, whether `temperature` / `reasoning_effort`
apply) are declared per model in `src/todoist_assistant/models.py`. Adding a new
model is a one-line registry entry.

## Development

```bash
uv run pytest            # tests + coverage (fails under 90%)
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy src          # type-check (strict)
uv run pip-audit         # dependency vulnerability scan
uv run detect-secrets-hook --baseline .secrets.baseline $(git ls-files -- ':!uv.lock')  # secret scan
```

The codebase is structured for testability with dependency injection:

```
src/todoist_assistant/
  cli.py              # argument parsing + wiring
  config.py           # config loading (file + env)
  models.py           # model registry & capabilities
  providers/          # OpenAI + Anthropic provider implementations
  todoist_client.py   # Todoist v4 API wrapper (flattens pagination)
  task_filter.py      # pure task selection/sorting logic
  suggestion_engine.py# prompting, dispatch, fallback, formatting
  app.py              # batch + interactive orchestration
```

### Supply-chain hardening

- All dependencies are pinned exactly and locked in `uv.lock` (with hashes).
- `[tool.uv] exclude-newer = "14 days"` enforces a rolling **14-day dependency
  cooldown**, so a freshly-published (potentially compromised) release is never
  resolved until it has had time to be vetted.
- CI runs `uv lock --check`, `pip-audit`, and pins GitHub Actions by commit SHA.
- **Secret scanning** (`detect-secrets`) runs in pre-commit and CI, blocking any
  commit that introduces a credential. Known false-positives (template
  placeholders, fake test keys) are allowlisted in `.secrets.baseline`.

## License

MIT
