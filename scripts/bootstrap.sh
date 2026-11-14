#!/usr/bin/env bash
# Set up a fresh checkout: install the locked dependencies and create config.ini.
# Safe to re-run. Never overwrites an existing config.ini.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv sync --locked

if [ -e config.ini ]; then
    echo "config.ini already exists; left unchanged."
else
    (umask 077 && cp config.ini.template config.ini)
    echo "Created config.ini (readable only by you). Add your API keys to it."
fi

echo "Ready. Try: uv run todoist-assistant --list-models"
