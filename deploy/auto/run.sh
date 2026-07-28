#!/usr/bin/env bash
# Run one daily batch (`avp auto`) with the environment the pipeline needs. Called by launchd, or by
# you to test. Path-portable: it resolves the project root from its own location.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"
mkdir -p "$ROOT/projects/_auto"

# Best-effort: bring up the runtime deps. colima hosts Postiz (containers restart:always once it's up).
if command -v colima >/dev/null 2>&1; then
  colima status >/dev/null 2>&1 || colima start >/dev/null 2>&1 || true
fi
# Ollama must be reachable for script/metadata/topic generation — warn loudly if not.
if ! curl -s -m 4 -o /dev/null http://localhost:11434/api/tags 2>/dev/null; then
  echo "[warn] Ollama not reachable on :11434 — generation will fail. Start Ollama first." >&2
fi

exec "$ROOT/.venv/bin/avp" auto --config "$ROOT/config.yaml" -v
