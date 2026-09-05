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

# Keep the Mac awake while we work. Measured 2026-08-29: an 8-segment build that takes ~25 minutes
# on a waking machine took SEVEN HOURS overnight — the laptop slept and the render only crept forward
# during brief DarkWake windows (segment 5→6 alone spanned 3h42m of wall clock for ~3 minutes of work).
# -i blocks idle sleep, -m keeps the disk spinning; both end when the command exits, so nothing stays
# pinned awake after the run. Closing the lid still sleeps the machine unless it is on AC power AND
# -s is set (see below); on battery, a closed lid ends the build — nothing here can prevent that.
CAFFEINATE=""
# -s as well as -i -m: a build died at 07:39 with no error because the lid was closed mid-footage.
# -i/-m only stop IDLE and DISK sleep; -s also holds off system sleep while on AC power, which is
# the case for a machine left running its daily job. It cannot stop a forced sleep, and does not
# wake a sleeping Mac — that still needs `pmset repeat wakeorpoweron`.
command -v caffeinate >/dev/null 2>&1 && CAFFEINATE="caffeinate -i -m -s"

exec $CAFFEINATE "$ROOT/.venv/bin/avp" auto --config "$ROOT/config.yaml" -v
