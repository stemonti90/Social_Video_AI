#!/usr/bin/env bash
# Persistent Mac GPU worker: polls the control server and renders any job that appears. Run by launchd
# (KeepAlive) or by hand. Reads the control URL + token from deploy/worker/.env (written by install.sh).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"

[ -f "$ROOT/deploy/worker/.env" ] && . "$ROOT/deploy/worker/.env"
: "${AVP_CONTROL_URL:?set AVP_CONTROL_URL (deploy/worker/.env)}"
: "${AVP_CONTROL_TOKEN:?set AVP_CONTROL_TOKEN (deploy/worker/.env)}"

# Generation needs Ollama; warn loudly if it isn't up (the worker keeps polling regardless).
if ! curl -s -m 4 -o /dev/null http://localhost:11434/api/tags 2>/dev/null; then
  echo "[warn] Ollama not reachable on :11434 — jobs will fail until it's running." >&2
fi

# Keep the Mac awake while we work. Measured 2026-08-29: an 8-segment build that takes ~25 minutes
# on a waking machine took SEVEN HOURS overnight — the laptop slept and the render only crept forward
# during brief DarkWake windows (segment 5→6 alone spanned 3h42m of wall clock for ~3 minutes of work).
# -i blocks idle sleep, -m keeps the disk spinning; both end when the command exits, so nothing stays
# pinned awake after the run. Closing the lid still sleeps the machine — that needs `caffeinate -d`
# plus an external power source, which is a decision for the operator, not a default.
CAFFEINATE=""
command -v caffeinate >/dev/null 2>&1 && CAFFEINATE="caffeinate -i -m"

exec $CAFFEINATE "$ROOT/.venv/bin/avp" worker --server "$AVP_CONTROL_URL" --token "$AVP_CONTROL_TOKEN" \
  --name "mac-m5" --poll 60 -v
