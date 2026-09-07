#!/usr/bin/env bash
# Install the launchd agent that publishes the TikTok back catalogue one video per slot (backfill.sh).
# Usage: ./install_backfill.sh HH:MM [HH:MM ...]   e.g. 12:50 14:50 16:50 18:50 20:50
# The agent is harmless once the queue is empty (it logs "queue empty" and exits); uninstall with
# the line printed at the end.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ $# -gt 0 ] || { echo "usage: $0 HH:MM [HH:MM ...]" >&2; exit 2; }
LABEL="com.astrostacker.backfill"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/projects/_auto"
chmod +x "$ROOT/deploy/auto/backfill.sh"

INTERVALS=""
for t in "$@"; do
  h="${t%%:*}"; m="${t##*:}"
  INTERVALS="$INTERVALS<dict><key>Hour</key><integer>$((10#$h))</integer><key>Minute</key><integer>$((10#$m))</integer></dict>"
done

# /bin/bash first, the script as its argument — see install.sh for why (exit 32256 otherwise).
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$ROOT/deploy/auto/backfill.sh</string></array>
  <key>StartCalendarInterval</key><array>$INTERVALS</array>
  <key>StandardOutPath</key><string>$ROOT/projects/_auto/backfill.launchd.log</string>
  <key>StandardErrorPath</key><string>$ROOT/projects/_auto/backfill.launchd.log</string>
  <key>RunAtLoad</key><false/>
</dict></plist>
PL

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
printf '✓ installed %s — one TikTok post per slot at %s local\n' "$LABEL" "$*"
echo "  queue:     $ROOT/projects/_auto/tiktok_backfill.txt (one slug per line)"
echo "  log:       $ROOT/projects/_auto/backfill.log"
echo "  uninstall: launchctl unload \"$PLIST\" && rm \"$PLIST\""
