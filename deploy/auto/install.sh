#!/usr/bin/env bash
# Install the daily launchd agent that runs `avp auto` once a day. Portable — computes paths itself.
# Usage: ./install.sh [HOUR] [MINUTE]   (defaults 08:00 local; pick a time BEFORE your first post slot)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOUR="${1:-8}"; MINUTE="${2:-0}"
LABEL="com.astrostacker.auto"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/projects/_auto"
chmod +x "$ROOT/deploy/auto/run.sh"

cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>$ROOT/deploy/auto/run.sh</string></array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>
  <key>StandardOutPath</key><string>$ROOT/projects/_auto/launchd.out.log</string>
  <key>StandardErrorPath</key><string>$ROOT/projects/_auto/launchd.err.log</string>
  <key>RunAtLoad</key><false/>
</dict></plist>
PL

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
printf '✓ installed %s — runs daily at %02d:%02d local\n' "$LABEL" "$HOUR" "$MINUTE"
echo "  test now:  $ROOT/deploy/auto/run.sh"
echo "  logs:      $ROOT/projects/_auto/launchd.{out,err}.log"
echo "  uninstall: launchctl unload \"$PLIST\" && rm \"$PLIST\""
