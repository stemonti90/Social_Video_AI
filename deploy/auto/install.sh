#!/usr/bin/env bash
# Install the daily launchd agent that runs `avp auto` once a day. Portable — computes paths itself.
# Usage: ./install.sh [HOUR] [MINUTE]   (defaults 08:00 local; pick a time BEFORE your first post slot)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Fire times as HH:MM arguments, one per video. Two runs a day beats one run making two videos,
# and the reason is the publisher: the native backend posts IMMEDIATELY (Instagram's API has no
# scheduled publish), so a single batch would push both videos out back to back. Spacing has to come
# from WHEN the pipeline runs. Generation takes ~25 min, so aim each run ~40 min before the slot.
TIMES=("${@:-12:20 18:20}")
[ $# -gt 0 ] && TIMES=("$@")
LABEL="com.astrostacker.auto"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/projects/_auto"
chmod +x "$ROOT/deploy/auto/run.sh"

INTERVALS=""
for t in "${TIMES[@]}"; do
  h="${t%%:*}"; m="${t##*:}"
  INTERVALS="$INTERVALS<dict><key>Hour</key><integer>$((10#$h))</integer><key>Minute</key><integer>$((10#$m))</integer></dict>"
done

# /bin/bash FIRST, the script as its argument — never the script as the program. launchd running a
# file that lives under ~/Desktop directly fails with "Operation not permitted" (exit 32256): the
# first scheduled run of the channel died that way and no video went out. The worker agent has run
# from this same folder for days, and the only difference was exactly this. Mirror it.
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$ROOT/deploy/auto/run.sh</string></array>
  <key>StartCalendarInterval</key><array>$INTERVALS</array>
  <key>StandardOutPath</key><string>$ROOT/projects/_auto/launchd.out.log</string>
  <key>StandardErrorPath</key><string>$ROOT/projects/_auto/launchd.err.log</string>
  <key>RunAtLoad</key><false/>
</dict></plist>
PL

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
printf '✓ installed %s — runs daily at %s local\n' "$LABEL" "${TIMES[*]}"
echo "  test now:  $ROOT/deploy/auto/run.sh"
echo "  logs:      $ROOT/projects/_auto/launchd.{out,err}.log"
echo "  uninstall: launchctl unload \"$PLIST\" && rm \"$PLIST\""
