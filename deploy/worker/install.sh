#!/usr/bin/env bash
# Install the always-on GPU worker as a launchd agent on this Mac. It fetches the control token from
# the server over SSH, so run it after the control service is deployed.
# Usage: ./install.sh [ssh-host] [control-url]   (defaults: titan-prod  http://192.168.1.184:8770)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${1:-titan-prod}"
URL="${2:-http://192.168.1.184:8770}"
LABEL="com.astrostacker.worker"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

echo "→ fetching control token from $HOST"
TOKEN="$(ssh "$HOST" "grep '^CONTROL_TOKEN=' /home/ste/svai-control/.env | cut -d= -f2")"
[ -n "$TOKEN" ] || { echo "✗ couldn't read CONTROL_TOKEN from the server"; exit 1; }

umask 077
{ echo "AVP_CONTROL_URL=$URL"; echo "AVP_CONTROL_TOKEN=$TOKEN"; } > "$ROOT/deploy/worker/.env"
chmod +x "$ROOT/deploy/worker/run-worker.sh"
mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/projects/_auto"

cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$ROOT/deploy/worker/run-worker.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$ROOT/projects/_auto/worker.out.log</string>
  <key>StandardErrorPath</key><string>$ROOT/projects/_auto/worker.err.log</string>
</dict></plist>
PL

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "✓ worker installed & started ($LABEL) — control $URL"
echo "  logs:      $ROOT/projects/_auto/worker.{out,err}.log"
echo "  stop:      launchctl unload \"$PLIST\""
