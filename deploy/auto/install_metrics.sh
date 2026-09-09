#!/usr/bin/env bash
# Install the launchd agent that measures: a metrics snapshot every night (23:40) and the weekly report
# on Sunday evening (21:00). Both are `avp report`; the report also snapshots first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LABEL="com.astrostacker.metrics"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/projects/_auto"
cat > "$ROOT/deploy/auto/metrics.sh" <<SH
#!/usr/bin/env bash
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\${PATH:-}"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"
if [ "\$(date +%u)" = "7" ] && [ "\$(date +%H)" -ge 20 ]; then
  "$ROOT/.venv/bin/avp" report --days 7 --config "$ROOT/config.yaml"
else
  "$ROOT/.venv/bin/avp" report --snapshot --config "$ROOT/config.yaml"
fi
SH
chmod +x "$ROOT/deploy/auto/metrics.sh"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$ROOT/deploy/auto/metrics.sh</string></array>
  <key>StartCalendarInterval</key><array>
    <dict><key>Hour</key><integer>23</integer><key>Minute</key><integer>40</integer></dict>
    <dict><key>Weekday</key><integer>0</integer><key>Hour</key><integer>21</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$ROOT/projects/_auto/metrics.out.log</string>
  <key>StandardErrorPath</key><string>$ROOT/projects/_auto/metrics.err.log</string>
  <key>RunAtLoad</key><false/>
</dict></plist>
PL
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "✓ installed $LABEL — snapshot 23:40 daily, report Sundays 21:00 → $ROOT/projects/_auto/report-*.md"
