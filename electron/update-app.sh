#!/usr/bin/env bash
# Rebuild the Electron app from source and reinstall it to /Applications, so the packaged app
# (app.asar) picks up code changes. Run after editing main.js / preload.js / renderer/*.
set -euo pipefail
cd "$(dirname "$0")"                       # electron/
APP="Social AstroStacker App.app"
DEST="/Applications/$APP"

echo "▶ Rebuilding (electron-builder, unsigned, --dir)…"
CSC_IDENTITY_AUTO_DISCOVERY=false ./node_modules/.bin/electron-builder --mac --dir >/tmp/avp-build.log 2>&1 \
  || { echo "✗ build failed — see /tmp/avp-build.log"; tail -5 /tmp/avp-build.log; exit 1; }

echo "▶ Quitting the running app…"
osascript -e "tell application \"$APP\" to quit" 2>/dev/null || true
sleep 2; pkill -f "$APP/Contents/MacOS" 2>/dev/null || true; sleep 1

echo "▶ Installing to /Applications…"
rm -rf "$DEST"
cp -R "dist/mac-arm64/$APP" /Applications/
xattr -cr "$DEST" 2>/dev/null || true              # strip detritus so ad-hoc signing succeeds
codesign --force --deep --sign - "$DEST" >/dev/null 2>&1 || true

echo "▶ Launching…"
open "$DEST"
echo "✓ Updated: $DEST"
