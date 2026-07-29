#!/usr/bin/env bash
# Deploy the control service to the always-on server (run FROM the Mac).
# Usage: ./deploy.sh [ssh-host] [dest-dir]   (defaults: titan-prod  /home/ste/svai-control)
set -euo pipefail

HOST="${1:-titan-prod}"
DEST="${2:-/home/ste/svai-control}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "→ syncing $DIR → $HOST:$DEST"
rsync -az --delete --exclude .env --exclude __pycache__ --exclude '*.pyc' "$DIR/" "$HOST:$DEST/"

ssh "$HOST" bash -s "$DEST" <<'EOS'
set -euo pipefail
DEST="$1"; cd "$DEST"
if [ ! -f .env ]; then
  { echo "CONTROL_TOKEN=$(openssl rand -hex 32)"; echo "POSTIZ_TOKEN="; } > .env
  chmod 600 .env
  echo "✓ generated .env (CONTROL_TOKEN set; POSTIZ_TOKEN empty until the server Postiz account exists)"
fi
docker compose up -d --build
docker compose ps
EOS
echo "✓ deployed. Control API on http://192.168.1.184:8770 (LAN)."
