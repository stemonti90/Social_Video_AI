#!/usr/bin/env bash
# Bring up a local Postiz (self-hosted social scheduler) for AVP publishing.
#
# Postiz's docs warn against copying a stale compose snapshot, so this pulls the OFFICIAL upstream
# compose into ./upstream and layers a local override that injects a generated JWT secret (and, when
# you fill them in, your platform OAuth credentials). The public API then lives at
# http://localhost:4007/api — which is what config.yaml `publish.postiz_url` already points to.
#
# Usage: ./setup.sh [up|down|logs|status|update]
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="$DIR/upstream"
REPO="https://github.com/gitroomhq/postiz-docker-compose.git"
OVERRIDE="$DIR/docker-compose.override.yaml"
SECRET_FILE="$DIR/.jwt_secret"
COMPOSE=(docker compose --project-directory "$UPSTREAM" -f "$UPSTREAM/docker-compose.yaml" -f "$OVERRIDE")

need() { command -v "$1" >/dev/null 2>&1 || { echo "✗ '$1' not found — install it first."; exit 1; }; }
need docker; need git; need openssl
docker compose version >/dev/null 2>&1 || { echo "✗ 'docker compose' (v2) not available."; exit 1; }

clone_or_update() {
  if [ -d "$UPSTREAM/.git" ]; then
    echo "↻ updating upstream compose…"; git -C "$UPSTREAM" pull --ff-only
  else
    echo "⇣ cloning official Postiz compose…"; git clone --depth 1 "$REPO" "$UPSTREAM"
  fi
}

ensure_override() {
  if [ ! -f "$SECRET_FILE" ]; then
    ( umask 077; openssl rand -hex 48 > "$SECRET_FILE" ); echo "✓ generated a JWT secret (kept in .jwt_secret)"
  fi
  if [ ! -f "$OVERRIDE" ]; then
    local secret; secret="$(cat "$SECRET_FILE")"
    cat > "$OVERRIDE" <<YAML
# Local overrides for AVP — GITIGNORED (holds your JWT secret).
# To CONNECT channels you must create a developer OAuth app on each platform and paste its
# credentials here, then re-run \`./setup.sh up\` (see README → "Connect your channels").
services:
  postiz:
    environment:
      JWT_SECRET: "$secret"
      # --- uncomment + fill to connect channels ---
      # YOUTUBE_CLIENT_ID: ""
      # YOUTUBE_CLIENT_SECRET: ""
      # TIKTOK_CLIENT_ID: ""
      # TIKTOK_CLIENT_SECRET: ""
      # FACEBOOK_APP_ID: ""          # Instagram connects through a Meta/Facebook app
      # FACEBOOK_APP_SECRET: ""
YAML
    echo "✓ wrote $(basename "$OVERRIDE")"
  fi
}

case "${1:-up}" in
  up)
    clone_or_update; ensure_override
    echo "▶ starting Postiz (first run pulls images + boots Temporal/Elasticsearch — a few minutes)…"
    "${COMPOSE[@]}" up -d
    cat <<'NEXT'

✓ Postiz is starting. Next steps:
  1) wait ~1-2 min, then open  http://localhost:4007  and create your account
  2) connect TikTok / Instagram / YouTube (each needs its own OAuth app — see README)
  3) Settings → API → generate a key → put it in config.yaml  publish.postiz_token
  4) test it:
       PYTHONPATH=src .venv/bin/avp publish <slug>       # dry run (writes publish_plan.json)
       PYTHONPATH=src .venv/bin/avp publish <slug> --go   # live
NEXT
    ;;
  down|logs|status)
    [ -d "$UPSTREAM" ] && [ -f "$OVERRIDE" ] || { echo "Run './setup.sh up' first."; exit 1; }
    case "$1" in
      down)   "${COMPOSE[@]}" down ;;
      logs)   "${COMPOSE[@]}" logs -f --tail=100 postiz ;;
      status) "${COMPOSE[@]}" ps ;;
    esac ;;
  update)
    clone_or_update; ensure_override
    "${COMPOSE[@]}" pull; "${COMPOSE[@]}" up -d ;;
  *)
    echo "usage: $0 [up|down|logs|status|update]"; exit 2 ;;
esac
