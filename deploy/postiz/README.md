# Postiz setup (social publishing for AVP)

[Postiz](https://postiz.com) is the self-hosted, open-source scheduler AVP posts through
(`src/avp/publish.py`). This folder brings up a local Postiz with one script.

> **Heads-up on resources.** The official stack is heavy: Postiz + Postgres + Redis **+ Temporal +
> Elasticsearch** (~7 containers). On a 24 GB Mac, **don't run a video build and Postiz at the same
> time** — bring Postiz up when you're about to publish, take it `down` when you're done.

## Prerequisites
- **Docker Desktop** running (`docker compose` v2), plus `git` and `openssl` (preinstalled on macOS).

## 1. Start it
```bash
deploy/postiz/setup.sh up
```
This clones the **official** upstream compose into `deploy/postiz/upstream/` (so you always get the
current version — Postiz changes between releases), generates a unique `JWT_SECRET`, and starts the
stack. First run pulls images and boots Temporal/Elasticsearch — give it a few minutes.

Then open **http://localhost:4007** and create your account (the first user is the admin).

## 2. Connect your channels
In the Postiz UI: **Add channel** → TikTok / Instagram / YouTube.

Each platform requires its **own developer OAuth app** — Postiz needs that app's client id/secret to
perform the OAuth handshake. Create the app on the platform's developer console, then put the
credentials in `deploy/postiz/docker-compose.override.yaml` (uncomment the relevant lines) and re-run
`setup.sh up`:

| Channel    | Override keys                              | Where to create the app |
|------------|--------------------------------------------|-------------------------|
| YouTube    | `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` | Google Cloud Console → OAuth (YouTube Data API v3) |
| TikTok     | `TIKTOK_CLIENT_ID` / `TIKTOK_CLIENT_SECRET`   | TikTok for Developers (Content Posting API) |
| Instagram  | `FACEBOOK_APP_ID` / `FACEBOOK_APP_SECRET`     | Meta for Developers (IG connects via a Facebook app) |

> **The real gating step.** Posting **publicly** also needs each platform's approval: TikTok's
> content-posting **audit** (unaudited apps can only post private/draft), Meta **app review** + an
> Instagram **Business/Creator** account linked to a Facebook Page, and YouTube OAuth consent. This is
> on the platforms, not on AVP — budget time for it.

## 3. Wire the API key into AVP
Postiz UI → **Settings → API** → generate a key. Put it in your `config.yaml`:
```yaml
publish:
  postiz_token: "PASTE_KEY_HERE"      # or: export AVP_POSTIZ_TOKEN=...
  # postiz_url defaults to http://localhost:4007/api — matches this setup
```
Channel ids are **auto-discovered** from the connected channels, so `publish.integrations` can stay
empty.

## 4. Publish
```bash
PYTHONPATH=src .venv/bin/avp publish <slug>                       # DRY RUN → publish_plan.json
PYTHONPATH=src .venv/bin/avp publish <slug> --go                 # post now
PYTHONPATH=src .venv/bin/avp publish <slug> --go --at 2026-07-01T18:00:00Z   # schedule (UTC ISO 8601)
```
Relevant `config.yaml publish` knobs: `platforms`, `privacy` (public/unlisted/private), `disclose_ai`
(→ TikTok `video_made_with_ai`), `made_for_kids`, `short_link`.

## Managing it
```bash
deploy/postiz/setup.sh status     # container status
deploy/postiz/setup.sh logs       # follow Postiz logs
deploy/postiz/setup.sh down       # stop (data persists in Docker volumes)
deploy/postiz/setup.sh update     # pull latest upstream compose + images, restart
```
Data lives in Docker named volumes (`postgres-volume`, `postiz-uploads`, …). To wipe everything:
`cd deploy/postiz/upstream && docker compose down -v`.

## Notes
- `upstream/`, `.jwt_secret`, and `docker-compose.override.yaml` are gitignored (the override holds your
  secret + credentials). The script + this README are the only committed files.
- Public-API rate limit is 30 requests/hour — fine for normal posting cadence.
