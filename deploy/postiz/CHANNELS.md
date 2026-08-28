# Connecting TikTok + Instagram to Postiz

Everything needed to take the pipeline from "builds videos" to "publishes them automatically".
Values below are already filled in for this deployment (`https://postiz.astrostackerpro.com`).

> **Read this first — the honest sequencing.** Both platforms review the REAL account behind the app.
> An account created yesterday with zero posts is a weak application. So: create the accounts, publish
> the first videos BY HAND (they're already built, in `~/Desktop/Social AstroStacker/`), and apply for
> API access once the channel looks alive. Waiting for the API before publishing wastes weeks.

## Phase 0 — accounts (only Stefano can do these)
Creating accounts and entering passwords is on you; the assistant can guide but never authenticates.

1. **TikTok** account for the channel.
2. **Instagram** account → convert to **Professional (Business)** in settings.
3. **Facebook Page**, then link it to the Instagram account (Meta's publishing API requires the pair).

## Phase 1 — TikTok app
Portal: <https://developers.tiktok.com> → Manage apps → Create an app.

| Field | Value |
|---|---|
| Redirect URI | `https://postiz.astrostackerpro.com/integrations/social/tiktok` |
| Products | **Login Kit** + **Content Posting API** (enable **Direct Post**) |
| Scopes | `user.info.basic`, `user.info.profile`, `video.create`, `video.publish`, `video.upload` |

HTTPS is mandatory — TikTok rejects `http://` redirect URIs. Ours is already HTTPS via the Cloudflare
tunnel, so this is satisfied.

**The audit gate (confirmed in Postiz's docs).** Until the app passes TikTok's audit:
- privacy is **forced to `SELF_ONLY`** regardless of what the pipeline sends,
- at most **5 users posting per 24h**, and the account must be private at post time.

So an unaudited app can only produce private/draft posts. Public automated posting REQUIRES the audit.
Budget days-to-weeks. Submitting needs: verified site URL, privacy policy, use-case description, and
usually a screencast of the integration working.

## Phase 2 — Meta app (Instagram)
Portal: <https://developers.facebook.com> → Create App → Business.

| Field | Value |
|---|---|
| Redirect URI | `https://postiz.astrostackerpro.com/integrations/social/instagram` |
| Products | **Instagram** + **Instagram Business Login** |
| Permissions | `instagram_basic`, `instagram_content_publish`, `instagram_manage_comments`, `instagram_manage_insights`, `pages_show_list`, `pages_read_engagement`, `business_management` |

`instagram_content_publish` is an advanced permission → **App Review** with a screencast showing the
flow. Requires the Business account ↔ Facebook Page link from Phase 0.

## Phase 3 — wire the credentials into Postiz
Put each app's id/secret in the server's Postiz override, then restart:

```bash
ssh titan-prod
nano /home/ste/postiz/docker-compose.override.yaml     # add under postiz: environment:
#   TIKTOK_CLIENT_ID / TIKTOK_CLIENT_SECRET
#   FACEBOOK_APP_ID  / FACEBOOK_APP_SECRET      (Instagram authenticates through the Meta app)
cd /home/ste/postiz && docker compose \
  --project-directory upstream -f upstream/docker-compose.yaml -f docker-compose.override.yaml up -d
```

Then open <https://postiz.astrostackerpro.com> → **Add channel** and complete the OAuth flow per
platform. Once a channel is connected, `avp auto` starts posting to it automatically — nothing else to
change: `connected_platforms()` picks it up, and until then videos are built and left ready.

## What the pipeline already does
- `publish.disclose_ai: true` → TikTok gets `video_made_with_ai` (the visuals are AI-generated).
- Posting times come from `auto.post_times` (12:00 / 18:00 / 21:00 Europe/Rome) — tune from analytics.
- Nothing is ever posted to a platform whose channel isn't connected.
