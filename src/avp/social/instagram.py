"""Instagram Reels via the Meta Graph API.

Flow (verified against a working client, 2026-08-29):

  1. POST /{ig-user-id}/media          media_type=REELS&video_url=…&caption=…  → creation_id
  2. GET  /{creation_id}?fields=status_code                                     → poll to FINISHED
  3. POST /{ig-user-id}/media_publish  creation_id=…                            → the live post

Notes that matter:

* **No file upload.** Step 1 takes a URL that Meta fetches; see `hosting.PublicCopy`.
* **Step 2 is not optional.** Meta returns a creation_id immediately and transcodes afterwards;
  publishing before the container reports FINISHED fails, and a container that goes to ERROR is the
  only place the real reason (bad codec, wrong aspect, too long) is ever reported.
* **The account must be a Business/Creator account linked to a Facebook Page.** Personal Instagram
  accounts cannot publish through the API at all — no scope fixes that.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from ..log import get_logger
from .base import Publisher, app_credentials, http, redirect_uri
from .hosting import PublicCopy

log = get_logger("avp.social.instagram")

GRAPH = "https://graph.facebook.com/v20.0"


class Instagram(Publisher):
    platform = "instagram"
    # instagram_content_publish is the one that actually posts; the others are what Meta requires to
    # discover which Page/IG account the token may act for.
    scopes = ("instagram_basic", "instagram_content_publish",
              "pages_show_list", "pages_read_engagement", "business_management")

    # ---------------------------------------------------------------- connecting
    def authorize_url(self, state: str, cfg=None) -> str:
        app_id, _ = app_credentials(self.platform, cfg)
        return "https://www.facebook.com/v20.0/dialog/oauth?" + urlencode({
            "client_id": app_id,
            "redirect_uri": redirect_uri(self.platform),
            "scope": ",".join(self.scopes),
            "response_type": "code",
            "state": state,
        })

    def exchange(self, code: str, cfg=None) -> dict:
        app_id, secret = app_credentials(self.platform, cfg)
        short = http("GET", f"{GRAPH}/oauth/access_token", params={
            "client_id": app_id, "client_secret": secret,
            "redirect_uri": redirect_uri(self.platform), "code": code})
        # Short-lived tokens die in ~1 hour. Exchange immediately for the 60-day long-lived one, or a
        # scheduled post tomorrow is already broken.
        long = http("GET", f"{GRAPH}/oauth/access_token", params={
            "grant_type": "fb_exchange_token", "client_id": app_id,
            "client_secret": secret, "fb_exchange_token": short["access_token"]})
        token = long["access_token"]

        pages = http("GET", f"{GRAPH}/me/accounts", params={
            "fields": "id,name,access_token,instagram_business_account{id,username}",
            "limit": 100, "access_token": token}).get("data", [])
        linked = [p for p in pages if p.get("instagram_business_account")]
        if not linked:
            raise RuntimeError(
                "No Instagram Business account is linked to any Facebook Page on this login. "
                "Convert the Instagram account to Business/Creator and link it to a Page, then retry.")
        page = linked[0]
        ig = page["instagram_business_account"]
        if len(linked) > 1:
            log.warning("Several linked accounts found — using @%s. Others: %s",
                        ig.get("username"), ", ".join(
                            p["instagram_business_account"].get("username", "?") for p in linked[1:]))
        return {
            # Publishing acts as the PAGE, so the page token is the one to keep — a user token gets
            # "(#200) requires instagram_content_publish" even when the scope was granted.
            "access_token": page.get("access_token") or token,
            "expires_in": long.get("expires_in", 60 * 24 * 3600),
            "ig_user_id": ig["id"], "page_id": page["id"],
            "account": f"@{ig.get('username', ig['id'])}",
            "scope": ",".join(self.scopes),
        }

    def refresh(self, record: dict, cfg=None) -> dict:
        # Page tokens derived from a long-lived user token do not expire on their own, but they die
        # when the password changes or the grant is revoked — both need a human, so say so plainly.
        raise RuntimeError("The Instagram token is no longer valid — run `avp connect instagram` again.")

    # ---------------------------------------------------------------- posting
    def post(self, video: Path, caption: str, meta: dict, cfg, token: str,
             record: dict, disclose_ai: bool) -> dict:
        ig_id = record.get("ig_user_id")
        if not ig_id:
            raise RuntimeError("No Instagram business id stored — run `avp connect instagram` again.")

        with PublicCopy(video, cfg) as pub:
            container = http("POST", f"{GRAPH}/{ig_id}/media", params={
                "media_type": "REELS", "video_url": pub.url,
                "caption": caption[:2200], "access_token": token})
            cid = container.get("id")
            if not cid:
                raise RuntimeError(f"Instagram did not return a container id: {container}")

            def status() -> dict:
                return http("GET", f"{GRAPH}/{cid}",
                            params={"fields": "status_code,status", "access_token": token})

            final = self.poll(status, lambda d: d.get("status_code") in ("FINISHED", "ERROR"),
                              what="Instagram container")
            if final.get("status_code") != "FINISHED":
                raise RuntimeError(f"Instagram could not process the video: "
                                   f"{final.get('status') or final}")

            # Publish while the copy is still up: Meta re-reads the source during publish.
            live = http("POST", f"{GRAPH}/{ig_id}/media_publish",
                        params={"creation_id": cid, "access_token": token})

        return {"container_id": cid, "post_id": live.get("id")}
