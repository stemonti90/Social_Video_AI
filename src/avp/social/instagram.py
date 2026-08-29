"""Instagram Reels via the *Instagram API with Instagram Login*.

Why this flavour and not Facebook Login + Pages: Meta's use-case app model locks
``instagram_content_publish`` away from Facebook Login for Business — verified live on
2026-08-29, where the FLB configuration offered exactly two permissions (pages_show_list,
business_management) and nothing Instagram-shaped. The Instagram-Login product is the door
Meta actually leaves open: its own app id + secret, scopes ``instagram_business_basic`` +
``instagram_business_content_publish``, tokens on graph.instagram.com, and **no Facebook
Page required** (the Page link stays useful for Business Suite, but the API ignores it).

The flow, end to end:

  1. ``instagram.com/oauth/authorize``            → consent as the IG professional account
  2. ``POST api.instagram.com/oauth/access_token``→ short-lived token + user_id (~1 h)
  3. ``GET graph.instagram.com/access_token``     → 60-day token (ig_exchange_token)
  4. ``GET graph.instagram.com/refresh_access_token`` renews it (token must be >24 h old
     and still valid — a fully expired token means reconnecting by hand)
  5. publish: container (REELS + video_url) → poll status_code → media_publish

Instagram still fetches the video from a URL — no upload path — so `hosting.PublicCopy`
stays in the loop exactly as before.
"""
from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlencode

from ..log import get_logger
from .base import Publisher, app_credentials, http, redirect_uri
from .hosting import PublicCopy

log = get_logger("avp.social.instagram")

GRAPH = "https://graph.instagram.com/v23.0"


class Instagram(Publisher):
    platform = "instagram"
    scopes = ("instagram_business_basic", "instagram_business_content_publish")

    # ---------------------------------------------------------------- connecting
    def authorize_url(self, state: str, cfg=None) -> str:
        app_id, _ = app_credentials(self.platform, cfg)
        return "https://www.instagram.com/oauth/authorize?" + urlencode({
            "client_id": app_id,
            "redirect_uri": redirect_uri(self.platform),
            "scope": ",".join(self.scopes),
            "response_type": "code",
            "state": state,
            # NOT force_reauth: despite the name (and Meta's own copy-paste embed URL), it does
            # not merely re-show the consent screen — it throws the browser out to a full
            # username+password login even when a valid Instagram session already exists
            # (verified 2026-08-29). The consent screen appears anyway on a first grant.
        })

    def exchange(self, code: str, cfg=None) -> dict:
        app_id, secret = app_credentials(self.platform, cfg)
        short = http("POST", "https://api.instagram.com/oauth/access_token", data={
            "client_id": app_id, "client_secret": secret,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri(self.platform), "code": code})
        if not short.get("access_token"):
            raise RuntimeError(f"Instagram refused the code exchange: {short}")
        # Short-lived tokens die in ~an hour; trade up to the 60-day one immediately or the
        # first scheduled post tomorrow is already broken.
        long = http("GET", "https://graph.instagram.com/access_token", params={
            "grant_type": "ig_exchange_token", "client_secret": secret,
            "access_token": short["access_token"]})
        token = long.get("access_token") or short["access_token"]
        rec = {"access_token": token,
               "expires_in": long.get("expires_in", 3600),
               "ig_user_id": str(short.get("user_id", "")),
               "scope": ",".join(self.scopes)}
        try:
            me = http("GET", f"{GRAPH}/me", params={
                "fields": "user_id,username", "access_token": token})
            rec["account"] = f"@{me.get('username', rec['ig_user_id'])}"
            # /me's user_id is the professional-account id the publish endpoints want; the
            # exchange's user_id is app-scoped and usually — but not always — identical.
            rec["ig_user_id"] = str(me.get("user_id") or rec["ig_user_id"])
        except Exception as e:  # noqa: BLE001 — a label is not worth failing the connect
            log.debug("Could not read the Instagram username (%s)", e)
        return rec

    def refresh(self, record: dict, cfg=None) -> dict:
        # ig_refresh_token renews a still-valid long-lived token for another 60 days. It
        # refuses tokens younger than 24h (we never hit that: we refresh near expiry) and
        # tokens already expired (that needs a human re-consent — say so plainly).
        d = http("GET", "https://graph.instagram.com/refresh_access_token", params={
            "grant_type": "ig_refresh_token", "access_token": record.get("access_token", "")})
        if not d.get("access_token"):
            raise RuntimeError("The Instagram token could not be refreshed — run "
                               "`avp connect instagram` again.")
        return {**record, "access_token": d["access_token"],
                "expires_in": d.get("expires_in", 60 * 24 * 3600), "expires_at": None}

    # ---------------------------------------------------------------- posting
    def post(self, video: Path, caption: str, meta: dict, cfg, token: str,
             record: dict, disclose_ai: bool) -> dict:
        ig_id = record.get("ig_user_id")
        if not ig_id:
            raise RuntimeError("No Instagram account id stored — run `avp connect instagram` again.")

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

            # Publish while the copy is still up: Meta may re-read the source during publish.
            live = http("POST", f"{GRAPH}/{ig_id}/media_publish",
                        params={"creation_id": cid, "access_token": token})

        return {"container_id": cid, "post_id": live.get("id")}
