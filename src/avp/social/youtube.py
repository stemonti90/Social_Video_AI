"""YouTube Shorts via the Data API v3 resumable upload.

  1. POST /upload/youtube/v3/videos?uploadType=resumable  metadata → Location: session URL
  2. PUT  session URL                                     the bytes → the created video

A #Shorts is just a normal upload that happens to be vertical and short — there is no separate Shorts
endpoint. YouTube classifies it from the aspect ratio and duration, which this pipeline already
produces (1080x1920, under 60s).

**Read this before promising public uploads:** while the Google Cloud project is unverified, the
``youtube.upload`` scope is capped — every upload lands as *private* and cannot be made public from the
API, no matter what ``privacyStatus`` says. Making that work needs Google's OAuth verification + a
YouTube API audit, which is a separate process from TikTok's. Until then this publisher is honest about
it: it reports the privacy YouTube actually applied, not the one requested.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import requests

from ..log import get_logger
from .base import TIMEOUT, Publisher, app_credentials, http, redirect_uri

log = get_logger("avp.social.youtube")

UPLOAD = "https://www.googleapis.com/upload/youtube/v3/videos"
TOKEN = "https://oauth2.googleapis.com/token"


class YouTube(Publisher):
    platform = "youtube"
    scopes = ("https://www.googleapis.com/auth/youtube.upload",)

    # ---------------------------------------------------------------- connecting
    def authorize_url(self, state: str, cfg=None) -> str:
        cid, _ = app_credentials(self.platform, cfg)
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
            "client_id": cid,
            "redirect_uri": redirect_uri(self.platform),
            "response_type": "code",
            "scope": " ".join(self.scopes),
            # Without BOTH of these Google returns no refresh token on re-consent, and the connection
            # silently becomes single-use — it works today and breaks tomorrow.
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        })

    def exchange(self, code: str, cfg=None) -> dict:
        cid, secret = app_credentials(self.platform, cfg)
        d = http("POST", TOKEN, data={
            "client_id": cid, "client_secret": secret, "code": code,
            "grant_type": "authorization_code", "redirect_uri": redirect_uri(self.platform)})
        if not d.get("refresh_token"):
            log.warning("Google returned no refresh token — revoke the app's access in your Google "
                        "account and reconnect, otherwise this link expires in an hour.")
        return {"access_token": d["access_token"], "refresh_token": d.get("refresh_token", ""),
                "expires_in": d.get("expires_in", 3600), "scope": d.get("scope", ""),
                "account": "YouTube channel"}

    def refresh(self, record: dict, cfg=None) -> dict:
        if not record.get("refresh_token"):
            raise RuntimeError("No Google refresh token — run `avp connect youtube` again.")
        cid, secret = app_credentials(self.platform, cfg)
        d = http("POST", TOKEN, data={"client_id": cid, "client_secret": secret,
                                      "grant_type": "refresh_token",
                                      "refresh_token": record["refresh_token"]})
        return {**record, "access_token": d["access_token"],
                "expires_in": d.get("expires_in", 3600), "expires_at": None}

    # ---------------------------------------------------------------- posting
    def post(self, video: Path, caption: str, meta: dict, cfg, token: str,
             record: dict, disclose_ai: bool) -> dict:
        yt = meta.get("youtube", {})
        # Parenthesised deliberately: `a or b if c else d` binds as `(a or b) if c else d`, which
        # threw away a perfectly good metadata title whenever the caption happened to be empty.
        first_line = next((l for l in (caption or "").splitlines() if l.strip()), "")
        title = ((yt.get("title") or first_line or "Untitled").strip())[:100]
        body = {
            "snippet": {"title": title or "Untitled",
                        "description": (yt.get("description") or caption)[:5000],
                        "tags": (yt.get("tags") or [])[:15],
                        "categoryId": "28"},          # 28 = Science & Technology
            "status": {"privacyStatus": getattr(cfg.publish, "privacy", "public"),
                       "selfDeclaredMadeForKids": bool(getattr(cfg.publish, "made_for_kids", False)),
                       # YouTube's own AI-disclosure surface for synthetic content.
                       "containsSyntheticMedia": bool(disclose_ai)},
        }
        size = video.stat().st_size
        start = requests.post(
            f"{UPLOAD}?{urlencode({'uploadType': 'resumable', 'part': 'snippet,status'})}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                     "X-Upload-Content-Length": str(size), "X-Upload-Content-Type": "video/mp4"},
            json=body, timeout=TIMEOUT)
        if start.status_code >= 400:
            raise RuntimeError(f"YouTube refused the upload session ({start.status_code}): "
                               f"{(start.text or '')[:400]}")
        session = start.headers.get("Location")
        if not session:
            raise RuntimeError("YouTube returned no resumable session URL.")

        with video.open("rb") as fh:
            r = requests.put(session, data=fh, timeout=(10, 900),
                             headers={"Content-Type": "video/mp4", "Content-Length": str(size)})
        if r.status_code >= 400:
            raise RuntimeError(f"YouTube upload failed ({r.status_code}): {(r.text or '')[:400]}")
        data = r.json()
        applied = data.get("status", {}).get("privacyStatus")
        if applied and applied != body["status"]["privacyStatus"]:
            log.warning("YouTube forced privacy to %r (asked for %r) — this is the unverified-app "
                        "cap; it lifts only after Google's OAuth verification and API audit.",
                        applied, body["status"]["privacyStatus"])
        return {"post_id": data.get("id"), "privacy": applied,
                "url": f"https://youtu.be/{data.get('id')}" if data.get("id") else None}
