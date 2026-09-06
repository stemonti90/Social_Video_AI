"""Upload-Post (upload-post.com): a third party whose OWN TikTok app has passed TikTok's audit.

Why it exists here. TikTok refused our app for production — "does not support personal or internal
company use" — and an unaudited app may only post SELF_ONLY. Upload-Post lets the account owner
authorise *their* approved app once (OAuth on their site) and then exposes one multipart endpoint
for many networks. We use it for TikTok (public posts) and Reddit. Instagram stays native.

Contract (from docs.upload-post.com/api/upload-video, read 2026-09-06):
  POST https://api.upload-post.com/api/upload     Authorization: Apikey <key>
  fields: video (file), title, user (profile id), platform[] (repeatable)
  TikTok: tiktok_title, privacy_level, post_mode (DIRECT_POST|MEDIA_UPLOAD), disable_comment/duet/
          stitch, is_aigc, brand_content_toggle, brand_organic_toggle
  Reddit: subreddit (required, no "r/"), reddit_title, flair_id
  response: {"success": bool, "results": {"<platform>": {"success", "url", "post_id", "error"}}}
  limits: TikTok 6 posts/min, 15/day per account; 3-600 s; up to 4 GB.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

from ..log import get_logger

log = get_logger("avp.social.uploadpost")

API = "https://api.upload-post.com/api/upload"
TIMEOUT = (30, 900)          # connect, read: a 30 MB upload on a slow link takes a while


def settings(cfg) -> dict:
    """The uploadpost block of publish config, with the key taken from the environment first."""
    s = dict(getattr(cfg.publish, "uploadpost", None) or {})
    s["api_key"] = os.environ.get("AVP_UPLOADPOST_KEY") or s.get("api_key") or ""
    return s


def configured(cfg) -> bool:
    return bool(settings(cfg).get("api_key")) and bool(settings(cfg).get("user"))


def post(platform: str, video: Path, caption: str, meta: dict, cfg, disclose_ai: bool = False,
         title: str | None = None) -> dict:
    """Publish one video to one platform through Upload-Post. Raises with the API's own message."""
    s = settings(cfg)
    if not s.get("api_key") or not s.get("user"):
        raise RuntimeError("Upload-Post is not configured: set publish.uploadpost.api_key (or "
                           "AVP_UPLOADPOST_KEY) and publish.uploadpost.user.")
    plat = platform.lower().strip()
    data: list[tuple[str, str]] = [("user", str(s["user"])), ("platform[]", plat),
                                   ("title", (title or caption.splitlines()[0] if caption else "")[:300])]
    if plat == "tiktok":
        privacy = {"public": "PUBLIC_TO_EVERYONE", "unlisted": "FOLLOWER_OF_CREATOR",
                   "private": "SELF_ONLY"}.get(str(s.get("privacy") or cfg.publish.privacy or "public"),
                                                "PUBLIC_TO_EVERYONE")
        data += [("tiktok_title", caption[:2200]), ("privacy_level", privacy),
                 ("post_mode", str(s.get("tiktok_post_mode") or "DIRECT_POST")),
                 ("disable_comment", "false"), ("disable_duet", "false"), ("disable_stitch", "false"),
                 ("is_aigc", "true" if disclose_ai else "false")]
    elif plat == "reddit":
        sub = str(s.get("subreddit") or "").strip().lstrip("r/")
        if not sub:
            raise RuntimeError("Upload-Post/Reddit needs publish.uploadpost.subreddit.")
        data += [("subreddit", sub), ("reddit_title", (title or caption.splitlines()[0] if caption else "")[:300])]
        if s.get("flair_id"):
            data.append(("flair_id", str(s["flair_id"])))
    else:
        data.append((f"{plat}_title", caption[:2200]))
    log.info("Upload-Post → %s as user %r (%.1f MB)", plat, s["user"], video.stat().st_size / 1e6)
    with video.open("rb") as fh:
        r = requests.post(API, headers={"Authorization": f"Apikey {s['api_key']}"},
                          data=data, files={"video": (video.name, fh, "video/mp4")}, timeout=TIMEOUT)
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[:500]}
    if r.status_code >= 400 or not body.get("success", False):
        raise RuntimeError(f"Upload-Post {r.status_code}: {str(body)[:400]}")
    res = (body.get("results") or {}).get(plat) or {}
    if not res.get("success", True):
        raise RuntimeError(f"Upload-Post/{plat}: {res.get('error') or res}")
    out = {"via": "uploadpost", "post_id": res.get("post_id"), "url": res.get("url"),
           "usage": body.get("usage")}
    log.info("Upload-Post/%s: published (%s)", plat, out["url"] or out["post_id"] or "ok")
    return out
