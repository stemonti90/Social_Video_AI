"""Slim, dependency-free Postiz public-API client for the control service (server-side).

Mirrors the verified contract in ../src/avp/publish.py (docs.postiz.com/public-api, 2026-06) but uses
only the stdlib (urllib) so the control container stays tiny. Keep the two in sync if Postiz changes.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone

_ALIASES = {
    "youtube": "youtube", "yt": "youtube", "shorts": "youtube",
    "tiktok": "tiktok", "tt": "tiktok",
    "instagram": "instagram", "ig": "instagram", "reels": "instagram",
}


def canon(platform: str) -> str:
    return _ALIASES.get((platform or "").lower().strip(), (platform or "").lower().strip())


def now_iso(plus_seconds: int = 60) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=plus_seconds)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def caption_for(platform: str, meta: dict) -> str:
    p = canon(platform)
    if p == "youtube":
        yt = meta.get("youtube", {}) or {}
        return f"{yt.get('title', '')}\n\n{yt.get('description', '')}".strip()
    if p == "tiktok":
        return (meta.get("tiktok", {}) or {}).get("caption", "")
    if p == "instagram":
        return (meta.get("instagram", {}) or {}).get("caption", "")
    return (meta.get("tiktok", {}) or {}).get("caption", "")


def _title_for(platform: str, meta: dict) -> str:
    yt_title = ((meta.get("youtube", {}) or {}).get("title") or "").strip()
    if yt_title:
        return yt_title
    cap = caption_for(platform, meta) or caption_for("tiktok", meta)
    return (cap.splitlines() or [""])[0].strip() or "Untitled"


def settings_for(platform: str, meta: dict, disclose_ai: bool = False, privacy: str = "public",
                 made_for_kids: bool = False) -> dict:
    """The provider-specific Postiz `settings` object required by POST /public/v1/posts."""
    p = canon(platform)
    if p == "youtube":
        yt = meta.get("youtube", {}) or {}
        vis = privacy if privacy in ("public", "unlisted", "private") else "public"
        tags = [{"value": t, "label": t} for t in (yt.get("tags") or [])][:15]
        title = (_title_for("youtube", meta))[:100] or "Untitled"
        if len(title) < 2:
            title = "Untitled"
        return {"__type": "youtube", "title": title, "type": vis,
                "selfDeclaredMadeForKids": "yes" if made_for_kids else "no",
                "thumbnail": None, "tags": tags}
    if p == "tiktok":
        pl = {"public": "PUBLIC_TO_EVERYONE", "unlisted": "FOLLOWER_OF_CREATOR",
              "private": "SELF_ONLY"}.get(privacy, "PUBLIC_TO_EVERYONE")
        return {"__type": "tiktok", "title": _title_for("tiktok", meta)[:90],
                "privacy_level": pl, "duet": False, "stitch": False, "comment": True,
                "autoAddMusic": "no", "brand_content_toggle": False, "brand_organic_toggle": False,
                "video_made_with_ai": bool(disclose_ai), "content_posting_method": "DIRECT_POST"}
    if p == "instagram":
        return {"__type": "instagram", "post_type": "post"}
    return {"__type": p}


class PostizError(Exception):
    pass


class PostizClient:
    def __init__(self, base: str, token: str):
        self.base = (base or "").rstrip("/")
        self.token = token or ""

    def _request(self, path: str, method: str = "GET", body: bytes | None = None,
                 content_type: str | None = None, timeout: int = 120):
        req = urllib.request.Request(f"{self.base}{path}", data=body, method=method)
        req.add_header("Authorization", self.token)          # raw key, no "Bearer"
        if content_type:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raise PostizError(f"{method} {path} → HTTP {e.code}: {e.read()[:200]!r}") from e
        except Exception as e:  # noqa: BLE001
            raise PostizError(f"{method} {path} failed: {e}") from e

    def list_integrations(self) -> list[dict]:
        data = self._request("/public/v1/integrations", timeout=30)
        return data if isinstance(data, list) else data.get("integrations", data.get("data", []))

    def upload(self, video_path: str) -> dict:
        with open(video_path, "rb") as f:
            content = f.read()
        boundary = uuid.uuid4().hex
        name = video_path.rsplit("/", 1)[-1]
        pre = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
               f"filename=\"{name}\"\r\nContent-Type: video/mp4\r\n\r\n").encode()
        body = pre + content + f"\r\n--{boundary}--\r\n".encode()
        return self._request("/public/v1/upload", "POST", body,
                             f"multipart/form-data; boundary={boundary}", timeout=600)

    def create_post(self, integration_id: str, caption: str, media: dict, settings: dict,
                    when_iso: str | None, short_link: bool = False) -> dict:
        payload = {
            "type": "schedule" if when_iso else "now",
            "date": when_iso or now_iso(),
            "shortLink": bool(short_link),
            "tags": [],
            "posts": [{"integration": {"id": integration_id},
                       "value": [{"content": caption, "image": [media]}],
                       "settings": settings}],
        }
        return self._request("/public/v1/posts", "POST", json.dumps(payload).encode(),
                             "application/json")


def discover(client: PostizClient) -> dict:
    """platform -> integration id, from the connected channels. Best-effort."""
    out: dict = {}
    for it in client.list_integrations():
        ident = str(it.get("identifier") or it.get("provider") or it.get("platform") or "").lower()
        iid = it.get("id") or it.get("integrationId")
        p = canon(ident)
        if iid and p and p not in out:
            out[p] = iid
    return out
