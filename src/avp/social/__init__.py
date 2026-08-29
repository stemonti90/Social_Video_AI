"""Native publishing to TikTok, Instagram and YouTube — no Postiz, no broker.

Why this exists: Postiz is a fine general-purpose scheduler, but for a single creator posting their own
videos it means running Next.js + Postgres + Redis, and — the part that actually costs us — it demands
six TikTok scopes and hard-fails auth if any one is missing. Talking to the platforms directly needs
three (`user.info.basic`, `video.upload`, `video.publish`), and TikTok's review explicitly penalises
apps that ask for scopes they don't exercise.

Everything durable is in the token store; the publishers themselves are stateless.
"""
from __future__ import annotations

import secrets
from pathlib import Path

from ..log import get_logger
from . import tokens
from .base import Publisher, redirect_uri
from .instagram import Instagram
from .tiktok import TikTok
from .youtube import YouTube

log = get_logger("avp.social")

_REGISTRY: dict[str, type[Publisher]] = {
    "tiktok": TikTok,
    "instagram": Instagram,
    "youtube": YouTube,
}

PLATFORMS = tuple(_REGISTRY)


def publisher(platform: str) -> Publisher:
    p = platform.lower().strip()
    if p not in _REGISTRY:
        raise RuntimeError(f"No native publisher for {platform!r} "
                           f"(have: {', '.join(sorted(_REGISTRY))}).")
    return _REGISTRY[p]()


# --------------------------------------------------------------------------- connecting
def start_connect(platform: str, cfg=None) -> tuple[str, str]:
    """(authorize_url, state) to hand the creator. `state` is echoed back by the platform and shown on
    the callback page, so a stale browser tab from an earlier attempt can be spotted rather than
    silently completing the wrong connection."""
    pub = publisher(platform)
    state = secrets.token_urlsafe(12)
    return pub.authorize_url(state, cfg), state


def finish_connect(platform: str, code: str, cfg=None) -> dict:
    """Trade the pasted code for tokens and store them. Returns the redacted summary."""
    pub = publisher(platform)
    # Platforms URL-encode the code in the redirect (TikTok appends a literal '*'); a code pasted
    # straight out of a browser address bar therefore arrives escaped and the exchange 400s with a
    # useless "invalid_grant".
    from urllib.parse import unquote
    rec = pub.exchange(unquote(code.strip()), cfg)
    tokens.put(platform, rec)
    return tokens.connected().get(platform, {})


# --------------------------------------------------------------------------- posting
def post(platform: str, video: Path, caption: str, meta: dict, cfg,
         disclose_ai: bool = False) -> dict:
    """Publish one rendered video. Raises on failure — callers decide whether one dead platform should
    stop the others."""
    pub = publisher(platform)
    token, record = pub.access_token(cfg)
    log.info("Publishing to %s as %s (%.1f MB)", platform,
             record.get("account", "?"), video.stat().st_size / 1e6)
    result = pub.post(video, caption, meta, cfg, token, record, disclose_ai)
    log.info("%s: published (%s)", platform, result.get("post_id") or result.get("publish_id") or "ok")
    return result
