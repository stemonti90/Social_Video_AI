"""Publish a finished video to socials via Postiz (open-source scheduler, AGPL-3.0).

Postiz covers TikTok, Instagram, YouTube + ~18 other networks. Real posting needs a
running Postiz with the channels connected (and their platform app approvals — e.g. the
TikTok content-posting audit, Meta app review). By DEFAULT this is a dry run: it builds
the per-platform plan (caption/hashtags from metadata.json + the video) and writes
publish_plan.json. Pass go=True (CLI --go) with Postiz configured to actually post.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from . import tts as tts_mod
from .config import Config, PublishConfig
from .log import get_logger
from .manifest import VideoProject

log = get_logger("avp.publish")


def _caption_for(platform: str, meta: dict) -> str:
    p = platform.lower()
    if p in ("youtube", "yt", "shorts"):
        yt = meta.get("youtube", {})
        return f"{yt.get('title', '')}\n\n{yt.get('description', '')}".strip()
    if p in ("tiktok", "tt"):
        return meta.get("tiktok", {}).get("caption", "")
    if p in ("instagram", "ig", "reels"):
        return meta.get("instagram", {}).get("caption", "")
    return meta.get("tiktok", {}).get("caption", "")


class PostizClient:
    """Minimal Postiz public-API client. Verify endpoints/schema for your Postiz version."""

    def __init__(self, cfg: PublishConfig):
        self.base = cfg.postiz_url.rstrip("/")
        self.token = os.getenv("AVP_POSTIZ_TOKEN", cfg.postiz_token)

    def _headers(self) -> dict:
        return {"Authorization": self.token}

    def upload(self, video: Path) -> dict:
        with open(video, "rb") as f:
            r = requests.post(f"{self.base}/public/v1/upload", headers=self._headers(),
                              files={"file": (video.name, f, "video/mp4")}, timeout=600)
        r.raise_for_status()
        return r.json()

    def post(self, integration_id: str, caption: str, media: dict) -> dict:
        body = {"type": "now",
                "posts": [{"integration": {"id": integration_id},
                           "value": [{"content": caption, "image": [media]}]}]}
        r = requests.post(f"{self.base}/public/v1/posts", headers=self._headers(),
                          json=body, timeout=120)
        r.raise_for_status()
        return r.json()


def stage_publish(project: VideoProject, cfg: Config, go: bool = False,
                  platforms: list[str] | None = None) -> list[dict]:
    meta_path = project.root / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    eng = cfg.publish.voice or tts_mod.primary_engine()
    video = project.output_for(eng)
    if not video.exists():
        video = project.output
    if not video.exists():
        raise RuntimeError("No rendered video found — run `build` first.")

    plats = platforms or cfg.publish.platforms
    plan = [{
        "platform": p,
        "video": str(video),
        "caption": _caption_for(p, meta),
        "disclosure_ai": bool(meta.get("disclosure_ai", False)),
    } for p in plats]
    (project.root / "publish_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))

    for it in plan:
        log.info("[%s] %s", it["platform"], (it["caption"][:90] or "(no caption)"))

    if not go:
        log.info("DRY RUN — wrote publish_plan.json. Configure Postiz + run with --go to post.")
        return plan

    client = PostizClient(cfg.publish)
    if not client.token:
        raise RuntimeError("No Postiz token (set publish.postiz_token or env AVP_POSTIZ_TOKEN).")
    log.warning("Live publish via Postiz at %s — confirm integration IDs for your setup.",
                cfg.publish.postiz_url)
    try:
        media = client.upload(video)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Postiz upload failed ({e}). Is Postiz running and reachable?") from e

    for it in plan:
        integ = cfg.publish.integrations.get(it["platform"])
        if not integ:
            log.warning("No Postiz integration id for %r (set publish.integrations) — skipping.",
                        it["platform"])
            continue
        try:
            client.post(integ, it["caption"], media)
            log.info("Posted to %s", it["platform"])
        except Exception as e:  # noqa: BLE001
            log.error("Post to %s failed: %s", it["platform"], e)
    return plan
