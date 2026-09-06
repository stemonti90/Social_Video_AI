"""Publish a finished video to socials.

Two backends, chosen by ``publish.backend``:

* **native** (default) — `avp.social` talks to TikTok / Meta / Google directly. Nothing else to run,
  and TikTok is asked for three scopes instead of six.
* **postiz** — the original path through a self-hosted Postiz (AGPL-3.0), still useful for the ~18
  other networks it supports.

Either way the platform app approvals are unavoidable (TikTok's content-posting audit, Meta app
review, Google's OAuth verification) — those are imposed by the platforms, not by the backend.

By DEFAULT this is a dry run: it builds the per-platform plan (caption + the exact provider settings
that would be sent) and writes publish_plan.json. Pass go=True (CLI --go) to actually post.

API contract (verified against https://docs.postiz.com/public-api, 2026-06):
  - Auth header: ``Authorization: {apiKey}`` (raw key, NOT ``Bearer``).
  - Base URL: ``{postiz_url}/public/v1`` (self-host) — postiz_url is NEXT_PUBLIC_BACKEND_URL.
  - GET  /public/v1/integrations          → the connected channels: [{id, name, identifier}, ...].
  - POST /public/v1/upload  (-F file=@…)  → {"id": "...", "path": "https://…"}.
  - POST /public/v1/posts                 → {type, date, shortLink, tags, posts:[{integration:{id},
        value:[{content, image:[media]}], settings:{__type, …}}]}. Each provider REQUIRES its own
        settings object (YouTube: title+type; TikTok: privacy_level + toggles; Instagram: post_type) —
        omitting it is why a naive post is rejected.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from . import tts as tts_mod
from .config import Config, PublishConfig
from .log import get_logger
from .manifest import VideoProject

log = get_logger("avp.publish")

# Map a platform name (and its aliases) to the canonical key used for captions/settings/discovery.
_ALIASES = {
    "youtube": "youtube", "yt": "youtube", "shorts": "youtube",
    "tiktok": "tiktok", "tt": "tiktok",
    "instagram": "instagram", "ig": "instagram", "reels": "instagram",
}


def _canon(platform: str) -> str:
    return _ALIASES.get(platform.lower().strip(), platform.lower().strip())


def _now_iso(plus_seconds: int = 60) -> str:
    """A near-future UTC timestamp. Postiz expects a `date` even for immediate posts; a small lead
    avoids 'scheduled in the past' rejections when the request takes a moment to arrive."""
    return (datetime.now(timezone.utc) + timedelta(seconds=plus_seconds)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _caption_for(platform: str, meta: dict) -> str:
    p = _canon(platform)
    if p == "youtube":
        yt = meta.get("youtube", {})
        return f"{yt.get('title', '')}\n\n{yt.get('description', '')}".strip()
    if p == "tiktok":
        return meta.get("tiktok", {}).get("caption", "")
    if p == "instagram":
        return meta.get("instagram", {}).get("caption", "")
    if p == "reddit":
        # Reddit shows a TITLE, then the video; the description carries the app link. No hashtags.
        yt = meta.get("youtube", {})
        return (yt.get("description") or meta.get("tiktok", {}).get("caption", "")).strip()
    return meta.get("tiktok", {}).get("caption", "")


def _title_for(platform: str, meta: dict) -> str:
    """A short title for platforms that need one (YouTube required 2-100, TikTok optional ≤90)."""
    yt_title = (meta.get("youtube", {}).get("title") or "").strip()
    if yt_title:
        return yt_title
    # fall back to the first line / sentence of any caption
    cap = _caption_for(platform, meta) or _caption_for("tiktok", meta)
    first = (cap.splitlines() or [""])[0].strip()
    return first or "Untitled"


def _settings_for(platform: str, meta: dict, pub: PublishConfig, disclose_ai: bool) -> dict:
    """Build the provider-specific Postiz `settings` object. These required fields are why the previous
    client (which sent no settings) could never post — each provider validates its own schema."""
    p = _canon(platform)
    if p == "youtube":
        yt = meta.get("youtube", {})
        vis = pub.privacy if pub.privacy in ("public", "unlisted", "private") else "public"
        tags = [{"value": t, "label": t} for t in (yt.get("tags") or [])][:15]
        title = (_title_for("youtube", meta))[:100] or "Untitled"
        if len(title) < 2:
            title = "Untitled"
        return {"__type": "youtube", "title": title, "type": vis,
                "selfDeclaredMadeForKids": "yes" if pub.made_for_kids else "no",
                "thumbnail": None, "tags": tags}
    if p == "tiktok":
        privacy = {"public": "PUBLIC_TO_EVERYONE", "unlisted": "FOLLOWER_OF_CREATOR",
                   "private": "SELF_ONLY"}.get(pub.privacy, "PUBLIC_TO_EVERYONE")
        return {"__type": "tiktok", "title": _title_for("tiktok", meta)[:90],
                "privacy_level": privacy, "duet": False, "stitch": False, "comment": True,
                "autoAddMusic": "no", "brand_content_toggle": False, "brand_organic_toggle": False,
                "video_made_with_ai": bool(disclose_ai), "content_posting_method": "DIRECT_POST"}
    if p == "instagram":
        return {"__type": "instagram", "post_type": "post"}   # "post" + a single video = a Reel
    return {"__type": p}


class PostizClient:
    """Minimal Postiz public-API client (verified against docs.postiz.com/public-api, 2026-06)."""

    def __init__(self, cfg: PublishConfig):
        self.base = cfg.postiz_url.rstrip("/")
        self.token = os.getenv("AVP_POSTIZ_TOKEN", cfg.postiz_token)

    def _headers(self) -> dict:
        return {"Authorization": self.token}

    def list_integrations(self) -> list[dict]:
        r = requests.get(f"{self.base}/public/v1/integrations", headers=self._headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else data.get("integrations", data.get("data", []))

    def upload(self, video: Path) -> dict:
        with open(video, "rb") as f:
            r = requests.post(f"{self.base}/public/v1/upload", headers=self._headers(),
                              files={"file": (video.name, f, "video/mp4")}, timeout=600)
        r.raise_for_status()
        return r.json()

    def create_post(self, integration_id: str, caption: str, media: dict, settings: dict,
                    when_iso: str | None, short_link: bool = False) -> dict:
        body = {
            "type": "schedule" if when_iso else "now",
            "date": when_iso or _now_iso(),
            "shortLink": bool(short_link),
            "tags": [],
            "posts": [{"integration": {"id": integration_id},
                       "value": [{"content": caption, "image": [media]}],
                       "settings": settings}],
        }
        r = requests.post(f"{self.base}/public/v1/posts", headers=self._headers(),
                          json=body, timeout=120)
        r.raise_for_status()
        return r.json()


def _integration_id(platform: str, configured: dict, discovered: dict[str, str]) -> str | None:
    """Resolve the Postiz integration id for a platform: an explicit config mapping wins, else the
    channel auto-discovered from GET /integrations by matching its `identifier`."""
    p = _canon(platform)
    for key in (platform, p):                       # honor whatever key the user wrote in config
        if key in configured and configured[key]:
            return configured[key]
    return discovered.get(p)


def _discover(client: PostizClient) -> dict[str, str]:
    """platform -> integration id, from the connected channels. Best-effort; never raises."""
    out: dict[str, str] = {}
    try:
        for it in client.list_integrations():
            ident = str(it.get("identifier") or it.get("provider") or it.get("platform") or "").lower()
            iid = it.get("id") or it.get("integrationId")
            p = _canon(ident)
            if iid and p and p not in out:
                out[p] = iid
    except Exception as e:  # noqa: BLE001 — discovery is a convenience; missing ids are reported later
        log.warning("Could not list Postiz integrations (%s) — relying on config.integrations.", e)
    return out


def _publish_native(plan: list[dict], video: Path, meta: dict, cfg: Config,
                    disclose_ai: bool, project: VideoProject) -> list[dict]:
    """Post directly to each platform. One dead platform must not take the others down with it — a
    failed TikTok upload is no reason to skip a perfectly good Instagram post — so failures are
    recorded per item and the loop continues."""
    from . import social

    for it in plan:
        plat = it["platform"]
        try:
            if (cfg.publish.via or {}).get(plat) == "uploadpost":
                from .social import uploadpost
                it["result"] = uploadpost.post(plat, video, it["caption"], meta, cfg, disclose_ai,
                                               title=_title_for(plat, meta))
            else:
                it["result"] = social.post(plat, video, it["caption"], meta, cfg, disclose_ai)
            it["posted"] = True
        except Exception as e:  # noqa: BLE001 — the reason belongs in the plan, not a traceback
            it["posted"] = False
            it["error"] = str(e)
            log.error("Post to %s failed: %s", plat, e)
    (project.root / "publish_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    ok = [i["platform"] for i in plan if i.get("posted")]
    log.info("Published to %s", ", ".join(ok) if ok else "nothing")
    return plan


def stage_publish(project: VideoProject, cfg: Config, go: bool = False,
                  platforms: list[str] | None = None, when: str | None = None) -> list[dict]:
    meta_path = project.root / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    eng = cfg.publish.voice or tts_mod.primary_engine()
    video = project.output_for(eng)
    if not video.exists():
        video = project.output
    if not video.exists():
        raise RuntimeError("No rendered video found — run `build` first.")

    # The pipeline's footage is real; disclose AI only if the user opted in (covers the AI voice/script)
    # or the script itself flagged realistic AI visuals.
    disclose_ai = bool(cfg.publish.disclose_ai or meta.get("disclosure_ai", False))
    plats = platforms or cfg.publish.platforms

    plan = [{
        "platform": _canon(p),
        "video": str(video),
        "caption": _caption_for(p, meta),
        "schedule": (when or "now") if (cfg.publish.backend or "native").lower() != "native" else "now",
        "disclose_ai": disclose_ai,
        "settings": _settings_for(p, meta, cfg.publish, disclose_ai),
    } for p in plats]
    (project.root / "publish_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))

    for it in plan:
        log.info("[%s] %s", it["platform"], (it["caption"][:90] or "(no caption)"))

    if not go:
        how = ("connect the accounts with `avp connect <platform>`"
               if (cfg.publish.backend or "native").lower() == "native" else "configure Postiz")
        log.info("DRY RUN — wrote publish_plan.json. To post: %s, then run with --go.", how)
        return plan

    if (cfg.publish.backend or "native").lower() == "native":
        # `when` is honoured by Postiz, which holds a queue. The native path has nowhere to hold one:
        # Instagram's Content Publishing API creates a container and publishes it, with no "publish
        # at" — so a caller that asked for a slot gets an immediate post instead. Say so out loud
        # rather than logging "scheduled" over a post that already went out; the way to space posts
        # on this backend is to run the pipeline twice, not to ask it to wait.
        if when:
            log.warning("Native backend posts IMMEDIATELY — the requested slot (%s) cannot be "
                        "honoured, because Instagram's API has no scheduled publish. Run the "
                        "pipeline at the time you want the post to go out.", when)
        return _publish_native(plan, video, meta, cfg, disclose_ai, project)

    client = PostizClient(cfg.publish)
    if not client.token:
        raise RuntimeError("No Postiz token (set publish.postiz_token or env AVP_POSTIZ_TOKEN).")
    log.warning("Live publish via Postiz at %s%s", cfg.publish.postiz_url,
                f" — scheduled {when}" if when else " — posting now")
    discovered = _discover(client)
    try:
        media = client.upload(video)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Postiz upload failed ({e}). Is Postiz running and reachable at "
                           f"{cfg.publish.postiz_url}?") from e

    for it in plan:
        integ = _integration_id(it["platform"], cfg.publish.integrations, discovered)
        if not integ:
            log.warning("No Postiz channel for %r (connect it in Postiz, or set publish.integrations) "
                        "— skipping.", it["platform"])
            it["posted"] = False
            continue
        try:
            res = client.create_post(integ, it["caption"], media, it["settings"], when,
                                     cfg.publish.short_link)
            it["posted"] = True
            log.info("Posted to %s (%s)", it["platform"], "scheduled" if when else "now")
        except Exception as e:  # noqa: BLE001
            it["posted"] = False
            log.error("Post to %s failed: %s", it["platform"], e)
    (project.root / "publish_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    return plan
