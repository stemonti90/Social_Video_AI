"""Unattended daily pipeline: pick N topics, generate+build N videos, and schedule them to Postiz at
the configured local times.

Topics come from a queue file (one per line, `#` comments allowed); when it runs low the LLM proposes
fresh ones, deduped against the queue AND every past project. Publishing happens only for platforms
whose channel is actually connected in Postiz — until TikTok/Instagram are connected (their OAuth apps
approved), the videos are still generated and left ready, and nothing is silently posted.

Everything here is path-portable (driven by config + cwd), so it moves to a Mac mini unchanged — but
the generation models are MLX/Metal, so the host must be Apple Silicon.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import llm
from .config import Config
from .log import get_logger
from .manifest import VideoProject

log = get_logger("avp.auto")

try:
    from zoneinfo import ZoneInfo
except Exception:  # noqa: BLE001 — zoneinfo is stdlib on 3.9+, but never let its absence crash a run
    ZoneInfo = None


# --------------------------------------------------------------------------- slugs
def slugify(text: str, maxlen: int = 48) -> str:
    """A topic → a safe [a-z0-9-] project slug (matches cli's _SLUG_RE). Accents are folded first
    (Perché → perche) so Italian topics don't lose letters."""
    folded = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")
    s = re.sub(r"-{2,}", "-", s)[:maxlen].strip("-")
    return s or "video"


def _unique_slug(base: str, projects_dir: Path) -> str:
    slug, i = base, 2
    while (projects_dir / slug).exists():
        slug, i = f"{base}-{i}", i + 1
    return slug


# --------------------------------------------------------------------------- topic queue
def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _queue_path(cfg: Config) -> Path:
    p = Path(cfg.auto.queue_path).expanduser()
    return p if p.is_absolute() else Path(cfg.paths.projects_dir).expanduser() / p


def load_queue(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def save_queue(path: Path, topics: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# AVP topic queue — one topic per line. Edit freely; the daily run pops from the top and\n" \
             "# the LLM refills from the bottom when it runs low. Lines starting with # are ignored.\n"
    path.write_text(header + "\n".join(topics) + ("\n" if topics else ""))


def existing_topics(cfg: Config) -> list[str]:
    """Raw topic/title labels of every past project — the dedup set for brainstorming."""
    root = Path(cfg.paths.projects_dir).expanduser()
    labels: list[str] = []
    for man in sorted(root.glob("*/manifest.json")):
        try:
            d = json.loads(man.read_text())
        except Exception:  # noqa: BLE001 — a broken manifest must not stop topic selection
            continue
        for k in ("topic", "title"):
            v = (d.get(k) or "").strip()
            if v:
                labels.append(v)
    return labels


def next_topics(cfg: Config, n: int, consume: bool = True) -> list[str]:
    """Return the next `n` topics. When consuming (a real run), refill via the LLM if the queue is low,
    then pop `n` off the top and persist the rest. When peeking (dry run), never mutate or call the LLM."""
    path = _queue_path(cfg)
    queue = load_queue(path)
    if consume and len(queue) < max(n, cfg.auto.refill_threshold):
        avoid = list(dict.fromkeys(queue + existing_topics(cfg)))
        fresh = llm.brainstorm_topics(cfg.llm, avoid=avoid, n=cfg.auto.refill_batch,
                                      theme=cfg.auto.theme, language=cfg.script.language)
        have = {_key(t) for t in queue}
        added = [t for t in fresh if _key(t) not in have]
        queue += added
        save_queue(path, queue)
        log.info("Topic queue refilled: +%d (now %d).", len(added), len(queue))
    topics = queue[:n]
    if consume and topics:
        save_queue(path, queue[len(topics):])
    return topics


# --------------------------------------------------------------------------- scheduling
def _zone(tz: str):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz)
        except Exception:  # noqa: BLE001 — bad tz name
            log.warning("Unknown timezone %r — using UTC.", tz)
    return timezone.utc


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def post_slots(now: datetime, times: list[str], tz: str, count: int) -> list[datetime]:
    """The next `count` posting datetimes (tz-aware) from the daily `times` (HH:MM), rolling into
    following days once today's remaining slots are used up. Only slots strictly in the future."""
    zone = _zone(tz)
    now = now.astimezone(zone)
    parsed: list[tuple[int, int]] = []
    for t in times:
        try:
            hh, mm = (int(x) for x in str(t).split(":")[:2])
            if 0 <= hh < 24 and 0 <= mm < 60:
                parsed.append((hh, mm))
        except Exception:  # noqa: BLE001 — skip a malformed time entry
            continue
    parsed = parsed or [(12, 0), (18, 0), (21, 0)]
    slots: list[datetime] = []
    for day in range(0, 15):                      # up to two weeks out — a safety bound, never reached
        base = now + timedelta(days=day)
        for hh, mm in parsed:
            cand = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if cand > now:
                slots.append(cand)
                if len(slots) >= count:
                    return slots
    return slots


# --------------------------------------------------------------------------- channels
def connected_platforms(cfg: Config) -> set[str]:
    """Which target platforms can actually be posted to right now (empty on any error).

    Native backend: a platform counts when its OAuth token is in the store — and TikTok counts only
    once the account may post PUBLICLY. That second test is what turns "remember to add TikTok when
    it gets unblocked" into something that needs no remembering: while the app is in review,
    creator_info offers FOLLOWER_OF_CREATOR / SELF_ONLY and a post would go out hidden, so TikTok
    stays off; the run after approval sees PUBLIC_TO_EVERYONE and switches it on by itself.

    This used to ask Postiz regardless of backend. Under the native backend that call failed, the
    target list came back empty, and the daily run would have BUILT every video and posted none —
    the warning said so, to a log nobody was reading at 07:20."""
    from . import publish as publish_mod
    if (cfg.publish.backend or "native").lower() == "native":
        return _native_connected(cfg)
    try:
        client = publish_mod.PostizClient(cfg.publish)
        if not client.token:
            return set()
        found = {publish_mod._canon(str(it.get("identifier") or it.get("provider")
                                        or it.get("platform") or "")) for it in client.list_integrations()}
        return found - {""}
    except Exception as e:  # noqa: BLE001 — unreachable Postiz just means "publish nothing yet"
        log.warning("Could not check Postiz channels (%s) — will generate only.", e)
        return set()


def _native_connected(cfg: Config) -> set[str]:
    from . import publish as publish_mod
    from .social import tokens as token_store
    found: set[str] = set()
    for plat in cfg.auto.platforms:
        canon = publish_mod._canon(plat)
        rec = token_store.get(canon)
        if not rec:
            continue
        if canon == "tiktok" and not tiktok_can_post_publicly(cfg, rec):
            log.info("TikTok is connected but may not post publicly yet (app in review or account "
                     "private) — leaving it out of today's targets.")
            continue
        found.add(canon)
    return found


def tiktok_can_post_publicly(cfg: Config, rec: dict) -> bool:
    """True only when TikTok's own creator_info lists PUBLIC_TO_EVERYONE. Refreshes the token if it
    has gone stale (access tokens last a day; the refresh token a year). Any error means "not yet":
    a wrong answer here posts a video nobody can see, so the safe default is to wait."""
    try:
        from .social import tokens as token_store
        from .social.tiktok import TikTok
        tt = TikTok()
        if not token_store.is_fresh(rec):
            rec = tt.refresh(rec, cfg)
            token_store.put("tiktok", rec)
        opts = tt.creator_info(rec["access_token"]).get("privacy_level_options") or []
        return "PUBLIC_TO_EVERYONE" in opts
    except Exception as e:  # noqa: BLE001 — never let a status probe abort the daily run
        log.warning("Could not check TikTok posting privileges (%s) — treating as not unblocked.", e)
        return False


# --------------------------------------------------------------------------- the daily run
def run_daily(cfg: Config, count: int | None = None, dry_run: bool = False,
              publish: bool = True, config_path: str = "config.yaml") -> list[dict]:
    from . import publish as publish_mod

    n = int(count or cfg.auto.count)
    topics = next_topics(cfg, n, consume=not dry_run)
    if not topics:
        log.error("No topics available (queue empty and brainstorm produced none).")
        return []
    slots = post_slots(datetime.now(_zone(cfg.auto.timezone)), cfg.auto.post_times, cfg.auto.timezone,
                       len(topics))
    projects_dir = Path(cfg.paths.projects_dir).expanduser()
    connected = connected_platforms(cfg) if (publish and not dry_run) else set()
    targets = [p for p in cfg.auto.platforms if publish_mod._canon(p) in connected]
    if publish and not dry_run and not targets:
        log.warning("Nothing in %s can be posted to right now — videos will be BUILT but not posted.",
                    cfg.auto.platforms)

    report: list[dict] = []
    for topic, slot in zip(topics, slots):
        slug = _unique_slug(slugify(topic), projects_dir)
        entry = {"topic": topic, "slug": slug, "scheduled_local": slot.isoformat(),
                 "scheduled_utc": _iso_utc(slot)}
        if dry_run:
            entry["would_publish_to"] = cfg.auto.platforms
            report.append(entry)
            continue
        try:
            from . import pipeline, stages
            project = VideoProject.create(slug, cfg)
            stages.stage_script(project, cfg, topic)
            pipeline.build(project, cfg, config_path=config_path)
            entry["built"] = True
            if publish and targets:
                publish_mod.stage_publish(project, cfg, go=True, platforms=targets, when=entry["scheduled_utc"])
                entry["published_to"] = targets
            elif publish:
                entry["published_to"] = []          # generated, awaiting a connected channel
        except Exception as e:  # noqa: BLE001 — one failed video must not sink the whole batch
            log.error("auto: video for %r failed: %s", topic, e)
            entry["built"] = False
            entry["error"] = str(e)
        report.append(entry)
    return report
