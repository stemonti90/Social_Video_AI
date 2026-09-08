"""Measure what was published: a daily snapshot of every post's numbers and a weekly report.

The growth loop is create → publish → MEASURE → analyse → learn → optimise. This module is the
"measure" and the arithmetic of "analyse"; it never writes prose it cannot back with a number.

Sources
  * Instagram Graph API — the account (followers, media count) and every Reel (likes, comments, and,
    when the token carries the insights permission, views, reach, saves, shares, watch time). Without
    that permission the report says so instead of guessing.
  * Upload-Post — the TikTok profile (followers, reach, impressions, profile views, a reach time
    series), every TikTok post published through it (views, likes, comments, shares, favorites, reach —
    from Upload-Post's snapshot cache, which fills a day or two after posting) and the audience's
    activity by hour, which is the only honest basis for a posting-time hypothesis.
  * The pipeline's own record — publish.py appends every successful post to projects/_auto/posts.jsonl
    (slug, lane, platform, post id/url, time), so a number can be attributed to a video, a lane, a hook.

Snapshots live in projects/_auto/metrics/YYYY-MM-DD.json; the report compares the newest with the
oldest inside the window, so "followers gained" is measured, not estimated.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)

IG = "https://graph.instagram.com/v21.0"
UP = "https://api.upload-post.com/api"
TIMEOUT = 30


# --------------------------------------------------------------------------- paths & records
def auto_dir(cfg) -> Path:
    return Path(cfg.paths.projects_dir).expanduser() / "_auto"


def posts_log(cfg) -> Path:
    return auto_dir(cfg) / "posts.jsonl"


def record_post(cfg, slug: str, lane: str, platform: str, result: dict | str | None) -> None:
    """One line per successful post: what went where, when, under which lane."""
    p = posts_log(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    rid, url = "", ""
    if isinstance(result, dict):
        rid = str(result.get("id") or result.get("post_id") or result.get("publish_id") or "")
        url = str(result.get("url") or result.get("post_url") or result.get("permalink") or "")
    elif isinstance(result, str):
        rid = result
    with p.open("a") as fh:
        fh.write(json.dumps({"slug": slug, "lane": lane, "platform": platform, "id": rid, "url": url,
                             "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")


def load_posts(cfg) -> list[dict]:
    p = posts_log(cfg)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def project_index(cfg) -> dict[str, dict]:
    """slug → {title, lane, hook, ig_caption, tt_caption, duration} for attributing platform posts."""
    root = Path(cfg.paths.projects_dir).expanduser()
    out: dict[str, dict] = {}
    for man in root.glob("*/manifest.json"):
        slug = man.parent.name
        try:
            d = json.loads(man.read_text())
        except Exception:  # noqa: BLE001
            continue
        info = {"title": d.get("title", ""), "lane": d.get("lane", "discovery"), "topic": d.get("topic", "")}
        meta = man.parent / "metadata.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                info["ig_caption"] = ((m.get("instagram") or {}).get("caption") or "").split("\n")[0].strip()
                info["tt_caption"] = re.split(r"\s#", (m.get("tiktok") or {}).get("caption") or "", 1)[0].strip()
            except Exception:  # noqa: BLE001
                pass
        script = man.parent / "script.json"
        if script.exists():
            try:
                s = json.loads(script.read_text())
                segs = [x for x in s.get("segments", []) if x.get("kind") != "cta"]
                info["hook"] = segs[0]["narration"] if segs else ""
                info["duration"] = round(sum(float(x.get("duration") or 0) for x in s.get("segments", [])), 1)
            except Exception:  # noqa: BLE001
                pass
        out[slug] = info
    return out


def attribute(post: dict, posts: list[dict], index: dict[str, dict]) -> str | None:
    """Which project a platform post belongs to: the publish record by id/url first, else the caption."""
    pid, url = str(post.get("id") or ""), str(post.get("url") or "")
    for rec in posts:
        if rec.get("platform") == post.get("platform") and rec.get("id") and (rec["id"] == pid or (url and rec["id"] in url)):
            return rec["slug"]
    cap = (post.get("caption") or "").strip().lower()
    if cap:
        key = "ig_caption" if post.get("platform") == "instagram" else "tt_caption"
        for slug, info in index.items():
            c = (info.get(key) or "").strip().lower()
            if c and (cap.startswith(c[:40]) or c.startswith(cap[:40])):
                return slug
    return None


# --------------------------------------------------------------------------- collectors
def collect_instagram(cfg) -> dict:
    from . import social
    token, rec = social.publisher("instagram").access_token(cfg)
    out: dict = {"account": {}, "posts": [], "insights_permission": True}
    me = requests.get(f"{IG}/me", params={"fields": "id,username,followers_count,media_count",
                                          "access_token": token}, timeout=TIMEOUT).json()
    out["account"] = {"followers": me.get("followers_count"), "media": me.get("media_count"),
                      "username": me.get("username")}
    url, params = f"{IG}/me/media", {"fields": "id,caption,permalink,timestamp,media_product_type,like_count,comments_count",
                                    "limit": 50, "access_token": token}
    media: list[dict] = []
    while url and len(media) < 300:
        d = requests.get(url, params=params, timeout=TIMEOUT).json()
        if "error" in d:
            log.warning("Instagram media list: %s", d["error"].get("message"))
            break
        media += d.get("data", [])
        url, params = (d.get("paging") or {}).get("next"), None
    for m in media:
        if m.get("media_product_type") != "REELS":
            continue
        post = {"platform": "instagram", "id": m["id"], "url": m.get("permalink", ""), "at": m.get("timestamp", ""),
                "caption": (m.get("caption") or "").split("\n")[0][:120],
                "likes": m.get("like_count"), "comments": m.get("comments_count")}
        if out["insights_permission"]:
            ins = requests.get(f"{IG}/{m['id']}/insights",
                               params={"metric": "views,reach,saved,shares,total_interactions,ig_reels_avg_watch_time",
                                       "access_token": token}, timeout=TIMEOUT).json()
            if "error" in ins:
                out["insights_permission"] = False
                out["insights_error"] = ins["error"].get("message", "")[:160]
            else:
                for dd in ins.get("data", []):
                    v = (dd.get("values") or [{}])[0].get("value")
                    post[{"saved": "saves", "ig_reels_avg_watch_time": "avg_watch_ms"}.get(dd["name"], dd["name"])] = v
        out["posts"].append(post)
    return out


def collect_tiktok(cfg) -> dict:
    from .social import uploadpost
    s = uploadpost.settings(cfg)
    key, user = s.get("api_key"), s.get("user") or "default"
    out: dict = {"account": {}, "posts": [], "audience_by_hour": []}
    if not key:
        out["error"] = "Upload-Post not configured"
        return out
    h = {"Authorization": f"Apikey {key}"}
    prof = requests.get(f"{UP}/analytics/{user}", params={"platforms": "tiktok"}, headers=h, timeout=TIMEOUT).json()
    tk = prof.get("tiktok") or {}
    out["account"] = {k: tk.get(k) for k in ("followers", "reach", "impressions", "profileViews")}
    out["reach_timeseries"] = tk.get("reach_timeseries") or []
    cursor, pages = None, 0
    while pages < 10:
        params = {"user": user, "platform": "tiktok", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        d = requests.get(f"{UP}/uploadposts/post-analytics/cached", params=params, headers=h, timeout=TIMEOUT).json()
        for p in d.get("posts") or []:
            met = p.get("metrics") or {}
            out["posts"].append({"platform": "tiktok", "id": str(p.get("post_id") or ""), "url": p.get("post_url", ""),
                                 "at": p.get("date") or p.get("upload_timestamp") or "", "caption": "",
                                 "views": met.get("views"), "likes": met.get("likes"), "comments": met.get("comments"),
                                 "shares": met.get("shares"), "saves": met.get("favorites"), "reach": met.get("reach")})
        cursor = d.get("next_cursor"); pages += 1
        if not d.get("has_more") or not cursor:
            break
    out["published_posts_in_range"] = d.get("published_posts_in_range") if isinstance(d, dict) else None
    aud = requests.get(f"{UP}/uploadposts/audience", params={"platform": "tiktok", "user": user}, headers=h, timeout=TIMEOUT).json()
    out["audience_by_hour"] = [(int(x.get("hour", 0)), int(x.get("followers_online", 0) or 0))
                               for x in aud.get("activity_by_hour") or []]
    return out


def snapshot(cfg, when: date | None = None) -> Path:
    """Collect both platforms and save projects/_auto/metrics/YYYY-MM-DD.json. Never raises: a platform
    that fails is recorded with its error and the rest is kept."""
    when = when or date.today()
    data: dict = {"date": when.isoformat(), "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for name, fn in (("instagram", collect_instagram), ("tiktok", collect_tiktok)):
        try:
            data[name] = fn(cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("metrics: %s collection failed (%s)", name, e)
            data[name] = {"error": str(e)[:200], "account": {}, "posts": []}
    posts, index = load_posts(cfg), project_index(cfg)
    for plat in ("instagram", "tiktok"):
        for p in data[plat].get("posts", []):
            p["slug"] = attribute(p, posts, index)
            info = index.get(p["slug"] or "", {})
            p["lane"] = info.get("lane") if p["slug"] else None
            p["title"] = info.get("title") if p["slug"] else None
    d = auto_dir(cfg) / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{when.isoformat()}.json"
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    log.info("metrics: snapshot %s — IG %d reels, TikTok %d posts", out.name,
             len(data["instagram"].get("posts", [])), len(data["tiktok"].get("posts", [])))
    return out


# --------------------------------------------------------------------------- arithmetic
def engagement(p: dict) -> float | None:
    """(likes + comments + shares + saves) / views, when views are known."""
    views = p.get("views")
    if not views:
        return None
    inter = sum(int(p.get(k) or 0) for k in ("likes", "comments", "shares", "saves"))
    return inter / views


def rank(posts: list[dict], key: str = "views", n: int = 5) -> tuple[list[dict], list[dict]]:
    """Top and bottom `n` by `key` among posts that have the number; ties broken by likes."""
    have = [p for p in posts if p.get(key) is not None]
    ordered = sorted(have, key=lambda p: (float(p.get(key) or 0), float(p.get("likes") or 0)), reverse=True)
    return ordered[:n], list(reversed(ordered[-n:])) if len(ordered) > n else []


def by_lane(posts: list[dict], key: str = "views") -> dict[str, dict]:
    acc: dict[str, list[float]] = defaultdict(list)
    for p in posts:
        if p.get(key) is not None:
            acc[p.get("lane") or "unknown"].append(float(p[key]))
    return {lane: {"n": len(v), "avg": sum(v) / len(v)} for lane, v in acc.items()}


def by_hour(posts: list[dict], key: str = "views", tz: str = "Europe/Rome") -> dict[int, dict]:
    """Average of `key` by local posting hour — a posting-time hypothesis, not a truth."""
    try:
        from zoneinfo import ZoneInfo
        z = ZoneInfo(tz)
    except Exception:  # noqa: BLE001
        z = timezone.utc
    acc: dict[int, list[float]] = defaultdict(list)
    for p in posts:
        if p.get(key) is None or not p.get("at"):
            continue
        try:
            t = datetime.fromisoformat(str(p["at"]).replace("Z", "+00:00")).astimezone(z)
        except Exception:  # noqa: BLE001
            continue
        acc[t.hour].append(float(p[key]))
    return {h: {"n": len(v), "avg": sum(v) / len(v)} for h, v in sorted(acc.items())}


def _load_snapshots(cfg, days: int) -> list[dict]:
    d = auto_dir(cfg) / "metrics"
    if not d.exists():
        return []
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    snaps = []
    for f in sorted(d.glob("*.json")):
        if f.stem >= cutoff:
            try:
                snaps.append(json.loads(f.read_text()))
            except Exception:  # noqa: BLE001
                continue
    return snaps


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}" if v < 10 else f"{v:,.0f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def report(cfg, days: int = 7, out_path: Path | None = None) -> str:
    """The weekly report in Markdown: performance, rankings, lanes, hours, and what is missing to say more."""
    snaps = _load_snapshots(cfg, days)
    if not snaps:
        snaps = [json.loads(snapshot(cfg).read_text())]
    first, last = snaps[0], snaps[-1]
    lines = [f"# Report {last['date']} — ultimi {days} giorni ({len(snaps)} rilevazioni)", ""]
    # accounts
    lines.append("## Account")
    for plat, label in (("instagram", "Instagram"), ("tiktok", "TikTok")):
        a0, a1 = first.get(plat, {}).get("account", {}), last.get(plat, {}).get("account", {})
        f0, f1 = a0.get("followers"), a1.get("followers")
        gained = (f1 - f0) if isinstance(f0, int) and isinstance(f1, int) else None
        extra = ""
        if plat == "tiktok":
            extra = f" · reach {_fmt(a1.get('reach'))} · impressioni {_fmt(a1.get('impressions'))} · visite profilo {_fmt(a1.get('profileViews'))}"
        lines.append(f"- {label}: follower {_fmt(f1)}" + (f" ({gained:+d} nel periodo)" if gained is not None else "") + extra)
        if last.get(plat, {}).get("error"):
            lines.append(f"  - errore raccolta: {last[plat]['error']}")
    ig_perm = last.get("instagram", {}).get("insights_permission", False)
    if not ig_perm:
        lines.append("- Instagram: views, reach, salvataggi e watch time NON disponibili — il token non ha il permesso "
                     "`instagram_business_manage_insights` (ricollegare l'account con quel permesso). Disponibili: like e commenti.")
    lines.append("")
    # posts
    posts = last.get("instagram", {}).get("posts", []) + last.get("tiktok", {}).get("posts", [])
    key = "views" if any(p.get("views") is not None for p in posts) else "likes"
    lines.append(f"## Video ({len(posts)} post, ordinati per {key})")
    lines.append("| piattaforma | data | corsia | video | views | like | comm. | share | salv. | eng. |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(posts, key=lambda p: float(p.get(key) or 0), reverse=True):
        e = engagement(p)
        lines.append(f"| {p['platform']} | {str(p.get('at') or '')[:10]} | {p.get('lane') or '—'} | {(p.get('title') or p.get('caption') or p.get('url') or '')[:38]} | "
                     f"{_fmt(p.get('views'))} | {_fmt(p.get('likes'))} | {_fmt(p.get('comments'))} | {_fmt(p.get('shares'))} | {_fmt(p.get('saves'))} | "
                     f"{(f'{e * 100:.1f}%' if e is not None else '—')} |")
    lines.append("")
    top, bottom = rank(posts, key)
    if top:
        lines.append(f"## Migliori 5 ({key})")
        lines += [f"- {(p.get('title') or p.get('caption') or p.get('url'))[:60]} — {p['platform']} — {_fmt(p.get(key))}" for p in top]
        lines.append("")
    if bottom:
        lines.append(f"## Peggiori 5 ({key})")
        lines += [f"- {(p.get('title') or p.get('caption') or p.get('url'))[:60]} — {p['platform']} — {_fmt(p.get(key))}" for p in bottom]
        lines.append("")
    lanes = by_lane(posts, key)
    if lanes:
        lines.append(f"## Per corsia (media {key})")
        lines += [f"- {lane}: {_fmt(v['avg'])} su {v['n']} post" for lane, v in sorted(lanes.items())]
        lines.append("")
    hours = by_hour(posts, key, getattr(cfg.auto, "timezone", "Europe/Rome"))
    if hours:
        lines.append(f"## Per ora di pubblicazione (media {key}, ora locale) — ipotesi da testare, non verità")
        lines += [f"- {h:02d}:00 → {_fmt(v['avg'])} su {v['n']} post" for h, v in hours.items()]
        lines.append("")
    aud = last.get("tiktok", {}).get("audience_by_hour") or []
    if any(n for _, n in aud):
        best = sorted(aud, key=lambda x: -x[1])[:4]
        lines.append("## Pubblico TikTok online per ora (fonte: Upload-Post audience)")
        lines.append("- ore migliori: " + ", ".join(f"{h:02d}:00 ({n})" for h, n in best))
        lines.append("")
    missing = []
    if not any(p.get("views") is not None for p in last.get("tiktok", {}).get("posts", [])):
        missing.append("metriche per post TikTok: la cache di Upload-Post si riempie 1-2 giorni dopo la pubblicazione")
    if not ig_perm:
        missing.append("metriche Instagram oltre like/commenti: serve il permesso insights sul token")
    if missing:
        lines.append("## Cosa manca per dire di più")
        lines += [f"- {m}" for m in missing]
        lines.append("")
    text = "\n".join(lines)
    out_path = out_path or (auto_dir(cfg) / f"report-{last['date']}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text
