"""Resolve footage for each segment.

Priority:
  1. A file you dropped at projects/<slug>/footage/NN.* (manual override, always wins).
  2. An automatic search of NASA's public-domain image library (images-api.nasa.gov).

NASA media is public domain and commercial-safe; we still record an attribution per
asset in the manifest (good practice + needed if you later mix in CC-BY sources).
"""
from __future__ import annotations

import json
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from .log import get_logger
from .manifest import VideoProject
from .models import Attribution, Script

log = get_logger("avp.footage")

NASA_SEARCH = "https://images-api.nasa.gov/search"
_UA = {"User-Agent": "avp/0.1 (+local pipeline)"}


def _safe_url(url: str) -> str:
    """Percent-encode characters urllib forbids (spaces, control chars) without
    double-encoding already-escaped sequences. NASA hrefs sometimes contain literal
    spaces, e.g. a nasa_id like 'What is a Black Hole'."""
    return urllib.parse.quote(url, safe="%:/?#[]@!$&'()*+,;=~")


def _get_json(url: str) -> dict | list:
    req = urllib.request.Request(_safe_url(url), headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(_safe_url(url), headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _best_media(collection_url: str, video: bool = False) -> str | None:
    """From an item's collection.json (a list of file URLs) pick the best asset."""
    files = _get_json(collection_url)
    if not isinstance(files, list):
        return None
    if video:
        mp4s = [u for u in files if isinstance(u, str) and u.lower().endswith(".mp4")]
        for marker in ("~large", "~medium", "~small", "~preview", "~mobile"):
            for u in mp4s:
                if marker in u.lower():
                    return u
        non_orig = [u for u in mp4s if "~orig" not in u.lower()]   # avoid huge originals
        return (non_orig or mp4s or [None])[0]
    imgs = [u for u in files if isinstance(u, str) and u.lower().endswith((".jpg", ".jpeg", ".png"))]
    for marker in ("~orig", "~large", "~medium"):
        for u in imgs:
            if marker in u:
                return u
    return imgs[0] if imgs else None


# Titles/descriptions containing these read as charts/figures/branding, not cinematic footage.
DIAGRAM_WORDS = (
    "map", "diagram", "plot", "graph", "chart", "spectrum", "spectra", "schematic",
    "infographic", "timeline", "histogram", "logo", "seal", "patch", "insignia",
    "data visualization", "model of", "simulation", "illustration of a",
)


def _is_diagram(title: str, desc: str) -> bool:
    text = f"{title} {desc}".lower()
    return any(w in text for w in DIAGRAM_WORDS)


def _key_terms(seg: "Script", script: Script) -> list[str]:
    """Significant words (>=4 chars) from a segment's keywords/visual/topic, for relevance scoring."""
    terms: list[str] = []
    for blob in (" ".join(seg.keywords), seg.visual, script.topic):
        terms += [w.lower() for w in re.findall(r"[A-Za-z]{4,}", blob or "")]
    return list(dict.fromkeys(terms))


def nasa_candidates(query: str, media_type: str = "image", limit: int = 30) -> list[dict]:
    qs = urllib.parse.urlencode({"q": query, "media_type": media_type})
    data = _get_json(f"{NASA_SEARCH}?{qs}")
    items = data.get("collection", {}).get("items", []) if isinstance(data, dict) else []
    out = []
    for it in items[:limit]:
        meta = (it.get("data") or [{}])[0]
        coll = it.get("href")
        if not coll:
            continue
        out.append({
            "collection": coll,
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "nasa_id": meta.get("nasa_id", ""),
            "center": meta.get("center", "NASA"),
        })
    return out


def _score(c: dict, terms: list[str]) -> int:
    title = c["title"].lower()
    score = 2 if any(t in title for t in terms) else 0   # relevance: subject in title
    if not _is_diagram(c["title"], c["description"]):     # prefer photos over figures
        score += 1
    return score


def _title_key(title: str) -> str:
    """Normalized title for dedup. NASA stores the same photo under several consecutive
    nasa_ids (e.g. …e000332/333/334 'Hubble Takes Mars Portrait'), so id-only dedup repeats it."""
    return "t:" + re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _pick(queries: list[str], used_ids: set, terms: list[str], media_type: str) -> dict | None:
    """Pool candidates across queries (dedup by id AND title, skip used), best by relevance."""
    pool: dict[str, dict] = {}
    seen_titles: set[str] = set()
    for q in queries:
        try:
            for c in nasa_candidates(q, media_type=media_type):
                cid, tk = c["nasa_id"], _title_key(c["title"])
                if not cid or cid in used_ids or tk in used_ids:
                    continue                      # already used on an earlier segment
                if cid in pool or tk in seen_titles:
                    continue                      # duplicate within this pool
                pool[cid] = c
                seen_titles.add(tk)
        except Exception as e:  # noqa: BLE001 — network issues shouldn't crash the run
            log.warning("NASA %s search failed for %r (%s)", media_type, q, e)
        if len(pool) >= 12:
            break
    if not pool:
        return None
    return max(pool.values(), key=lambda c: _score(c, terms))


WIKI_API = "https://commons.wikimedia.org/w/api.php"
GENERIC_QUERIES = ["nebula", "galaxy", "deep space", "star field astronomy", "cosmos"]


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def wikimedia_pick(queries: list[str], used_ids: set) -> dict | None:
    """Search Wikimedia Commons (free media — incl. ESA/Hubble/ESO under free licenses)."""
    for q in queries:
        params = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": q, "gsrnamespace": "6", "gsrlimit": "20",
            "prop": "imageinfo", "iiprop": "url|extmetadata|mime", "iiurlwidth": "1920",
        })
        try:
            data = _get_json(f"{WIKI_API}?{params}")
        except Exception as e:  # noqa: BLE001
            log.warning("Wikimedia search failed for %r (%s)", q, e)
            continue
        pages = (data.get("query", {}) or {}).get("pages", {}) if isinstance(data, dict) else {}
        for p in pages.values():
            ii = (p.get("imageinfo") or [{}])[0]
            mime = ii.get("mime", "")
            if not mime.startswith("image/") or mime in ("image/svg+xml", "image/tiff", "image/gif"):
                continue
            title = p.get("title", "")
            uid = "wm:" + title
            if uid in used_ids:
                continue
            ext = ii.get("extmetadata", {}) or {}
            if _is_diagram(title, _strip_html(ext.get("ImageDescription", {}).get("value", ""))):
                continue
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            return {"url": url, "id": uid, "title": title,
                    "credit": _strip_html(ext.get("Artist", {}).get("value", "")) or "Wikimedia Commons",
                    "license": ext.get("LicenseShortName", {}).get("value", "")}
    return None


def _try_nasa(project: VideoProject, seg, queries, terms, used_ids, cfg) -> bool:
    chosen, is_video = None, False
    if cfg.video.prefer_video:
        v = _pick(queries, used_ids, terms, "video")
        if v and _score(v, terms) >= 2:
            chosen, is_video = v, True
    if chosen is None:
        chosen = _pick(queries, used_ids, terms, "image")
    if chosen is None:
        chosen = _pick(["deep space nebula", "galaxy", "starfield"], used_ids, terms, "image")
    if not chosen:
        return False
    dest = project.footage_dir / f"{seg.index:02d}{'.mp4' if is_video else '.jpg'}"
    try:
        asset = _best_media(chosen["collection"], video=is_video)
        if not asset:
            return False
        _download(asset, dest)
    except Exception as e:  # noqa: BLE001 — any archive hiccup → fall through to next source
        log.warning("Segment %d: NASA fetch failed (%s)", seg.index, e)
        return False
    used_ids.add(chosen["nasa_id"])
    used_ids.add(_title_key(chosen["title"]))   # also block same-photo-different-id reuse
    seg.footage = dest.name
    seg.credit = chosen["center"] or "NASA"
    project.manifest.add_attribution(Attribution(
        source=chosen["center"] or "NASA",
        credit=f"{chosen['title']} — {chosen['center'] or 'NASA'}".strip(" —"),
        license="Public Domain (NASA)", url=asset, asset_id=chosen["nasa_id"]))
    log.info("Segment %d ← NASA %s %r", seg.index, "video" if is_video else "image", chosen["title"])
    return True


def _try_wikimedia(project: VideoProject, seg, queries, used_ids) -> bool:
    wm = wikimedia_pick(queries, used_ids)
    if not wm:
        return False
    dest = project.footage_dir / f"{seg.index:02d}.jpg"
    try:
        _download(wm["url"], dest)
    except Exception as e:  # noqa: BLE001
        log.warning("Segment %d: Wikimedia download failed (%s)", seg.index, e)
        return False
    used_ids.add(wm["id"])
    seg.footage = dest.name
    seg.credit = (wm["credit"] or "Wikimedia")[:48]
    project.manifest.add_attribution(Attribution(
        source="Wikimedia Commons",
        credit=f"{wm['credit']} / {wm['license']}".strip(" /"),
        license=wm["license"] or "Wikimedia (free license)", url=wm["url"], asset_id=wm["id"]))
    log.info("Segment %d ← Wikimedia %r", seg.index, wm["title"])
    return True


def resolve_footage(project: VideoProject, script: Script, cfg, allow_download: bool = True) -> Script:
    fdir = project.footage_dir
    used_ids: set[str] = set()   # dedup the same asset across segments
    for seg in script.segments:
        manual = sorted(fdir.glob(f"{seg.index:02d}.*")) or sorted(fdir.glob(f"{seg.index}.*"))
        if manual:
            seg.footage = manual[0].name
            used_ids.add(manual[0].stem)
            log.info("Segment %d ← manual %s", seg.index, manual[0].name)
            continue
        if seg.kind == "cta":
            from . import captions as captions_mod  # lazy: uses Pillow
            dest = fdir / f"{seg.index:02d}.png"
            captions_mod.render_endcard(dest, cfg.funnel, cfg.video)
            seg.footage = dest.name
            log.info("Segment %d ← app endcard (%s)", seg.index, cfg.funnel.app_name)
            continue
        if not allow_download:
            log.warning("Segment %d has no footage (drop a file at %s/%02d.jpg)",
                        seg.index, fdir, seg.index)
            continue

        terms = _key_terms(seg, script)
        queries = [q for q in (" ".join(seg.keywords), seg.visual, script.topic) if q]
        # Fallback chain — never leave a segment black, never let one segment abort the stage:
        try:
            if _try_nasa(project, seg, queries, terms, used_ids, cfg):          # 1) NASA (PD)
                continue
            if _try_wikimedia(project, seg, queries + GENERIC_QUERIES, used_ids):  # 2) Wikimedia Commons
                continue
        except Exception as e:  # noqa: BLE001 — one bad segment must not fail footage
            log.warning("Segment %d: archive lookup error (%s) — using generated backdrop", seg.index, e)
        from . import captions as captions_mod                              # 3) generated backdrop
        dest = fdir / f"{seg.index:02d}.png"
        captions_mod.render_cosmic_backdrop(dest, cfg.video, seg.index)
        seg.footage = dest.name
        seg.credit = ""
        log.info("Segment %d ← generated cosmic backdrop (no archive match)", seg.index)
    return script
