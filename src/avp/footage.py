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
    return urllib.parse.quote(url or "", safe="%:/?#[]@!$&'()*+,;=~")


def _get_json(url: str) -> dict | list:
    req = urllib.request.Request(_safe_url(url), headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(_safe_url(url), headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    _verify_download(dest)   # raise on a truncated body / HTML error page → caller tries next source


def _verify_download(dest: Path) -> None:
    """A 200 can still hand back an HTML error page or a truncated body (NASA/Wikimedia hiccups).
    Reject those here so the resolver falls through to the next source (NASA→Wikimedia→backdrop)
    instead of writing a broken file that aborts the whole assemble at clip time."""
    if not dest.exists() or dest.stat().st_size < 1024:        # < 1KB ≈ error page / empty body
        raise ValueError(f"download too small/empty: {dest.name} ({dest.stat().st_size if dest.exists() else 0}B)")
    if dest.suffix.lower() in (".jpg", ".jpeg", ".png"):
        from PIL import Image                                  # already a dep; lazy to keep import light
        with Image.open(dest) as im:
            im.verify()                                        # decodes the header; raises on corruption


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
    kw = seg.keywords if isinstance(seg.keywords, (list, tuple)) else []
    keywords_blob = " ".join(str(k) for k in kw if k is not None)
    for blob in (keywords_blob, seg.visual, script.topic):
        terms += [w.lower() for w in re.findall(r"[A-Za-z]{4,}", str(blob or ""))]
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
    title = (c.get("title") or "").lower()
    desc = (c.get("description") or "").lower()
    score = 0
    for t in terms:                          # weighted relevance: title >> description
        if t in title:
            score += 3
        elif t in desc:
            score += 1
    if _is_diagram(title, desc):             # reuse the already-extracted fields (no KeyError on sparse dicts)
        score -= 5                           # real penalty: push diagrams/figures below photos
    else:
        score += 2                           # cinematic-photo bonus
    return score


def _relevance(c: dict, terms: list[str]) -> float:
    """Normalized 0-1 relevance of a candidate to its segment: the weighted fraction of the segment's
    key terms found in the asset title (full weight) or description (partial). A diagram is capped
    low. This is the text-side relevance the score-floor gates on (a generic 'galaxy' filler against
    a specific segment scores near 0); footage.use_clip can refine it on the pixels when available."""
    if not terms:
        return 0.5                           # nothing to match against → neutral, never blocks
    title = (c.get("title") or "").lower()
    desc = (c.get("description") or "").lower()
    hit = sum(1.0 if t in title else 0.4 if t in desc else 0.0 for t in terms)
    rel = hit / len(terms)
    if _is_diagram(title, desc):
        rel *= 0.4
    return round(min(1.0, rel), 3)


def _title_key(title: str) -> str:
    """Normalized title for dedup. NASA stores the same photo under several consecutive
    nasa_ids (e.g. …e000332/333/334 'Hubble Takes Mars Portrait'), so id-only dedup repeats it."""
    return "t:" + re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _pick(queries: list[str], used_ids: set, terms: list[str], media_type: str) -> tuple[dict | None, float]:
    """Pool candidates across queries (dedup by id AND title, skip used); return the best by relevance
    score together with its normalized 0-1 relevance to the segment."""
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
        return None, 0.0
    best = max(pool.values(), key=lambda c: _score(c, terms))
    return best, _relevance(best, terms)


WIKI_API = "https://commons.wikimedia.org/w/api.php"
GENERIC_QUERIES = ["nebula", "galaxy", "deep space", "star field astronomy", "cosmos"]


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _wm_license_ok(lic: str) -> bool:
    """Commons hosts only free licenses, but be defensive for a MONETIZED channel: reject
    Non-Commercial or No-Derivatives (overlaying captions + Ken Burns makes a derivative)."""
    s = (lic or "").lower()
    return not any(b in s for b in
                   ("non-commercial", "noncommercial", "by-nc", "-nc-", "by-nd", "-nd",
                    "noderiv", "no deriv", "no-deriv", "fair use"))


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
            lic = ext.get("LicenseShortName", {}).get("value", "")
            if not _wm_license_ok(lic):                 # never pull NC / ND media into a monetized video
                continue
            if _is_diagram(title, _strip_html(ext.get("ImageDescription", {}).get("value", ""))):
                continue
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            return {"url": url, "id": uid, "title": title,
                    "credit": _strip_html(ext.get("Artist", {}).get("value", "")) or "Wikimedia Commons",
                    "license": lic}
    return None


def _refined_queries(seg) -> list[str]:
    """Tighter, more segment-specific queries to retry with when the first pick is below the floor:
    each keyword on its own (more targeted than the joined blob) plus the visual cue."""
    kws = [str(k).strip() for k in (seg.keywords or []) if str(k).strip()]
    return [q for q in (kws + [seg.visual]) if q]


def _report_entry(seg, chosen, rel, floor, outcome, rationale) -> dict:
    return {"index": seg.index, "segment": (seg.narration or "")[:120],
            "query": " ".join(str(k) for k in (seg.keywords or [])) or seg.visual,
            "asset": (chosen or {}).get("title", "") if chosen else None,
            "relevance": round(float(rel), 3), "floor": round(float(floor), 3),
            "outcome": outcome, "rationale": rationale}


def _try_nasa(project: VideoProject, seg, queries, terms, used_ids, cfg, report: list) -> bool:
    floor = float(getattr(cfg.video, "footage_relevance_floor", 0.35) or 0.0)
    strict = bool(getattr(cfg.video, "footage_strict", False))
    chosen, is_video, rel = None, False, 0.0
    if cfg.video.prefer_video:
        v, vr = _pick(queries, used_ids, terms, "video")
        if v and _score(v, terms) >= 2:
            chosen, is_video, rel = v, True, vr
    if chosen is None:
        chosen, rel = _pick(queries, used_ids, terms, "image")

    outcome = "accepted"
    if chosen is not None and rel < floor:                 # below floor → re-search tighter, keep the better
        alt, alt_rel = _pick(_refined_queries(seg), used_ids, terms, "image")
        if alt is not None and alt_rel > rel:
            chosen, is_video, rel, outcome = alt, False, alt_rel, "regenerated"

    # no asset, or strict mode refusing a still-below-floor hit → let the caller fall through (Wikimedia → backdrop)
    if chosen is None or (strict and rel < floor):
        report.append(_report_entry(seg, None, rel, floor, "fallback",
                                     "no NASA asset" if chosen is None else "below floor (strict) → fallback"))
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
    if rel < floor and outcome == "accepted":
        outcome = "accepted-below-floor"      # best-effort (non-strict): kept but flagged in the report
    report.append(_report_entry(seg, chosen, rel, floor, outcome, f"relevance {rel:.2f} vs floor {floor:.2f}"))
    log.info("Segment %d ← NASA %s %r (rel %.2f, %s)", seg.index,
             "video" if is_video else "image", chosen["title"], rel, outcome)
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
    report: list[dict] = []      # per-segment relevance audit (text, query, asset, score, outcome)
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
        floor = float(getattr(cfg.video, "footage_relevance_floor", 0.35) or 0.0)
        # Fallback chain — never leave a segment black, never let one segment abort the stage:
        try:
            if _try_nasa(project, seg, queries, terms, used_ids, cfg, report):  # 1) NASA (PD) — also reports
                continue
            if _try_wikimedia(project, seg, queries + GENERIC_QUERIES, used_ids):  # 2) Wikimedia Commons
                report.append(_report_entry(seg, {"title": seg.credit}, 0.0, floor, "fallback",
                                            "NASA below floor → Wikimedia"))
                continue
        except Exception as e:  # noqa: BLE001 — one bad segment must not fail footage
            log.warning("Segment %d: archive lookup error (%s) — using generated backdrop", seg.index, e)
        from . import captions as captions_mod                              # 3) generated backdrop
        dest = fdir / f"{seg.index:02d}.png"
        captions_mod.render_cosmic_backdrop(dest, cfg.video, seg.index)
        seg.footage = dest.name
        seg.credit = ""
        report.append(_report_entry(seg, None, 0.0, floor, "fallback", "no archive match → generated backdrop"))
        log.info("Segment %d ← generated cosmic backdrop (no archive match)", seg.index)
    if report:
        try:
            (project.root / "footage_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False))
        except Exception:  # noqa: BLE001
            pass
    return script
