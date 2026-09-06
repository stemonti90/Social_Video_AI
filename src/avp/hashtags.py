"""Platform hashtags from a curated bank, not from the model's imagination.

What went wrong. Asked for "layered" Instagram tags the writer produced compounds nobody follows —
#jupiterspot, #stormsystems, #iceanddust — and a TikTok caption with six generic tags and none of
the discovery tags that platform actually indexes (#LearnOnTikTok, #SpaceTok, #ScienceTok). A tag
with no community behind it is decoration; a missing community tag is reach left on the table.

So the tiers that are the same for every video come from a bank (editable in config under
`publish.hashtags`), and the model's job shrinks to the one tier it can do: NARROW tags that name
the exact object, mission or phenomenon of THIS video. Even those are checked — a narrow tag is
kept only if its letters occur in the script (title, narration, keywords), so "#greatredspot" on a
Red Spot video survives and "#jupiterspot" does not.

Per platform:
  * Instagram: broad → mid → narrow (validated) → community, brand last, capped (default 20).
  * TikTok: the discovery core + up to two narrow tags + brand, capped (default 9). Case is kept
    as written in the bank — #LearnOnTikTok reads, #learnontiktok does not.
  * YouTube: untouched here (its tags are a separate field with its own rules).
"""
from __future__ import annotations

import re

BRAND_TAG = "#astrostackerpro"

DEFAULTS: dict = {
    "instagram": {
        "broad": ["#astronomy", "#space", "#universe", "#cosmos", "#nasa"],
        "mid": ["#astrophysics", "#planetaryscience", "#solarsystem", "#spaceexploration",
                "#spacefacts", "#sciencefacts", "#astronomyfacts"],
        "community": ["#astrophotography", "#astronomylovers", "#spacelovers", "#stargazing",
                      "#nightsky", "#telescope", "#amateurastronomy", "#backyardastronomy"],
        "narrow_max": 6,
        "max": 20,
    },
    "tiktok": {
        "core": ["#LearnOnTikTok", "#SpaceTok", "#ScienceTok", "#astronomy", "#space", "#nasa"],
        "narrow_max": 2,
        "max": 9,
    },
}

_TAG = re.compile(r"#\w+")


def bank(overrides: dict | None = None) -> dict:
    """DEFAULTS with per-platform keys replaced by whatever config provides (lists replace, not merge)."""
    out = {p: dict(v) for p, v in DEFAULTS.items()}
    for plat, spec in (overrides or {}).items():
        if isinstance(spec, dict):
            out.setdefault(plat, {}).update(spec)
    return out


def _norm(tag: str) -> str:
    body = "".join(ch for ch in str(tag).strip().lstrip("#") if ch.isalnum() or ch == "_")
    return f"#{body}" if body else ""


def _compact(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def narrow_tags(candidates: list[str], script_text: str, limit: int, taken: set[str]) -> list[str]:
    """The model's tags that name something the script actually says — letters must occur in the
    compacted script text — deduplicated (case-insensitive) against `taken`, first `limit` kept."""
    body = _compact(script_text)
    out: list[str] = []
    for raw in candidates or []:
        tag = _norm(raw)
        key = tag.lower()
        if len(tag) < 4 or key in taken or key == BRAND_TAG:
            continue
        if body and key.lstrip("#") in body:
            out.append(tag)
            taken.add(key)
            if len(out) >= limit:
                break
    return out


def _dedupe(tags: list[str], taken: set[str]) -> list[str]:
    out = []
    for t in tags:
        t = _norm(t)
        if t and t.lower() not in taken:
            taken.add(t.lower())
            out.append(t)
    return out


def _strip_tags(caption: str) -> str:
    return re.sub(r"\s{2,}", " ", _TAG.sub("", caption or "")).strip(" \n")


def finalize(data: dict, script_text: str, overrides: dict | None = None) -> dict:
    """Rebuild the Instagram and TikTok tag lists and captions in place from the bank + validated
    narrow tags. The model's proposals are read from `instagram_hashtags` (if still present),
    `instagram.hashtags` and the inline tags of both captions."""
    spec = bank(overrides)
    proposed: list[str] = []
    for src in (data.get("instagram_hashtags"), (data.get("instagram") or {}).get("hashtags")):
        if isinstance(src, list):
            proposed += [str(t) for t in src]
    for plat in ("instagram", "tiktok"):
        d = data.get(plat)
        if isinstance(d, dict):
            proposed += _TAG.findall(d.get("caption", "") or "")
            if isinstance(d.get("hashtags"), list):
                proposed += [str(t) for t in d["hashtags"]]
    data.pop("instagram_hashtags", None)

    ig = data.get("instagram")
    if isinstance(ig, dict):
        s = spec["instagram"]
        taken: set[str] = {BRAND_TAG}
        tags = _dedupe(list(s.get("broad") or []), taken)
        tags += _dedupe(list(s.get("mid") or []), taken)
        tags += narrow_tags(proposed, script_text, int(s.get("narrow_max", 6)), taken)
        tags += _dedupe(list(s.get("community") or []), taken)
        tags = tags[: max(1, int(s.get("max", 20)) - 1)] + [BRAND_TAG]
        ig["hashtags"] = tags
        ig["caption"] = f"{_strip_tags(ig.get('caption', ''))}\n\n{' '.join(tags)}".strip()

    tt = data.get("tiktok")
    if isinstance(tt, dict):
        s = spec["tiktok"]
        taken = {BRAND_TAG}
        tags = _dedupe(list(s.get("core") or []), taken)
        tags += narrow_tags(proposed, script_text, int(s.get("narrow_max", 2)), taken)
        tags = tags[: max(1, int(s.get("max", 9)) - 1)] + [BRAND_TAG]
        tt["hashtags"] = tags
        tt["caption"] = f"{_strip_tags(tt.get('caption', ''))} {' '.join(tags)}".strip()
    return data
