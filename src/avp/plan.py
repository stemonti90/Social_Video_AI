"""The daily plan, in the format the channel's owner asked for — one block of thirteen fields per video.

The plan is not written in advance and hoped for: it is produced FROM the videos the pipeline built
and published that day (title, lane, hook, script, visual structure, duration, CTA, both captions,
both hashtag sets, both posting times), so it is always true. `avp plan` prints it and saves
projects/_auto/plan-YYYY-MM-DD.md; the daily run refreshes it after every video.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

from . import lanes as lanes_mod

OBJECTIVES = {
    "discovery": "Discovery — reach, viralità, nuovi utenti",
    "education": "Value / Education — save, share, follow",
    "product": "Product / Community — fiducia, riconoscibilità, interesse per l'app",
}


def _projects_of(cfg, day: date) -> list[Path]:
    """Projects created (manifest) or published (posts.jsonl) on `day`, oldest first."""
    root = Path(cfg.paths.projects_dir).expanduser()
    picked: dict[str, float] = {}
    for man in root.glob("*/manifest.json"):
        try:                                  # the DAY THE SCRIPT WAS BORN: manifests are rewritten by every
            # stage (fresh inode, fresh birth time); script.json is created once and edited in place
            st = (man.parent / "script.json").stat() if (man.parent / "script.json").exists() else man.stat()
            created = datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime))
        except OSError:
            continue
        if created.date() == day:
            picked[man.parent.name] = created.timestamp()
    log = root / "_auto" / "posts.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                rec = json.loads(line)
                at = datetime.fromisoformat(rec["at"].replace("Z", "+00:00")).astimezone()
            except Exception:  # noqa: BLE001
                continue
            if at.date() == day and (root / rec["slug"]).exists():
                picked.setdefault(rec["slug"], at.timestamp())
    return [root / s for s, _ in sorted(picked.items(), key=lambda kv: kv[1])]


def _posting_times(cfg, slug: str) -> dict[str, str]:
    """When each platform post actually went out (local time), from the publish record."""
    out: dict[str, str] = {}
    log = Path(cfg.paths.projects_dir).expanduser() / "_auto" / "posts.jsonl"
    if not log.exists():
        return out
    for line in log.read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if rec.get("slug") == slug:
            try:
                out[rec["platform"]] = datetime.fromisoformat(rec["at"].replace("Z", "+00:00")).astimezone().strftime("%H:%M")
            except Exception:  # noqa: BLE001
                pass
    return out


def video_block(cfg, root: Path, n: int) -> str:
    slug = root.name
    try:
        script = json.loads((root / "script.json").read_text())
    except Exception:  # noqa: BLE001
        return f"## VIDEO {n} — {slug}\n\n(copione non disponibile)\n"
    try:
        meta = json.loads((root / "metadata.json").read_text())
    except Exception:  # noqa: BLE001
        meta = {}
    try:
        lane = json.loads((root / "manifest.json").read_text()).get("lane", "discovery")
    except Exception:  # noqa: BLE001
        lane = "discovery"
    segs = script.get("segments", [])
    content = [s for s in segs if s.get("kind") != "cta"]
    cta = [s for s in segs if s.get("kind") == "cta"]
    duration = sum(float(s.get("duration") or 0) for s in segs)
    ig = (meta.get("instagram") or {}).get("caption", "") or ""
    tt = (meta.get("tiktok") or {}).get("caption", "") or ""
    ig_body, _, ig_tags = ig.partition("\n\n")
    tt_body = re.split(r"\s#", tt, 1)[0]
    tt_tags = " ".join(re.findall(r"#\w+", tt))
    times = _posting_times(cfg, slug)
    slot = "programmato"
    lines = [f"## VIDEO {n} — {script.get('title', slug)}", "",
             f"- **Titolo/idea**: {script.get('title', '')} — {script.get('topic', '')}",
             f"- **Obiettivo**: {OBJECTIVES.get(lane, lane)}",
             f"- **Hook**: {content[0]['narration'] if content else '—'}",
             "- **Script**:"]
    lines += [f"  {i}. {s['narration']}" for i, s in enumerate(content, 1)]
    if cta:
        lines.append(f"  CTA: {cta[0]['narration']}")
    lines.append("- **Struttura visuale**:")
    lines += [f"  {i}. {s.get('visual', '')}" for i, s in enumerate(content, 1)]
    lines += [f"- **Durata**: {duration:.0f} s",
              f"- **CTA**: Instagram «{lanes_mod.cta(lane, 'instagram', slug)}» · TikTok «{lanes_mod.cta(lane, 'tiktok', slug)}»",
              f"- **Caption Instagram**: {ig_body.strip()}",
              f"- **Caption TikTok**: {tt_body.strip()}",
              f"- **Hashtag Instagram**: {ig_tags.strip() or '—'}",
              f"- **Hashtag TikTok**: {tt_tags or '—'}",
              f"- **Orario Instagram**: {times.get('instagram', slot)}",
              f"- **Orario TikTok**: {times.get('tiktok', slot)}", ""]
    return "\n".join(lines)


def build(cfg, day: date | None = None, out_path: Path | None = None) -> str:
    day = day or date.today()
    roots = _projects_of(cfg, day)
    header = [f"# Piano del {day.isoformat()} — {len(roots)} video", ""]
    if not roots:
        header.append("Nessun video prodotto in questa data.")
    body = [video_block(cfg, r, i) for i, r in enumerate(roots, 1)]
    text = "\n".join(header + body)
    out_path = out_path or (Path(cfg.paths.projects_dir).expanduser() / "_auto" / f"plan-{day.isoformat()}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text
