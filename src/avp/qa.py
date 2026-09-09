"""The pre-publish checklist, run by the machine, and a hard gate: a video that fails is not approved.

Why. The Venus and six other back-catalogue videos went out on TikTok — and the James Webb one on
both platforms — WITHOUT the AstroStackerPro endcard: their CTA segment had lost its picture and the
assembler painted black for the last seconds. Every earlier check looked at the script, the audio and
the subtitles; nobody looked at the last frame of the finished file. This module does exactly that,
on the mp4 that is about to be uploaded, and `publish --go` refuses when anything fundamental fails.

Deterministic on purpose: pixels and probes, no model. Every check names what it measured.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

MIN_SECONDS, MAX_SECONDS = 20.0, 60.0
ENDCARD_MAX_DIFF = 28.0       # mean |frame − endcard| on a 270×480 greyscale; a black or photo tail scores 60-120


def _probe(mp4: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(mp4)],
                         capture_output=True, text=True).stdout or "{}"
    d = json.loads(out)
    s = (d.get("streams") or [{}])[0]
    return {"width": int(s.get("width") or 0), "height": int(s.get("height") or 0),
            "fps": s.get("r_frame_rate", ""), "duration": float((d.get("format") or {}).get("duration") or 0)}


def _frame(mp4: Path, t: float, out: Path, size: str = "270:480"):
    from PIL import Image
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(mp4), "-frames:v", "1",
                    "-vf", f"scale={size}", str(out)], check=False)
    return Image.open(out).convert("L") if out.exists() else None


def frames_match(frame, reference) -> tuple[bool, float]:
    """Is `frame` the `reference` picture? Mean absolute greyscale difference at 270×480."""
    from PIL import ImageChops, ImageStat
    ref = reference.convert("L").resize(frame.size)
    diff = ImageStat.Stat(ImageChops.difference(frame, ref)).mean[0]
    return diff <= ENDCARD_MAX_DIFF, diff


def endcard_present(mp4: Path, endcard_png: Path, work: Path) -> tuple[bool, str]:
    """The last second of the video must show the endcard — the brand's official ending."""
    from PIL import Image
    if not endcard_png.exists():
        return False, f"endcard image missing ({endcard_png.name})"
    dur = _probe(mp4)["duration"]
    frame = _frame(mp4, dur - 0.6, work / "qa_last.png")
    if frame is None:
        return False, "could not read the last frame"
    ok, diff = frames_match(frame, Image.open(endcard_png))
    return ok, f"last frame vs endcard: mean diff {diff:.1f} (limit {ENDCARD_MAX_DIFF:.0f})"


def watermark_contrast(frame, wm_rgba, x: int, y: int) -> float:
    """How much brighter the watermark's text pixels are than the plate around them, in the frame region
    where the mark was drawn. The mark is translucent white, so an absolute threshold fails on dark skies
    and bright deserts alike; the text-minus-surround contrast does not."""
    import numpy as np
    ow, oh = wm_rgba.size
    region = np.asarray(frame.convert("L").crop((x, y, x + ow, y + oh)), dtype=float)
    alpha = np.asarray(wm_rgba.split()[-1], dtype=float)
    if region.shape != alpha.shape:
        return 0.0
    # the mark is drawn at video.watermark_opacity, so its alpha tops out around 0.55×255: thresholds
    # are relative to the mask's own maximum, not absolute
    top = float(alpha.max()) or 1.0
    text, plate = region[alpha > 0.6 * top], region[alpha < 0.08 * top]
    if text.size == 0 or plate.size == 0:
        return 0.0
    return float(text.mean() - plate.mean())


def watermark_present(mp4: Path, cfg, work: Path, eng: str = "kokoro") -> tuple[bool, str]:
    """The brand handle sits top-right on every spoken frame: its text is brighter than what surrounds
    it, in two frames far apart (backgrounds change, the watermark does not)."""
    from PIL import Image
    wm_png = work.parent / f"watermark_{eng}.png"
    if not wm_png.exists():
        return False, f"watermark image missing ({wm_png.name})"
    wm = Image.open(wm_png).convert("RGBA")
    w, h = cfg.video.width, cfg.video.height
    x, y = w - wm.size[0] - int(w * 0.05), int(h * 0.075)
    dur = _probe(mp4)["duration"]
    contrasts = []
    for t in (2.0, max(2.5, min(dur * 0.55, dur - 8.0))):
        frame = _frame(mp4, t, work / f"qa_wm_{int(t)}.png", size=f"{w}:{h}")
        if frame is None:
            return False, "could not read a frame"
        contrasts.append(watermark_contrast(frame, wm, x, y))
    ok = all(c >= 12.0 for c in contrasts)
    return ok, "watermark text vs surround: " + ", ".join(f"+{c:.0f}" for c in contrasts) + " luminance"


def check(project, cfg, platforms: list[str] | None = None) -> list[str]:
    """Everything fundamental about the file and its captions. Returns the failures (empty = approved)."""
    problems: list[str] = []
    slug = project.root.name
    mp4 = project.root / f"{slug}.mp4"
    work = project.root / "work"
    work.mkdir(exist_ok=True)
    if not mp4.exists():
        return [f"final video missing: {mp4.name}"]
    p = _probe(mp4)
    if not (MIN_SECONDS <= p["duration"] <= MAX_SECONDS):
        problems.append(f"duration {p['duration']:.1f}s outside {MIN_SECONDS:.0f}-{MAX_SECONDS:.0f}s")
    if (p["width"], p["height"]) != (cfg.video.width, cfg.video.height):
        problems.append(f"format {p['width']}x{p['height']}, expected {cfg.video.width}x{cfg.video.height}")
    # the official ending
    if getattr(cfg.funnel, "enabled", True):
        cta = [s for s in _segments(project) if s.get("kind") == "cta"]
        card = project.footage_dir / (cta[0].get("footage") or f"{cta[0]['index']:02d}.png") if cta else None
        if card is None:
            problems.append("no CTA segment in the script — the official ending is missing")
        else:
            ok, why = endcard_present(mp4, card, work)
            if not ok:
                problems.append(f"official AstroStackerPro ending NOT on screen at the end ({why})")
    if getattr(cfg.video, "watermark", True):
        ok, why = watermark_present(mp4, cfg, work)
        if not ok:
            problems.append(f"watermark not detected ({why})")
    # words
    from .llm import morbid_in_script
    from .subtitles import italian_lint
    script = _script_dict(project)
    if script and (mw := morbid_in_script(script)):
        problems.append(f"morbid word in the script: {mw!r}")
    problems += context_problems(script)
    sub_lang = getattr(cfg.script, "subtitle_language", None)
    if sub_lang and sub_lang != cfg.script.language:
        subs = project.root / f"subtitles.{sub_lang}.json"
        if not subs.exists():
            problems.append("translated subtitles missing")
        elif sub_lang == "it":
            for d in json.loads(subs.read_text()):
                for prob in italian_lint(d.get("text", "")):
                    problems.append(f"Italian subtitle: {prob}")
    fc = project.root / "factcheck.json"
    if fc.exists():
        try:
            rep = json.loads(fc.read_text())
            open_wrong = [f for f in rep.get("findings", []) if f.get("verdict") == "wrong" and not f.get("applied")]
            if open_wrong:
                problems.append(f"{len(open_wrong)} fact-check finding(s) marked wrong and not applied")
        except Exception:  # noqa: BLE001
            pass
    # captions and hashtags per platform
    meta_path = project.root / "metadata.json"
    if not meta_path.exists():
        problems.append("metadata.json missing (no captions)")
    else:
        meta = json.loads(meta_path.read_text())
        for plat in platforms or ["instagram", "tiktok"]:
            cap = ((meta.get(plat) or {}).get("caption") or "").strip()
            tags = re.findall(r"#\w+", cap)
            if not cap:
                problems.append(f"{plat}: caption missing")
                continue
            if "#astrostackerpro" not in [t.lower() for t in tags]:
                problems.append(f"{plat}: brand hashtag missing")
            if len(tags) < 3:
                problems.append(f"{plat}: only {len(tags)} hashtag(s)")
            if re.search(r"\b(literally|mind-blowing|incredible)\b", cap, re.I):
                problems.append(f"{plat}: filler word in the caption")
        ig, tt = (meta.get("instagram") or {}).get("caption", ""), (meta.get("tiktok") or {}).get("caption", "")
        if ig and tt and ig.split("\n")[0].strip() == tt.split("#")[0].strip():
            problems.append("Instagram and TikTok captions are identical — adapt them")
    return problems


def context_problems(script: dict | None) -> list[str]:
    """The viewer must know what the video is about by the second line, and no line may be a flash:
    the subject named in segments 1-2 (topic keywords), every content line at least 9 words."""
    from .llm import names_subject
    if not script:
        return []
    segs = [s for s in script.get("segments", []) if s.get("kind") != "cta"]
    out = []
    topic = script.get("topic") or ""
    if topic and segs and not names_subject(" ".join(s.get("narration", "") for s in segs[:2]), topic):
        out.append(f"the subject ({topic!r}) is not named in the first two lines — no context for the viewer")
    short = [s["index"] for s in segs if len(str(s.get("narration", "")).split()) < 9]
    if short:
        out.append(f"line(s) {short} shorter than 9 words (< 4 s on screen)")
    return out


def _script_dict(project) -> dict | None:
    f = project.root / "script.json"
    try:
        return json.loads(f.read_text()) if f.exists() else None
    except Exception:  # noqa: BLE001
        return None


def _segments(project) -> list[dict]:
    d = _script_dict(project)
    return list(d.get("segments") or []) if d else []


def report(project, cfg, platforms: list[str] | None = None) -> str:
    """The checklist as text, PASS/FAIL per item, for `avp qa`."""
    problems = check(project, cfg, platforms)
    lines = ["QA — " + project.root.name]
    lines += [f"  FAIL  {p}" for p in problems]
    lines.append("  " + ("APPROVED" if not problems else f"NOT APPROVED ({len(problems)} problem(s))"))
    return "\n".join(lines)
