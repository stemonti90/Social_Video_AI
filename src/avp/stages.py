"""The five pipeline stages, each operating on a VideoProject.

script  -> generate copy with the LLM, write an editable script.md  (HUMAN CHECKPOINT)
voice   -> synthesize narration with the selected TTS engine(s)
footage -> resolve/download a clip per segment
captions-> word-timed ASS karaoke file
assemble-> ffmpeg render to a 9:16 mp4
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import captions as captions_mod
from . import ffmpeg
from . import footage as footage_mod
from . import llm
from . import stt as stt_mod
from . import tts as tts_mod
from .config import Config
from .log import get_logger
from .manifest import VideoProject
from .models import Script, Segment

log = get_logger("avp.stages")


def _json(d: dict) -> str:
    return json.dumps(d, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- script
def emit_script_md(script: Script, path: Path) -> None:
    lines = [
        f"# {script.title}",
        "",
        f"<!-- avp  topic: {script.topic} | target: {script.target_seconds}s | "
        f"disclosure_ai: {str(script.disclosure_ai).lower()} -->",
        "",
        "_Edit the NARRATION lines freely — **check every fact**. Keep the `## n` headers._",
        "",
    ]
    for s in script.segments:
        lines += [
            f"## {s.index}",
            f"NARRATION: {s.narration}",
            f"VISUAL: {s.visual}",
            f"KEYWORDS: {', '.join(s.keywords)}",
            "",
        ]
    path.write_text("\n".join(lines))


def parse_script_md(path: Path, base: Script) -> Script:
    """Overlay human edits from script.md onto the structured base (from script.json).

    Assumes you edit text within the existing `## n` segments (not add/remove segments).
    """
    by_index = {s.index: s for s in base.segments}
    current: int | None = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("## "):
            try:
                current = int(line[3:].strip())
            except ValueError:
                current = None
        elif line.startswith("# "):
            base.title = line[2:].strip()
        elif current in by_index:
            seg = by_index[current]
            upper = line.upper()
            if upper.startswith("NARRATION:"):
                seg.narration = line.split(":", 1)[1].strip()
            elif upper.startswith("VISUAL:"):
                seg.visual = line.split(":", 1)[1].strip()
            elif upper.startswith("KEYWORDS:"):
                seg.keywords = [k.strip() for k in line.split(":", 1)[1].split(",") if k.strip()]
    return base


def load_script(project: VideoProject) -> Script:
    base = Script.from_dict(json.loads(project.script_json.read_text()))
    if project.script_md.exists():
        base = parse_script_md(project.script_md, base)
    return base


def stage_script(project: VideoProject, cfg: Config, topic: str | None) -> Script:
    if not topic:
        topic = project.manifest.data.get("topic") or ""
    if not topic:
        raise ValueError("No topic given and no existing one. Pass --topic.")

    script = llm.generate_script(cfg.llm, topic, cfg.script.target_seconds, language=cfg.script.language)
    if cfg.funnel.enabled:
        script.segments.append(Segment(
            index=len(script.segments) + 1,
            narration=cfg.funnel.cta_line.format(app=cfg.funnel.app_name),
            visual="App endcard", keywords=[], kind="cta"))
    project.script_json.write_text(_json(script.to_dict()))
    emit_script_md(script, project.script_md)

    project.manifest.data["topic"] = topic
    project.manifest.data["title"] = script.title
    project.manifest.data.setdefault("models", {})["llm"] = cfg.llm.model
    project.manifest.mark("script", "done", title=script.title, segments=len(script.segments))
    log.info("Script ready: %r (%d segments)", script.title, len(script.segments))
    return script


# --------------------------------------------------------------------------- voice
def stage_voice(project: VideoProject, cfg: Config) -> Script:
    script = load_script(project)
    providers = tts_mod.get_providers(cfg)
    primary = tts_mod.primary_engine(cfg)
    engines: list[str] = []
    for prov in providers:
        adir = project.audio_dir / prov.name
        adir.mkdir(parents=True, exist_ok=True)
        try:
            seg_paths = []
            for seg in script.segments:
                out = adir / f"{seg.index:02d}.wav"
                if out.exists():   # idempotent: reuse cached audio (delete audio/ to re-synth)
                    log.info("[%s] segment %d/%d (cached)", prov.name, seg.index, len(script.segments))
                else:
                    log.info("[%s] segment %d/%d", prov.name, seg.index, len(script.segments))
                    prov.synthesize(seg.narration, out)
                seg_paths.append(out)
            ffmpeg.concat_audio(seg_paths, adir / "narration.wav", gap=cfg.video.segment_gap)
            engines.append(prov.name)
        except Exception as e:  # noqa: BLE001
            if prov.name == primary:
                raise  # the engine that feeds the cut must work
            log.warning("TTS engine %r failed (%s) — skipping it; primary %r is unaffected.",
                        prov.name, e, primary)

    for seg in script.segments:
        wav = project.audio_dir / primary / f"{seg.index:02d}.wav"
        if wav.exists():
            seg.duration = round(ffmpeg.ffprobe_duration(wav), 3)
    project.script_json.write_text(_json(script.to_dict()))

    total = sum(s.duration or 0 for s in script.segments)
    if total < 60:
        log.warning("Primary narration ≈ %.0fs (<60s) — lengthen the script for TikTok eligibility.", total)

    project.manifest.data.setdefault("models", {})["tts"] = engines
    project.manifest.mark("voice", "done", engines=engines)
    return script


# --------------------------------------------------------------------------- footage
def stage_footage(project: VideoProject, cfg: Config, download: bool = True) -> Script:
    script = load_script(project)
    footage_mod.resolve_footage(project, script, cfg, allow_download=download)
    project.script_json.write_text(_json(script.to_dict()))
    missing = [s.index for s in script.segments if not s.footage]
    project.manifest.mark("footage", "done" if not missing else "partial", missing=missing)
    if missing:
        log.warning("Segments without footage: %s (drop files in %s)", missing, project.footage_dir)
    return script


# --------------------------------------------------------------------------- captions
def _produced_engines(project: VideoProject, cfg: Config) -> list[str]:
    """Engines that actually have narration audio (so we render one video per voice)."""
    candidates = [p.name for p in tts_mod.get_providers(cfg)]
    return [e for e in candidates if (project.audio_dir / e / "narration.wav").exists()]


def stage_captions(project: VideoProject, cfg: Config) -> None:
    script = load_script(project)
    engines = _produced_engines(project, cfg)
    if not engines:
        raise RuntimeError("No narration audio found; run `voice` first.")
    for eng in engines:
        narration = project.audio_dir / eng / "narration.wav"
        words = stt_mod.transcribe(narration, script.narration, cfg.stt, cfg.script.language)
        captions_mod.write_ass(words, project.root / f"captions.{eng}.ass", cfg.captions, cfg.video)
        (project.root / f"captions.{eng}.json").write_text(
            _json([{"text": w.text, "start": w.start, "end": w.end} for w in words]))
        log.info("captions[%s]: %d words", eng, len(words))
    project.manifest.mark("captions", "done", method=cfg.stt.engine, engines=engines)


# --------------------------------------------------------------------------- assemble
def _resolve_music(cfg: Config) -> Path | None:
    if cfg.video.music:
        p = Path(cfg.video.music)
        return p if p.exists() else None
    mdir = Path(cfg.paths.music_dir)
    if mdir.exists():
        for pat in ("*.mp3", "*.m4a", "*.wav", "*.aac"):
            hits = sorted(mdir.glob(pat))
            if hits:
                return hits[0]
    return None


def _assemble_engine(project: VideoProject, cfg: Config, script: Script, eng: str) -> Path | None:
    """Render one final mp4 for a single voice engine, with crossfades, credits and loudness."""
    adir = project.audio_dir / eng
    work = project.root / "work" / eng
    work.mkdir(parents=True, exist_ok=True)
    gap = cfg.video.segment_gap or 0.0
    trans = cfg.video.transition or 0.0

    clips: list[Path] = []
    content_durs: list[float] = []
    segs_used: list[Segment] = []
    for seg in script.segments:
        seg_audio = adir / f"{seg.index:02d}.wav"
        if not seg_audio.exists():
            log.warning("[%s] segment %d: missing audio, skipping", eng, seg.index)
            continue
        content = ffmpeg.ffprobe_duration(seg_audio) + gap     # on-screen time incl. trailing gap
        render_dur = content + trans                           # extra tail for the crossfade
        src = (project.footage_dir / seg.footage) if seg.footage else None
        if not src or not src.exists():
            src = ffmpeg.black_still(work / f"black_{seg.index:02d}.png", cfg.video.width, cfg.video.height)
        clip = work / f"clip_{seg.index:02d}.mp4"
        kb = cfg.video.ken_burns and seg.kind != "cta"         # static endcard, no zoom
        ffmpeg.make_clip(src, render_dur, cfg.video.width, cfg.video.height, cfg.video.fps, kb, clip,
                         seek=cfg.video.video_seek)
        clips.append(clip)
        content_durs.append(content)
        segs_used.append(seg)
    if not clips:
        return None

    video_silent = work / "video.mp4"
    if trans and len(clips) > 1:
        ffmpeg.concat_videos_xfade(clips, content_durs, trans, video_silent)
    else:
        ffmpeg.concat_videos(clips, video_silent)

    audio_mix = work / "audio.m4a"
    ffmpeg.mix_audio(adir / "narration.wav", _resolve_music(cfg), cfg.video.music_gain_db,
                     audio_mix, cfg.video.loudness_lufs)

    out = project.output_for(eng)
    ass = project.root / f"captions.{eng}.ass"
    cap_json = project.root / f"captions.{eng}.json"
    if ass.exists() and ffmpeg.has_filter("subtitles"):   # native libass burn if available
        ffmpeg.mux(video_silent, audio_mix, ass, out, cfg.video.crf, cfg.video.fps)
        log.info("[%s] → %s", eng, out)
        return out

    items: list[dict] = []
    if cap_json.exists():
        from .stt import Word
        words = [Word(**w) for w in json.loads(cap_json.read_text())]
        for png, s, e in captions_mod.render_caption_pngs(
                words, project.root / f"captions_png_{eng}", cfg.captions, cfg.video):
            items.append({"path": png, "start": s, "end": e,
                          "x": "(main_w-overlay_w)/2", "y": f"main_h-overlay_h-{cfg.captions.margin_v}"})
    if cfg.video.show_credits:
        cdir = project.root / f"credits_png_{eng}"
        cdir.mkdir(exist_ok=True)
        t0 = 0.0
        for seg, dur in zip(segs_used, content_durs):
            if seg.kind != "cta" and seg.credit:
                cp = cdir / f"cr_{seg.index:02d}.png"
                captions_mod.render_credit(cp, seg.credit, cfg.video)
                items.append({"path": cp, "start": t0 + 0.1, "end": max(t0 + 0.2, t0 + dur - 0.05),
                              "x": "30", "y": "main_h-overlay_h-30"})
            t0 += dur

    if items:
        ffmpeg.overlay_items(video_silent, audio_mix, items, out, cfg.video.crf, cfg.video.fps)
    else:
        ffmpeg.mux(video_silent, audio_mix, None, out, cfg.video.crf, cfg.video.fps)
    log.info("[%s] → %s", eng, out)
    return out


def stage_assemble(project: VideoProject, cfg: Config) -> Path:
    ffmpeg.ensure_ffmpeg()
    script = load_script(project)
    engines = _produced_engines(project, cfg)
    if not engines:
        raise RuntimeError("No narration audio — did `voice` run?")

    outputs = []
    for eng in engines:
        out = _assemble_engine(project, cfg, script, eng)
        if out:
            outputs.append(str(out))

    # Convenience: <slug>.mp4 mirrors the chosen primary engine.
    primary = tts_mod.primary_engine(cfg)
    primary_out = project.output_for(primary)
    if primary_out.exists():
        shutil.copyfile(primary_out, project.output)

    project.manifest.mark("assemble", "done", outputs=outputs, primary=primary)
    log.info("Outputs ready: %s", outputs)
    return project.output


# --------------------------------------------------------------------------- metadata
def stage_metadata(project: VideoProject, cfg: Config) -> None:
    script = load_script(project)
    meta = llm.generate_metadata(cfg.llm, script, cfg.funnel, cfg.script.language)
    meta["disclosure_ai"] = bool(project.manifest.data.get("disclosure_ai", False))
    (project.root / "metadata.json").write_text(_json(meta))
    _write_metadata_md(project, cfg, meta)
    project.manifest.mark("metadata", "done")
    log.info("Metadata ready: %s", project.root / "metadata.json")


def _write_metadata_md(project: VideoProject, cfg: Config, meta: dict) -> None:
    yt = meta.get("youtube", {})
    tk = meta.get("tiktok", {})
    ig = meta.get("instagram", {})
    lines = [
        f"# Metadata — {project.slug}", "",
        "## YouTube",
        f"**Title:** {yt.get('title', '')}", "",
        yt.get("description", ""), "",
        f"**Tags:** {', '.join(yt.get('tags', []))}", "",
        "## TikTok", tk.get("caption", ""), "",
        "## Instagram", ig.get("caption", ""), "",
        "---",
        f"AI-disclosure required: {meta.get('disclosure_ai', False)}",
        f"App: {cfg.funnel.app_name} — {cfg.funnel.url}",
    ]
    (project.root / "metadata.md").write_text("\n".join(lines))
