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
from . import normalize as normalize_mod
from . import stt as stt_mod
from . import tts as tts_mod
from .config import Config
from .log import get_logger
from .manifest import VideoProject
from .models import Script, Segment, dedupe_segments

log = get_logger("avp.stages")

ENDCARD_SECONDS = 2.5     # the app endcard is shown (with music) but not spoken, for this long


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
    n = len(base.segments)                        # coherence guard at build time (any script source)
    base.segments = dedupe_segments(base.segments)
    if len(base.segments) < n:
        log.info("Coherence check: dropped %d duplicate segment(s) from the script.", n - len(base.segments))
    return base


def stage_script(project: VideoProject, cfg: Config, topic: str | None) -> Script:
    if not topic:
        topic = project.manifest.data.get("topic") or ""
    if not topic:
        raise ValueError("No topic given and no existing one. Pass --topic.")

    script = llm.generate_script(cfg.llm, topic, cfg.script.target_seconds,
                                 language=cfg.script.language, refine_passes=cfg.script.refine_passes,
                                 best_of=getattr(cfg.llm, "best_of", 1))
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
    primary = tts_mod.primary_engine()
    engines: list[str] = []
    for prov in providers:
        adir = project.audio_dir / prov.name
        adir.mkdir(parents=True, exist_ok=True)
        try:
            seg_paths = []
            do_trim = getattr(cfg.video, "trim_silence", True)
            for seg in script.segments:
                out = adir / f"{seg.index:02d}.wav"   # canonical, edge-trimmed segment audio
                if out.exists():   # idempotent: reuse cached audio (delete audio/ to re-synth)
                    log.info("[%s] segment %d/%d (cached)", prov.name, seg.index, len(script.segments))
                elif seg.kind == "cta":
                    ffmpeg.silence(out, ENDCARD_SECONDS)   # endcard is shown, NOT spoken
                    log.info("[%s] segment %d/%d (silent endcard)", prov.name, seg.index, len(script.segments))
                else:
                    log.info("[%s] segment %d/%d", prov.name, seg.index, len(script.segments))
                    raw = adir / f"{seg.index:02d}.raw.wav"
                    # speechText: digits → spoken words so the TTS never reads "1665" letter-by-letter
                    speech = normalize_mod.segment_speech(
                        seg.narration, cfg.script.language, cfg.script.normalize_numbers)
                    prov.synthesize(speech, raw)
                    if do_trim:
                        try:
                            ffmpeg.trim_silence(raw, out)   # kill per-segment dead air at the source
                            raw.unlink(missing_ok=True)
                        except Exception as e:  # noqa: BLE001
                            log.warning("[%s] trim seg %d failed (%s) — using raw", prov.name, seg.index, e)
                            raw.replace(out)
                    else:
                        raw.replace(out)
                seg_paths.append(out)
            ffmpeg.concat_audio(seg_paths, adir / "narration.wav", gap=cfg.video.segment_gap)
            engines.append(prov.name)
        except Exception as e:  # noqa: BLE001
            log.warning("TTS engine %r failed (%s) — skipping it.", prov.name, e)

    if not engines:
        raise RuntimeError("All TTS engines failed — no narration was produced.")
    if primary not in engines:   # primary broke (e.g. Chatterbox) but another worked → use that
        log.warning("Primary voice %r unavailable — using %r for the final cut instead.",
                    primary, engines[0])
        primary = engines[0]

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
    content_dur = sum(s.duration for s in script.segments if s.kind != "cta" and s.duration) or None
    # The audio was synthesized from speechText (digits→words), so the even-timing fallback must
    # split the SAME speechText — otherwise karaoke words wouldn't match what's heard.
    speech_fallback = " ".join(
        normalize_mod.segment_speech(s.narration, cfg.script.language, cfg.script.normalize_numbers)
        for s in script.segments if s.narration.strip() and s.kind != "cta")
    # Translate FIRST — the Ollama model is still warm from the metadata stage — THEN evict it. The STT
    # aligner (parakeet/MLX) and the ~7GB model both want RAM; running them together on a memory-tight
    # Mac makes the aligner's subprocess produce no output and silently fall back to even timing. Freeing
    # the model here gives it headroom. The translated subs feed assemble, not STT, so order is safe.
    sub_lang = cfg.script.subtitle_language
    if sub_lang and sub_lang != cfg.script.language:   # EN audio + translated subtitles (karaoke)
        sub_path = project.root / f"subtitles.{sub_lang}.json"
        if not sub_path.exists():
            trans = llm.translate_segments(cfg.llm, [s.narration for s in script.segments], sub_lang)
            sub_path.write_text(_json([{"index": s.index, "text": t}
                                       for s, t in zip(script.segments, trans)]))
            log.info("Translated %d segments → %s subtitles", len(trans), sub_lang)
    _free_memory(cfg)   # evict the Ollama model so the STT aligner has RAM headroom (see note above)
    for eng in engines:
        narration = project.audio_dir / eng / "narration.wav"
        words, method = stt_mod.transcribe(narration, speech_fallback, cfg.stt, cfg.script.language,
                                           duration=content_dur)
        captions_mod.write_ass(words, project.root / f"captions.{eng}.ass", cfg.captions, cfg.video)
        (project.root / f"captions.{eng}.json").write_text(
            _json([{"text": w.text, "start": w.start, "end": w.end} for w in words]))
        # per-word karaoke debug: word/start/end/duration + whether timing is real-aligned or estimated
        confidence = "aligned" if method in ("parakeet", "whisperx") else "estimated"
        debug = {
            "engine": eng, "method": method, "confidence": confidence,
            "word_count": len(words),
            "total_duration": round(words[-1].end, 3) if words else 0.0,
            "words": [{"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3),
                       "duration": round(w.end - w.start, 3)} for w in words],
            "segments": [{"index": s.index, "narration": s.narration,
                          "speech": normalize_mod.segment_speech(s.narration, cfg.script.language,
                                                                 cfg.script.normalize_numbers),
                          "duration": s.duration}
                         for s in script.segments if s.kind != "cta"],
        }
        (project.root / f"captions.{eng}.debug.json").write_text(_json(debug))
        log.info("captions[%s]: %d words (%s timing)", eng, len(words), confidence)
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


def _music_mood_decision(project: VideoProject, cfg: Config) -> dict:
    """Resolve the music mood. With music_mood='auto' classify it from the script tone (deterministic,
    logged); otherwise honour the configured mood. The effective bed gain follows the mood only when
    the user left video.music_gain_db at its default 0. Persists music_decision.json so the choice
    (mood, rationale, scores, params, voice/music gain) is auditable."""
    from . import music
    configured = str(getattr(cfg.video, "music_mood", "ethereal") or "ethereal")
    if configured.lower() == "auto":
        try:
            text = load_script(project).narration
        except Exception:  # noqa: BLE001
            text = ""
        d = music.classify_mood(text, cfg.script.language)
        log.info("Music mood (auto) → %s — %s", d["mood"], d["rationale"])
    else:
        d = {"mood": configured, "rationale": "configured by video.music_mood", "scores": {},
             "params": music.MOOD_PARAMS.get(configured, {})}
    user_gain = float(getattr(cfg.video, "music_gain_db", 0.0) or 0.0)
    d["gain_db"] = user_gain if user_gain else float(d.get("params", {}).get("gain_db", 0.0))
    try:
        (project.root / "music_decision.json").write_text(_json(d))
    except Exception:  # noqa: BLE001
        pass
    return d


def _music_gain_db(project: VideoProject, cfg: Config) -> float:
    """Effective bed trim: an explicit video.music_gain_db wins; otherwise the auto-mood's suggestion."""
    user = float(getattr(cfg.video, "music_gain_db", 0.0) or 0.0)
    if user:
        return user
    f = project.root / "music_decision.json"
    if f.exists():
        try:
            return float(json.loads(f.read_text()).get("gain_db", 0.0))
        except Exception:  # noqa: BLE001
            return 0.0
    return 0.0


def _resolve_or_generate_music(project: VideoProject, cfg: Config) -> Path | None:
    """Pick the music bed by `video.music_source`: a library track, an ORIGINAL generated
    track (Stable Audio Open, cached once per project), or none. Generation never blocks the
    build — on any failure it falls back to the library."""
    src = getattr(cfg.video, "music_source", "library")
    if src == "none":
        return None
    if src == "generate":
        out = project.root / "music.wav"
        decision = _music_mood_decision(project, cfg)     # logs + persists the mood choice
        if out.exists():
            log.info("Music: reusing generated %s (mood=%s)", out.name, decision["mood"])
            return out
        try:
            import hashlib
            from . import music
            slug = getattr(project, "slug", None) or project.root.name
            seed = int(hashlib.md5(slug.encode()).hexdigest()[:8], 16)   # stable per project
            log.info("Music: generating original track (Stable Audio Open, mood=%s) — "
                     "first time can take a few minutes…", decision["mood"])
            return music.generate_track(out, mood=decision["mood"],
                                        seconds=cfg.video.music_seconds, steps=cfg.video.music_steps,
                                        seed=seed, device=cfg.tts.device)
        except Exception as e:  # noqa: BLE001 — never let music generation block the build
            log.warning("Music generation failed (%s) — falling back to the library.", e)
    return _resolve_music(cfg)


def _assemble_engine(project: VideoProject, cfg: Config, script: Script, eng: str) -> Path | None:
    """Render one final mp4 for a single voice engine, with crossfades, credits and loudness."""
    adir = project.audio_dir / eng
    work = project.root / "work" / eng
    shutil.rmtree(work, ignore_errors=True)   # clear any truncated clips from a prior crashed run
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
    music_bed = _resolve_or_generate_music(project, cfg)   # also writes music_decision.json (mood/gain)
    ffmpeg.mix_audio(adir / "narration.wav", music_bed,
                     _music_gain_db(project, cfg), audio_mix, cfg.video.loudness_lufs)

    out = project.output_for(eng)
    ass = project.root / f"captions.{eng}.ass"
    cap_json = project.root / f"captions.{eng}.json"
    sub_lang = cfg.script.subtitle_language
    want_translated = bool(sub_lang and sub_lang != cfg.script.language)
    if ass.exists() and ffmpeg.has_filter("subtitles") and not want_translated:   # native libass burn
        ffmpeg.mux(video_silent, audio_mix, ass, out, cfg.video.crf, cfg.video.fps)
        log.info("[%s] → %s", eng, out)
        return out

    items: list[dict] = []
    cap_y = f"main_h-overlay_h-{cfg.captions.margin_v}"
    # captions cover the SPOKEN content only — the silent endcard gets no subtitle
    caption_dur = sum(d for seg, d in zip(segs_used, content_durs) if seg.kind != "cta")
    sub_json = (project.root / f"subtitles.{sub_lang}.json") if sub_lang else None
    if want_translated and sub_json and sub_json.exists():   # EN audio + translated KARAOKE subtitles
        trans = {d["index"]: d["text"] for d in json.loads(sub_json.read_text())}
        # The translation has no per-word timestamps, so distribute each segment's translated words
        # across THAT segment's audio window (length-weighted). The karaoke then tracks the EN voice
        # segment-by-segment — so the IT subtitles still "pop" word-by-word.
        words = stt_mod.distribute_words([
            (trans.get(seg.index) or seg.narration, dur)
            for seg, dur in zip(segs_used, content_durs) if seg.kind != "cta"
        ])
        for png, s, e in captions_mod.render_caption_pngs(
                words, project.root / f"subs_png_{eng}_{sub_lang}", cfg.captions, cfg.video,
                total_dur=caption_dur):
            items.append({"path": png, "start": s, "end": e, "x": "(main_w-overlay_w)/2", "y": cap_y})
    elif cap_json.exists():
        from .stt import Word
        words = [Word(**w) for w in json.loads(cap_json.read_text())]
        for png, s, e in captions_mod.render_caption_pngs(
                words, project.root / f"captions_png_{eng}", cfg.captions, cfg.video,
                total_dur=caption_dur):
            items.append({"path": png, "start": s, "end": e, "x": "(main_w-overlay_w)/2", "y": cap_y})
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


def _free_memory(cfg: Config) -> None:
    """Free RAM before the ffmpeg-heavy assemble. ffmpeg SIGSEGVs under memory pressure, and a
    build accumulates GBs (Ollama model ~8GB held from the script stage, torch/TTS caches). Evict
    the Ollama model and release torch's cached MPS memory so clip rendering has headroom. All
    best-effort — reversible (each model reloads on demand) and never raises."""
    try:
        llm.unload_all(cfg.llm)     # evict ALL Ollama models (incl. unrelated user-loaded ones)
    except Exception:  # noqa: BLE001
        pass
    import gc
    gc.collect()
    try:
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def stage_assemble(project: VideoProject, cfg: Config) -> Path:
    ffmpeg.ensure_ffmpeg()
    _free_memory(cfg)            # reclaim RAM (Ollama + torch) so ffmpeg won't SIGSEGV under load
    script = load_script(project)
    engines = _produced_engines(project, cfg)
    if not engines:
        raise RuntimeError("No narration audio — did `voice` run?")

    outputs = []
    for eng in engines:
        out = _assemble_engine(project, cfg, script, eng)
        if out:
            outputs.append(str(out))

    # Convenience: <slug>.mp4 mirrors the primary engine — or the first one that actually produced
    # audio, so a failed/unused engine never leaves the canonical file missing.
    produced = _produced_engines(project, cfg)
    primary = tts_mod.primary_engine()
    if primary not in produced and produced:
        primary = produced[0]
    primary_out = project.output_for(primary)
    if primary_out.exists():
        shutil.copyfile(primary_out, project.output)

    project.manifest.mark("assemble", "done", outputs=outputs, primary=primary)
    log.info("Outputs ready: %s", outputs)
    export_outputs(project, cfg)
    return project.output


def export_outputs(project: VideoProject, cfg: Config) -> Path | None:
    """Copy the shareable artifacts (final mp4 + metadata) into a per-project folder on the
    Desktop (paths.export_dir → <export_dir>/<slug>/), so each finished video is immediately
    ready to share/store. Idempotent; export_dir="" disables it. Never raises — a failed export
    must not fail the build."""
    root = (getattr(cfg.paths, "export_dir", "") or "").strip()
    if not root:
        return None
    try:
        dest = Path(root).expanduser() / project.slug
        dest.mkdir(parents=True, exist_ok=True)
        copied = []
        for src in (project.output, project.root / "metadata.md", project.root / "metadata.json"):
            if src.exists():
                shutil.copy2(src, dest / src.name)
                copied.append(src.name)
        if copied:
            log.info("Exported to %s — %s", dest, ", ".join(copied))
        return dest
    except Exception as e:  # noqa: BLE001 — export is a convenience, never break the build
        log.warning("Could not export outputs to %s (%s).", root, e)
        return None


# --------------------------------------------------------------------------- metadata
def stage_metadata(project: VideoProject, cfg: Config) -> None:
    script = load_script(project)
    meta = llm.generate_metadata(cfg.llm, script, cfg.funnel, cfg.script.language)
    meta["disclosure_ai"] = bool(project.manifest.data.get("disclosure_ai", False))
    (project.root / "metadata.json").write_text(_json(meta))
    _write_metadata_md(project, cfg, meta)
    project.manifest.mark("metadata", "done")
    log.info("Metadata ready: %s", project.root / "metadata.json")
    export_outputs(project, cfg)   # refresh the Desktop folder now that metadata exists


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
