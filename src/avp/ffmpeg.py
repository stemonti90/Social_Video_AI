"""Thin wrappers around ffmpeg/ffprobe. We drive ffmpeg in small, debuggable passes
rather than one giant filtergraph."""
from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path

from .log import get_logger

log = get_logger("avp.ffmpeg")

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}

# GUI-launched apps (Finder/Dock, incl. the packaged app) inherit a minimal PATH WITHOUT Homebrew,
# so a bare "ffmpeg" raises FileNotFoundError. Resolve to an absolute path, falling back to the
# usual install dirs, so the engine works the same however it's launched (shell, app, cron).
_FALLBACK_BINDIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin", "/usr/bin")


@functools.lru_cache(maxsize=None)
def _bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for d in _FALLBACK_BINDIRS:
        cand = os.path.join(d, name)
        if os.path.exists(cand):
            return cand
    return name   # not found anywhere — let it fail with a clear error


def ensure_ffmpeg() -> None:
    missing = [n for n in ("ffmpeg", "ffprobe") if not os.path.exists(_bin(n))]
    if missing:
        raise RuntimeError(f"{'/'.join(missing)} not found — run ./setup.sh or `brew install ffmpeg`.")


def run(args: list[str], retries: int = 6) -> None:
    """Run an ffmpeg pass. brew's minimal ffmpeg transiently SIGSEGVs under load, so we
    retry on signal death (negative returncode) with growing backoff (lets memory pressure
    clear) but fail fast on a genuine ffmpeg error."""
    cmd = [_bin("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", *args]
    log.debug("ffmpeg %s", " ".join(args))
    last_rc = 1
    for attempt in range(retries):
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            return
        last_rc = rc
        if rc > 0:   # genuine ffmpeg error (bad args/filter) — retrying won't help
            break
        log.warning("ffmpeg killed by signal %d (attempt %d/%d) — retrying",
                    -rc, attempt + 1, retries)
        time.sleep(0.8 * (attempt + 1))
    raise subprocess.CalledProcessError(last_rc, cmd)


def _wav_duration(path: Path) -> float | None:
    """Read duration straight from the WAV header (stdlib, no subprocess → cannot SIGSEGV).
    brew's minimal ffprobe occasionally SIGSEGVs on a wav; this avoids it entirely."""
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            if rate > 0:
                return w.getnframes() / float(rate)
    except Exception:  # noqa: BLE001 — not a PCM wav we can read; caller falls back to ffprobe
        return None
    return None


def ffprobe_duration(path: Path) -> float:
    if Path(path).suffix.lower() == ".wav":          # fast, crash-proof path for audio
        d = _wav_duration(path)
        if d is not None:
            return d
    last_err = None
    for _ in range(3):   # ffprobe can transiently SIGSEGV under heavy load — retry
        try:
            out = subprocess.run(
                [_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                check=True, capture_output=True, text=True,
            ).stdout
            return float(json.loads(out)["format"]["duration"])
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"ffprobe failed for {path}: {last_err}")


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXT


def _concat_list(parts: list[Path], list_path: Path) -> None:
    list_path.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))


def trim_silence(src: Path, out: Path, thresh_db: float = -40.0, keep: float = 0.06) -> None:
    """Trim leading + trailing near-silence from a wav, keeping a small natural margin, so the
    narration has no dead air at segment edges (and the hook lands in the first frames)."""
    g = (f"silenceremove=start_periods=1:start_threshold={thresh_db}dB:start_silence={keep}:detection=peak,"
         f"areverse,"
         f"silenceremove=start_periods=1:start_threshold={thresh_db}dB:start_silence={keep}:detection=peak,"
         f"areverse")
    run(["-i", str(src), "-af", g, str(out)])


def concat_audio(parts: list[Path], out: Path, gap: float = 0.0) -> None:
    """Concatenate wavs into one continuous narration. With gap>0 insert a short breath AFTER
    every segment EXCEPT the last — so there is never a trailing silence (no dead air)."""
    if gap and gap > 0 and len(parts) > 1:
        inputs = []
        for p in parts:
            inputs += ["-i", str(p)]
        n = len(parts)
        fc = "".join(f"[{i}:a]apad=pad_dur={gap}[a{i}];" for i in range(n - 1))   # no pad on last
        fc += "".join(f"[a{i}]" for i in range(n - 1)) + f"[{n - 1}:a]" + f"concat=n={n}:v=0:a=1[a]"
        run([*inputs, "-filter_complex", fc, "-map", "[a]", str(out)])
    else:
        lst = out.with_suffix(".concat.txt")
        _concat_list(parts, lst)
        run(["-f", "concat", "-safe", "0", "-i", str(lst), str(out)])
        lst.unlink(missing_ok=True)


def concat_videos(parts: list[Path], out: Path) -> None:
    lst = out.with_suffix(".concat.txt")
    _concat_list(parts, lst)
    run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    lst.unlink(missing_ok=True)


def black_still(out: Path, w: int, h: int) -> Path:
    run(["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}", "-frames:v", "1", str(out)])
    return out


def silence(out: Path, seconds: float, rate: int = 24000) -> Path:
    """A silent mono wav (matches Kokoro's 24 kHz) — used for the endcard, which is shown but not
    spoken, so music keeps playing over it while the voice is silent."""
    run(["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", f"anullsrc=r={rate}:cl=mono",
         "-c:a", "pcm_s16le", str(out)])
    return out


def _still_clip_via_pil(src: Path, w: int, h: int, duration: float, fps: int, out: Path) -> None:
    """Cover-crop the image to exactly w×h with PIL (robust, low memory — `draft` decodes big
    JPEGs at reduced size), then encode a static clip with a MINIMAL ffmpeg pass (no scale
    filter). This survives tight memory where ffmpeg's own scaler SIGSEGVs."""
    from PIL import Image, ImageOps
    tmp = Path(str(out) + ".frame.png")
    with Image.open(src) as im:
        im.draft("RGB", (w, h))
        ImageOps.fit(im.convert("RGB"), (w, h), method=Image.LANCZOS).save(tmp)
    try:
        run(["-loop", "1", "-i", str(tmp), "-t", f"{duration:.3f}", "-vf", "setsar=1",
             "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)])
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def make_clip(src: Path, duration: float, w: int, h: int, fps: int,
              ken_burns: bool, out: Path, seek: float = 0.0) -> None:
    """Render one segment clip (image or video) scaled+cropped to w x h for `duration` s."""
    cover = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    if is_video(src):
        vf = f"{cover},fps={fps},setsar=1"
        s = seek
        if seek and seek > 0:               # land ~30% in (past intros), leaving enough tail
            try:
                vdur = ffprobe_duration(src)
                s = min(max(seek, vdur * 0.30), max(0.0, vdur - duration - 0.5))
            except Exception:  # noqa: BLE001
                s = seek
        pre = ["-stream_loop", "-1"]
        if s and s > 0:
            pre += ["-ss", f"{s:.3f}"]
        run([*pre, "-i", str(src), "-t", f"{duration:.3f}",
             "-an", "-vf", vf, "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)])
        return

    if ken_burns:
        frames = max(1, int(duration * fps))
        uw, uh = int(w * 1.5), int(h * 1.5)
        # Pre-cover-crop to a 1.5x intermediate with PIL (low memory) so ffmpeg only has to zoom,
        # not decode+scale a huge NASA image — that scaling is what SIGSEGVs under memory pressure.
        pre = Path(str(out) + ".kb.png")
        try:
            from PIL import Image, ImageOps
            with Image.open(src) as im:
                im.draft("RGB", (uw, uh))
                ImageOps.fit(im.convert("RGB"), (uw, uh), method=Image.LANCZOS).save(pre)
            vf = f"zoompan=z='min(zoom+0.0008,1.15)':d={frames}:s={w}x{h}:fps={fps},setsar=1"
            run(["-loop", "1", "-i", str(pre), "-t", f"{duration:.3f}", "-vf", vf,
                 "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)])
            return
        except Exception as e:  # noqa: BLE001 — zoompan can still SIGSEGV under load; fall back
            log.warning("Ken Burns failed for %s (%s) — rendering a static clip instead.", src.name, e)
        finally:
            try:
                pre.unlink()
            except OSError:
                pass
    # static (or fallback): PIL does the scaling, ffmpeg just encodes — minimal memory, robust.
    _still_clip_via_pil(src, w, h, duration, fps, out)


# Warm up a thin/metallic small-model TTS voice: roll off rumble, tame the harsh ~3 kHz
# "tinny" presence, add a touch of low-mid body, gentle compression for evenness.
VOICE_WARM = ("highpass=f=80,"
              "equalizer=f=2800:width_type=q:w=1.5:g=-5,"     # tame harsh 'tinny' presence (stronger)
              "equalizer=f=200:width_type=q:w=1.0:g=2.5,"     # add warmth/body
              "treble=g=-4.5:f=7000,"                         # high-shelf: soften the metallic top end
              "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=140")


def mix_audio(voice: Path, music: Path | None, music_gain_db: float, out: Path,
              loudness_lufs: float = -14.0) -> None:
    """Warm the narration, then mux it with optional music kept **clearly audible** under the
    voice (the bed is loudness-normalized so even a quiet generated track comes through, then
    only gently ducked), and EBU R128 loudness-normalize the result."""
    norm = f"loudnorm=I={loudness_lufs}:TP=-1.5:LRA=11" if loudness_lufs else "anull"
    if music and music.exists():
        if has_filter("sidechaincompress"):
            try:
                vdur = ffprobe_duration(voice)
            except Exception:  # noqa: BLE001
                vdur = 0.0
            fadeout = f"[mduck]afade=t=out:st={max(0.0, vdur - 2.0):.3f}:d=2[mf];" if vdur > 3.0 else ""
            mlabel = "[mf]" if fadeout else "[mduck]"
            fc = (
                f"[0:a]{VOICE_WARM}[vw];[vw]asplit=2[v0][v1];"          # warmed voice → mix + key
                # normalize the (often low-energy) generated bed to a steady ~-21 LUFS so it sits
                # clearly under the voice, +music_gain_db trim, 2 s fade-in
                f"[1:a]loudnorm=I=-21:TP=-2:LRA=11,volume={music_gain_db}dB,afade=t=in:st=0:d=2[mraw];"
                # GENTLE duck: ~3-4 dB dip under speech, music stays present (no hard pumping)
                f"[mraw][v0]sidechaincompress=threshold=0.1:ratio=2.5:attack=20:release=400:detection=rms:makeup=1[mduck];"
                f"{fadeout}"
                f"[v1]volume=-2dB[vq];"                                  # voice a touch lower vs music
                f"[vq]{mlabel}amix=inputs=2:duration=first:dropout_transition=2:normalize=0[mx];"
                f"[mx]{norm}[a]"
            )
        else:
            fc = (f"[0:a]{VOICE_WARM},volume=-2dB[vw];"
                  f"[1:a]loudnorm=I=-21:TP=-2,volume={music_gain_db}dB[m];"
                  f"[vw][m]amix=inputs=2:duration=first:dropout_transition=2[mx];"
                  f"[mx]{norm}[a]")
        run(["-i", str(voice), "-stream_loop", "-1", "-i", str(music),
             "-filter_complex", fc, "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(out)])
    else:
        run(["-i", str(voice), "-af", f"{VOICE_WARM},{norm}", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(out)])


@functools.lru_cache(maxsize=None)
def has_filter(name: str) -> bool:
    """True if this ffmpeg build exposes the given filter (e.g. 'subtitles' needs libass)."""
    try:
        out = subprocess.run([_bin("ffmpeg"), "-hide_banner", "-filters"],
                             capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001
        return False
    return re.search(rf"(?m)^\s*\S+\s+{re.escape(name)}\s", out) is not None


def _h264_out(crf: int, fps: int = 30) -> list[str]:
    """Platform-friendly H.264/AAC output flags (TikTok/Reels/Shorts): High@4.0, yuv420p,
    bt709 SDR color tags, ~2 s GOP, 48 kHz stereo AAC, +faststart (moov atom up front)."""
    return [
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-profile:v", "high", "-level", "4.0", "-pix_fmt", "yuv420p",
        "-r", str(fps), "-g", str(max(1, fps * 2)),
        "-color_range", "tv", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", "-movflags", "+faststart",
    ]


def mux(video: Path, audio: Path, ass: Path | None, out: Path, crf: int, fps: int = 30) -> None:
    """Combine the silent video + mixed audio, burning captions if the build supports it."""
    vf = []
    if ass and ass.exists():
        if has_filter("subtitles"):
            esc = ass.as_posix().replace("\\", "\\\\").replace(":", r"\:")
            vf = ["-vf", f"subtitles={esc}"]
        else:
            log.warning("ffmpeg has no 'subtitles' filter (built without libass); "
                        "skipping burned captions. Overlay handled separately if available.")
    run(["-i", str(video), "-i", str(audio), *vf,
         "-map", "0:v:0", "-map", "1:a:0", *_h264_out(crf, fps), str(out)])


def overlay_timed(video: Path, audio: Path, items, out: Path, margin_v: int, crf: int, fps: int = 30) -> None:
    """Overlay timed transparent PNGs onto the video and mux audio, in one pass.

    items = [(png_path, start, end), ...]. Used for captions when ffmpeg lacks libass.
    """
    inputs = ["-i", str(video), "-i", str(audio)]
    for png, _s, _e in items:
        inputs += ["-i", str(png)]
    chains, prev = [], "0:v"
    for i, (_png, s, e) in enumerate(items, start=2):  # 0=video, 1=audio, 2.. = pngs
        label = f"v{i}"
        chains.append(
            f"[{prev}][{i}:v]overlay=x=(main_w-overlay_w)/2:"
            f"y=main_h-overlay_h-{margin_v}:enable='between(t,{s:.3f},{e:.3f})'[{label}]"
        )
        prev = label
    run([*inputs, "-filter_complex", ";".join(chains), "-map", f"[{prev}]", "-map", "1:a:0",
         *_h264_out(crf, fps), str(out)])


def concat_videos_xfade(clips: list[Path], content_durs: list[float], transition: float,
                        out: Path) -> None:
    """Crossfade-concatenate clips. Each clip must be rendered at content_durs[i] + transition."""
    if len(clips) == 1:
        run(["-i", str(clips[0]), "-c", "copy", str(out)])
        return
    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]
    chains, prev, offset = [], "0:v", 0.0
    for i in range(1, len(clips)):
        offset += content_durs[i - 1]   # cumulative content where this transition begins
        lbl = f"x{i}"
        chains.append(
            f"[{prev}][{i}:v]xfade=transition=fade:duration={transition}:offset={offset:.3f}[{lbl}]"
        )
        prev = lbl
    run([*inputs, "-filter_complex", ";".join(chains), "-map", f"[{prev}]",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)])


def overlay_items(video: Path, audio: Path, items: list[dict], out: Path, crf: int, fps: int = 30) -> None:
    """Overlay timed PNGs (captions + credits) and mux audio in one pass.

    items: [{path, start, end, x, y}] where x/y are ffmpeg overlay position expressions.
    """
    if not items:
        mux(video, audio, None, out, crf, fps)
        return
    inputs = ["-i", str(video), "-i", str(audio)]
    for it in items:
        inputs += ["-i", str(it["path"])]
    chains, prev = [], "0:v"
    for i, it in enumerate(items, start=2):
        lbl = f"v{i}"
        chains.append(
            f"[{prev}][{i}:v]overlay=x={it['x']}:y={it['y']}:"
            f"enable='between(t,{it['start']:.3f},{it['end']:.3f})'[{lbl}]"
        )
        prev = lbl
    run([*inputs, "-filter_complex", ";".join(chains), "-map", f"[{prev}]", "-map", "1:a:0",
         *_h264_out(crf, fps), str(out)])
