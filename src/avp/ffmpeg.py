"""Thin wrappers around ffmpeg/ffprobe. We drive ffmpeg in small, debuggable passes
rather than one giant filtergraph."""
from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path

from .log import get_logger

log = get_logger("avp.ffmpeg")

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe not found — run ./setup.sh or `brew install ffmpeg`.")


def run(args: list[str], retries: int = 3) -> None:
    """Run an ffmpeg pass. brew's minimal ffmpeg transiently SIGSEGVs under load, so we
    retry on signal death (negative returncode) but fail fast on a genuine ffmpeg error."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
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
        time.sleep(0.6)
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
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
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


def concat_audio(parts: list[Path], out: Path, gap: float = 0.0) -> None:
    """Concatenate wavs; with gap>0 append that many seconds of silence after each part."""
    if gap and gap > 0:
        inputs = []
        for p in parts:
            inputs += ["-i", str(p)]
        n = len(parts)
        fc = "".join(f"[{i}:a]apad=pad_dur={gap}[a{i}];" for i in range(n))
        fc += "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[a]"
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
        # Scale up first so the slow zoom stays sharp, then zoompan, then fit to frame.
        vf = (f"scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
              f"zoompan=z='min(zoom+0.0008,1.15)':d={frames}:s={w}x{h}:fps={fps},setsar=1")
    else:
        vf = f"{cover},setsar=1"
    run(["-loop", "1", "-i", str(src), "-t", f"{duration:.3f}", "-vf", vf,
         "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)])


def mix_audio(voice: Path, music: Path | None, music_gain_db: float, out: Path,
              loudness_lufs: float = -14.0) -> None:
    """Mux narration with optional ducked music, then EBU R128 loudness-normalize."""
    norm = f"loudnorm=I={loudness_lufs}:TP=-1.5:LRA=11" if loudness_lufs else "anull"
    if music and music.exists():
        fc = (f"[1:a]volume={music_gain_db}dB[m];"
              f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=2[mx];"
              f"[mx]{norm}[a]")
        run(["-i", str(voice), "-stream_loop", "-1", "-i", str(music),
             "-filter_complex", fc, "-map", "[a]", "-c:a", "aac", "-b:a", "192k", str(out)])
    else:
        run(["-i", str(voice), "-af", norm, "-c:a", "aac", "-b:a", "192k", str(out)])


@functools.lru_cache(maxsize=None)
def has_filter(name: str) -> bool:
    """True if this ffmpeg build exposes the given filter (e.g. 'subtitles' needs libass)."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                             capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001
        return False
    return re.search(rf"(?m)^\s*\S+\s+{re.escape(name)}\s", out) is not None


def mux(video: Path, audio: Path, ass: Path | None, out: Path, crf: int) -> None:
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
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out)])


def overlay_timed(video: Path, audio: Path, items, out: Path, margin_v: int, crf: int) -> None:
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
         "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out)])


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


def overlay_items(video: Path, audio: Path, items: list[dict], out: Path, crf: int) -> None:
    """Overlay timed PNGs (captions + credits) and mux audio in one pass.

    items: [{path, start, end, x, y}] where x/y are ffmpeg overlay position expressions.
    """
    if not items:
        mux(video, audio, None, out, crf)
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
         "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out)])
