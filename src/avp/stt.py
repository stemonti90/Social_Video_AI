"""Word-level timing for karaoke captions.

Three strategies, selectable via stt.engine:
  - "even"     : no extra install — distribute words evenly across the audio length.
                 Good enough to validate the pipeline; not perfectly synced.
  - "whisperx" : accurate forced-alignment (BSD). CPU on Mac, fine for ~1 min clips.
  - "parakeet" : NVIDIA Parakeet on MLX (fast, Apple-Silicon GPU). Weights CC-BY-4.0
                 -> attribute NVIDIA when you use it.
Any backend failure falls back to "even" so the build never blocks.

NOTE: confirm parakeet-mlx CLI flags / JSON schema against the installed version.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import ffmpeg as ffmpeg_mod
from .config import STTConfig
from .ffmpeg import ffprobe_duration
from .log import get_logger

log = get_logger("avp.stt")


def _venv_exe(name: str) -> str:
    """Resolve a console script installed next to the running interpreter (.venv/bin)."""
    cand = Path(sys.executable).parent / name
    return str(cand) if cand.exists() else name


def _exec_env() -> dict:
    """parakeet-mlx and whisperx shell out to `ffmpeg` BY BARE NAME to decode the audio. When avp is
    launched from a GUI app (Electron), child processes inherit a minimal PATH with no ffmpeg, so the
    aligner exits 0 but writes NOTHING (a silent failure → even-timing fallback). Put the resolved
    ffmpeg directory on PATH so the aligner decodes regardless of how avp was started."""
    env = dict(os.environ)
    ff_dir = str(Path(ffmpeg_mod._bin("ffmpeg")).resolve().parent)
    if os.path.isdir(ff_dir) and ff_dir not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = os.pathsep.join([ff_dir, env.get("PATH", "")]).strip(os.pathsep)
    return env


@dataclass
class Word:
    text: str
    start: float
    end: float


# Extra "hold" after a word ending in punctuation, in the same char-unit scale as word length —
# a comma/period naturally lengthens the preceding word, so the karaoke highlight lingers there.
_PUNCT_UNITS = {",": 3, ";": 4, ":": 4, ")": 1, "]": 1, "—": 3, "…": 5, ".": 6, "!": 6, "?": 6}


def words_even(text: str, duration: float) -> list[Word]:
    """Estimate per-word timings when no real aligner is available. Weighted, NOT uniform: each
    word's on-screen time is proportional to its (alphanumeric) length, plus an extra hold after
    trailing punctuation (comma/period/…) — so the karaoke highlight follows the spoken rhythm
    instead of ticking robotically. Words stay contiguous and fill exactly `duration`."""
    tokens = text.split()
    if not tokens or duration <= 0:
        return []
    units: list[float] = []
    for t in tokens:
        alnum = len(re.sub(r"[^\w]", "", t, flags=re.UNICODE))
        u = float(max(2, alnum))                     # floor so 1-char words aren't a flash
        for ch in reversed(t):                       # trailing punctuation → extra hold
            if ch in _PUNCT_UNITS:
                u += _PUNCT_UNITS[ch]
            elif ch.isalnum():
                break
        units.append(u)
    scale = duration / sum(units)
    words: list[Word] = []
    t0 = 0.0
    for tok, u in zip(tokens, units):
        t1 = t0 + u * scale
        words.append(Word(tok, t0, t1))
        t0 = t1
    words[-1].end = duration                          # kill float drift on the last word
    return words


def distribute_words(segments: list[tuple[str, float]]) -> list[Word]:
    """Flatten (text, duration) segments into one contiguous Word stream: each segment's words are
    length-weighted across ITS OWN duration. Gives translated subtitles (which have no real word
    timestamps) per-word karaoke that tracks the spoken audio segment-by-segment."""
    out: list[Word] = []
    t0 = 0.0
    for text, dur in segments:
        for w in words_even(text, dur):
            out.append(Word(w.text, t0 + w.start, t0 + w.end))
        t0 += float(dur)
    return out


def _whisperx(audio: Path, cfg: STTConfig, language: str = "en") -> list[Word]:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [_venv_exe("whisperx"), str(audio), "--model", cfg.model, "--language", language,
             "--device", "cpu", "--compute_type", "int8",
             "--output_format", "json", "--output_dir", td],
            check=True, capture_output=True, text=True,   # capture noisy torchcodec/progress output
            env=_exec_env(),                              # ensure ffmpeg is on PATH (GUI-spawned = minimal)
        )
        files = sorted(Path(td).glob("*.json"))
        if not files:
            raise RuntimeError("whisperx produced no JSON output")
        data = json.loads(files[0].read_text())
    words: list[Word] = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append(Word(str(w.get("word", "")).strip(), float(w["start"]), float(w["end"])))
    return words


def _parakeet(audio: Path, cfg: STTConfig) -> list[Word]:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [_venv_exe("parakeet-mlx"), str(audio), "--model", cfg.parakeet_model,
             "--output-format", "json", "--output-dir", td],
            check=True, capture_output=True, text=True,
            env=_exec_env(),                              # ensure ffmpeg is on PATH (GUI-spawned = minimal)
        )
        files = sorted(Path(td).glob("*.json"))
        if not files:                    # exited 0 but wrote nothing (e.g. OOM-killed mid-write) — was a bare StopIteration
            raise RuntimeError("parakeet-mlx produced no JSON output")
        data = json.loads(files[0].read_text())
    # Parakeet emits sub-word TOKENS; rebuild whole words (a new word begins with a leading space).
    words: list[Word] = []
    cur = None
    for sent in data.get("sentences", []):
        for tok in sent.get("tokens", []):
            t = tok.get("text", "")
            if not t or "start" not in tok:
                continue
            if t.startswith(" ") or cur is None:
                if cur and cur["t"].strip():
                    words.append(Word(cur["t"].strip(), cur["s"], cur["e"]))
                cur = {"t": t, "s": float(tok["start"]), "e": float(tok["end"])}
            else:
                cur["t"] += t
                cur["e"] = float(tok["end"])
        if cur and cur["t"].strip():
            words.append(Word(cur["t"].strip(), cur["s"], cur["e"]))
            cur = None
    return words


def _safe_duration(audio: Path, text: str) -> float:
    """Audio length for even-timing. Never raises: probing must not kill the build."""
    try:
        return ffprobe_duration(audio)
    except Exception as e:  # noqa: BLE001 — ffprobe itself can SIGSEGV on brew's minimal build
        n = max(1, len(text.split()))
        est = n * 0.38   # ~158 wpm — rough but keeps captions (and the build) alive
        log.warning("Duration probe failed (%s) — estimating %.1fs from %d words.", e, est, n)
        return est


def transcribe(audio: Path, text_fallback: str, cfg: STTConfig, language: str = "en",
               duration: float | None = None) -> tuple[list[Word], str]:
    """Return (words, method). method ∈ {"parakeet","whisperx","even"} so callers can record whether
    the karaoke timing is real-aligned or estimated. Any backend failure falls back to even timing."""
    engine = cfg.engine.lower()
    # even-timing must spread the words over the SPOKEN content only; pass `duration` (content
    # length, excluding the silent endcard) so captions don't drift into the endcard.
    dur = duration if duration is not None else _safe_duration(audio, text_fallback)
    for attempt in range(2):                      # one retry: MLX/Metal can transiently fail to allocate
        try:
            if engine == "whisperx":
                real = _whisperx(audio, cfg, language)
                if real:
                    return real, "whisperx"
            elif engine == "parakeet":
                real = _parakeet(audio, cfg)
                if real:
                    return real, "parakeet"
            break                                 # engine is "even", or the backend returned no words
        except Exception as e:  # noqa: BLE001 — any backend issue should not block the build
            if attempt == 0:
                log.warning("STT backend %r failed (%s) — retrying once.", engine, e)
                time.sleep(2)                     # let memory settle before the retry
            else:
                log.warning("STT backend %r failed (%s) — using even timing.", engine, e)
    return words_even(text_fallback, dur), "even"
