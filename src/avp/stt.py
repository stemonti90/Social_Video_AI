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
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import STTConfig
from .ffmpeg import ffprobe_duration
from .log import get_logger

log = get_logger("avp.stt")


def _venv_exe(name: str) -> str:
    """Resolve a console script installed next to the running interpreter (.venv/bin)."""
    cand = Path(sys.executable).parent / name
    return str(cand) if cand.exists() else name


@dataclass
class Word:
    text: str
    start: float
    end: float


def words_even(text: str, duration: float) -> list[Word]:
    tokens = text.split()
    if not tokens:
        return []
    per = duration / len(tokens)
    return [Word(t, i * per, (i + 1) * per) for i, t in enumerate(tokens)]


def _whisperx(audio: Path, cfg: STTConfig, language: str = "en") -> list[Word]:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [_venv_exe("whisperx"), str(audio), "--model", cfg.model, "--language", language,
             "--device", "cpu", "--compute_type", "int8",
             "--output_format", "json", "--output_dir", td],
            check=True, capture_output=True, text=True,   # capture noisy torchcodec/progress output
        )
        data = json.loads(next(Path(td).glob("*.json")).read_text())
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
        )
        data = json.loads(next(Path(td).glob("*.json")).read_text())
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
               duration: float | None = None) -> list[Word]:
    engine = cfg.engine.lower()
    # even-timing must spread the words over the SPOKEN content only; pass `duration` (content
    # length, excluding the silent endcard) so captions don't drift into the endcard.
    dur = duration if duration is not None else _safe_duration(audio, text_fallback)
    try:
        if engine == "whisperx":
            return _whisperx(audio, cfg, language) or words_even(text_fallback, dur)
        if engine == "parakeet":
            return _parakeet(audio, cfg) or words_even(text_fallback, dur)
    except Exception as e:  # noqa: BLE001 — any backend issue should not block the build
        log.warning("STT backend %r failed (%s) — using even timing.", engine, e)
    return words_even(text_fallback, dur)
