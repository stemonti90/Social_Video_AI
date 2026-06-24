"""Text-to-speech providers behind a common interface.

Two commercial-license-clean engines are implemented and interchangeable:
  - Kokoro   (Apache-2.0)  — tiny, fast, very consistent; great default narrator.
  - Chatterbox (MIT)       — higher expressiveness + optional voice cloning.
Set tts.engine to 'kokoro', 'chatterbox', or 'both' (A/B the same script).

Heavy deps (kokoro/chatterbox/torch) are imported lazily so the rest of the CLI
works even before they're installed.

NOTE: confirm the exact library call signatures against the installed versions on
first run — they are isolated here precisely so they're easy to adjust.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from .config import TTSConfig
from .log import get_logger

log = get_logger("avp.tts")


def split_sentences(text: str, max_chars: int = 300) -> list[str]:
    """Chunk long text on sentence boundaries so engines stay within comfortable lengths."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    cur = ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = f"{cur} {p}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks or [text.strip()]


class TTSProvider(ABC):
    name = "base"
    sample_rate = 24000

    @abstractmethod
    def synthesize(self, text: str, out_path: Path) -> None:
        """Render `text` to a wav at `out_path`."""


class KokoroProvider(TTSProvider):
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, lang_code: str = "a", voice: str = "af_heart", device: str = "mps",
                 speed: float = 1.0):
        self.lang = lang_code
        self.voice = voice
        self.speed = speed
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            from kokoro import KPipeline  # lazy
            self._pipe = KPipeline(lang_code=self.lang, repo_id="hexgrad/Kokoro-82M")
        return self._pipe

    def synthesize(self, text: str, out_path: Path) -> None:
        import numpy as np
        import soundfile as sf

        pipe = self._pipeline()
        chunks = []
        for _gs, _ps, audio in pipe(text, voice=self.voice, speed=self.speed):
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(arr.astype("float32"))
        data = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
        sf.write(str(out_path), data, self.sample_rate)


# language -> (Kokoro lang_code, default voice). EN voice (af_heart) is native; IT (if_sara) is
# the Italian voice. For an EN-voice video with Italian subtitles set language=en + subtitle_language=it.
LANG_KOKORO = {"en": ("a", "af_heart"), "it": ("i", "if_sara")}


def get_providers(cfg) -> list[TTSProvider]:
    """Kokoro only (Apache-2.0, commercial-clean). Chatterbox was removed (English-only + flaky on
    MPS); any legacy engine value just falls back to Kokoro."""
    language = getattr(cfg.script, "language", "en")
    lang_code, voice = LANG_KOKORO.get(language, LANG_KOKORO["en"])
    # Italian reads more naturally a touch slower; an explicit cfg.tts.speed always wins.
    speed = cfg.tts.speed if cfg.tts.speed != 1.0 else (0.94 if language == "it" else 1.0)
    if cfg.tts.engine.lower() not in ("kokoro", ""):
        log.info("Voice engine %r is no longer available — using Kokoro.", cfg.tts.engine)
    return [KokoroProvider(lang_code, voice, cfg.tts.device, speed)]


def primary_engine() -> str:
    """The single shipped voice engine (Chatterbox was removed). No arg — callers used to pass
    inconsistent values (cfg vs cfg.tts) that were silently ignored."""
    return "kokoro"
