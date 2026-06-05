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

    def __init__(self, lang_code: str = "a", voice: str = "af_heart", device: str = "mps"):
        self.lang = lang_code
        self.voice = voice
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
        for _gs, _ps, audio in pipe(text, voice=self.voice):
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(arr.astype("float32"))
        data = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
        sf.write(str(out_path), data, self.sample_rate)


class ChatterboxProvider(TTSProvider):
    name = "chatterbox"

    def __init__(self, cfg: TTSConfig):
        self.device = cfg.device
        self.ref = cfg.chatterbox_ref
        self._model = None

    def _load(self):
        if self._model is None:
            from chatterbox.tts import ChatterboxTTS  # lazy
            self._model = ChatterboxTTS.from_pretrained(device=self.device)
            self.sample_rate = self._model.sr
        return self._model

    def synthesize(self, text: str, out_path: Path) -> None:
        import torch
        import torchaudio

        model = self._load()
        wavs = []
        for chunk in split_sentences(text):
            kwargs = {"audio_prompt_path": self.ref} if self.ref else {}
            wav = None
            for attempt in range(3):   # MPS sampling occasionally yields inf/nan — retry
                try:
                    wav = model.generate(chunk, **kwargs)
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 2:
                        raise
            wavs.append(wav)
        wav = torch.cat(wavs, dim=-1) if len(wavs) > 1 else wavs[0]
        torchaudio.save(str(out_path), wav.cpu(), model.sr)


# language -> (Kokoro lang_code, default voice). Chatterbox is English-only.
LANG_KOKORO = {"en": ("a", "af_heart"), "it": ("i", "if_sara")}


def get_providers(cfg) -> list[TTSProvider]:
    """cfg is the full Config. Engines depend on tts.engine AND the script language."""
    language = getattr(cfg.script, "language", "en")
    lang_code, voice = LANG_KOKORO.get(language, LANG_KOKORO["en"])
    kok = lambda: KokoroProvider(lang_code, voice, cfg.tts.device)        # noqa: E731
    cbx = lambda: ChatterboxProvider(cfg.tts)                             # noqa: E731
    engine = cfg.tts.engine.lower()
    if engine == "both":
        provs = [kok()]
        if language == "en":
            provs.append(cbx())
        else:
            log.info("Chatterbox is English-only — using Kokoro for language %r.", language)
        return provs
    if engine == "chatterbox":
        if language != "en":
            log.info("Chatterbox is English-only — using Kokoro for language %r.", language)
            return [kok()]
        return [cbx()]
    if engine == "kokoro":
        return [kok()]
    raise ValueError(f"Unknown tts.engine {cfg.tts.engine!r} (use kokoro | chatterbox | both)")


def primary_engine(cfg) -> str:
    names = [p.name for p in get_providers(cfg)]
    return cfg.tts.primary if cfg.tts.primary in names else (names[0] if names else "kokoro")
