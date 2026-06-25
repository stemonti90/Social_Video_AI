"""Configuration: dataclasses + YAML loader with env overrides.

Defaults are chosen to be commercial-license-clean and to run on an Apple Silicon Mac
with 24GB unified memory. Copy config.example.yaml -> config.yaml to customize.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from .log import get_logger

log = get_logger("avp.config")


@dataclass
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "gemma4:26b-mlx"   # the ONE script model (Gemma 4, Apache-2.0, native Apple MLX)
    temperature: float = 0.7
    best_of: int = 2             # generate N drafts and keep the LLM-judged best (1 = off); quality lever


@dataclass
class ScriptConfig:
    target_seconds: int = 75     # aim above the 60s TikTok Creator-Rewards minimum
    language: str = "en"         # en | it — language of the spoken narration
    subtitle_language: str | None = None   # if set & != language → translated phrase subtitles (e.g. EN audio, IT subs)
    refine_passes: int = 1       # extra LLM critique→refine passes on the script (0 = single draft)
    normalize_numbers: bool = True   # rewrite digits → spoken words for TTS ("85%"→"ottantacinque per cento")


@dataclass
class FunnelConfig:
    """Drives viewers to the app (objective #2). Appended as a spoken+on-screen endcard."""
    enabled: bool = True
    app_name: str = "AstroStackerPro"
    tagline: str = "Turn your phone into an astrophotography studio."
    url: str = "https://www.astrostackerpro.com/"
    handle: str = "@astrostackerpro"
    cta_line: str = "Want to capture the cosmos yourself? Get {app} — link in the bio."
    cta_button: str = "Get the app  ·  Link in bio"   # endcard pill label; set per channel language


@dataclass
class TTSConfig:
    engine: str = "kokoro"       # kokoro | chatterbox | both (Chatterbox is EN-only + flaky on MPS)
    primary: str = "kokoro"      # which engine feeds the final cut when engine == both
    kokoro_voice: str = "af_heart"
    kokoro_lang: str = "a"       # 'a' = American English, 'b' = British
    chatterbox_ref: str | None = None   # path to a 5-10s reference wav to clone a voice
    device: str = "mps"          # Apple Silicon GPU
    speed: float = 1.0           # Kokoro speech rate; <1 = slower/less rushed (IT auto-eases to 0.94)


@dataclass
class STTConfig:
    engine: str = "parakeet"     # parakeet (MLX, fast & clean) | whisperx | even (no install)
    model: str = "large-v3"
    parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v3"   # v3 = multilingual (EU langs)


@dataclass
class VideoConfig:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    music: str | None = None     # path to a royalty-free track; if None, auto-pick from music_dir
    music_gain_db: float = 0.0     # trim on the loudness-normalized (~-23 LUFS) music bed; 0 = audible under the voice
    music_source: str = "library"  # library (assets/music) | generate (Stable Audio Open) | none
    music_mood: str = "ethereal"   # auto (classify from script tone) | ethereal | cinematic | dark | tense | emotional | documentary
    music_steps: int = 100         # Stable Audio diffusion steps (higher = better, slower)
    music_seconds: float = 45.0    # generated bed length (Stable Audio Open max ≈47s; looped to fit)
    ken_burns: bool = True
    crf: int = 20
    transition: float = 0.4      # crossfade seconds between clips (0 = hard cut)
    segment_gap: float = 0.12    # short breath between segments (not after the last → no dead air)
    trim_silence: bool = True    # edge-trim each segment's TTS silence (continuous narration)
    loudness_lufs: float = -14.0  # EBU R128 normalization target (0 = disable)
    prefer_video: bool = True    # use NASA *video* clips when available, else stills + Ken Burns
    video_seek: float = 3.0      # skip N seconds into video clips (avoids NASA title cards)
    show_credits: bool = True    # burn a small source credit per clip
    footage_relevance_floor: float = 0.35   # min normalized relevance (0-1) of a visual to its segment; below → re-search then fallback
    footage_strict: bool = False            # strict: refuse a below-floor visual outright (generated backdrop) vs best-effort


@dataclass
class CaptionStyle:
    font: str = "Montserrat"     # bundled SIL OFL font (commercial-safe); see assets/fonts/
    fontsize: int = 84
    primary_color: str = "&H00FFFFFF"     # ASS colors are &HAABBGGRR  (white)
    highlight_color: str = "&H0000E5FF"   # amber
    outline: int = 4
    margin_v: int = 320
    group: int = 3                        # words shown per caption line


@dataclass
class PublishConfig:
    """Publishing to socials via Postiz (open-source scheduler; covers TikTok/IG/YouTube + more)."""
    postiz_url: str = "http://localhost:5000"
    postiz_token: str = ""        # or set env AVP_POSTIZ_TOKEN
    platforms: list[str] = field(default_factory=lambda: ["youtube", "tiktok", "instagram"])
    integrations: dict = field(default_factory=dict)  # platform -> Postiz channel id
    voice: str = "kokoro"         # which engine's video to publish


@dataclass
class PathsConfig:
    projects_dir: str = "projects"
    footage_dir: str = "assets/footage"
    music_dir: str = "assets/music"
    # Per-project copy of the shareable outputs (final mp4 + metadata), for quick share/storage.
    # ~ is expanded; set to "" to disable. One subfolder per project: <export_dir>/<slug>/.
    export_dir: str = "~/Desktop/Social AstroStacker"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    script: ScriptConfig = field(default_factory=ScriptConfig)
    funnel: FunnelConfig = field(default_factory=FunnelConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    captions: CaptionStyle = field(default_factory=CaptionStyle)
    publish: PublishConfig = field(default_factory=PublishConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        data: dict = {}
        if path and Path(path).exists():
            try:
                loaded = yaml.safe_load(Path(path).read_text())
            except yaml.YAMLError as e:
                raise RuntimeError(f"Invalid YAML in config file {path}: {e}") from e
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                raise RuntimeError(
                    f"Config file {path} must contain a YAML mapping at the top level, "
                    f"got {type(loaded).__name__}."
                )
            data = loaded

        def _section(name: str, dc):
            """Build one section dataclass, tolerating non-mapping values and IGNORING unknown
            keys (with a warning). This makes a typo'd `config-set` or a config written for a
            different version non-fatal instead of crashing every later command at load time."""
            raw = data.get(name) or {}
            if not isinstance(raw, dict):
                log.warning("config section %r is not a mapping (got %s) — using defaults.",
                            name, type(raw).__name__)
                return dc()
            known = {f.name for f in fields(dc)}
            clean = {k: v for k, v in raw.items() if k in known}
            for k in raw.keys() - clean.keys():
                log.warning("config: ignoring unknown key %s.%s", name, k)
            return dc(**clean)

        cfg = cls(
            llm=_section("llm", LLMConfig),
            script=_section("script", ScriptConfig),
            funnel=_section("funnel", FunnelConfig),
            tts=_section("tts", TTSConfig),
            stt=_section("stt", STTConfig),
            video=_section("video", VideoConfig),
            captions=_section("captions", CaptionStyle),
            publish=_section("publish", PublishConfig),
            paths=_section("paths", PathsConfig),
        )

        # Environment overrides win over the file (but an empty/unset env var must NOT blank a
        # good default — `OLLAMA_HOST=""` should fall back, not override with "").
        cfg.llm.host = os.getenv("OLLAMA_HOST") or cfg.llm.host
        cfg.llm.model = os.getenv("AVP_LLM_MODEL") or cfg.llm.model
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
