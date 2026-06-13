"""Configuration: dataclasses + YAML loader with env overrides.

Defaults are chosen to be commercial-license-clean and to run on an Apple Silicon Mac
with 24GB unified memory. Copy config.example.yaml -> config.yaml to customize.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "qwen3:14b"     # general writer (Apache-2.0). Any installed Ollama model works.
    temperature: float = 0.7


@dataclass
class ScriptConfig:
    target_seconds: int = 75     # aim above the 60s TikTok Creator-Rewards minimum
    language: str = "en"         # en | it — language of the spoken narration + captions


@dataclass
class FunnelConfig:
    """Drives viewers to the app (objective #2). Appended as a spoken+on-screen endcard."""
    enabled: bool = True
    app_name: str = "AstroStackerPro"
    tagline: str = "Turn your phone into an astrophotography studio."
    url: str = "https://apps.apple.com/app/astrostackerpro"   # TODO: confirm real App Store URL
    handle: str = "@astrostackerpro"
    cta_line: str = "Want to capture the cosmos yourself? Get {app} — link in the bio."


@dataclass
class TTSConfig:
    engine: str = "both"         # kokoro | chatterbox | both
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
    music_gain_db: float = -18.0   # base music level before sidechain ducking under the voice
    music_source: str = "library"  # library (assets/music) | generate (Stable Audio Open) | none
    music_mood: str = "ethereal"   # ethereal | cinematic | dark  (when music_source='generate')
    music_steps: int = 100         # Stable Audio diffusion steps (higher = better, slower)
    music_seconds: float = 40.0    # generated track length (looped + ducked under the voice)
    ken_burns: bool = True
    crf: int = 20
    transition: float = 0.4      # crossfade seconds between clips (0 = hard cut)
    segment_gap: float = 0.22    # silence inserted after each segment (pacing/breath)
    loudness_lufs: float = -14.0  # EBU R128 normalization target (0 = disable)
    prefer_video: bool = True    # use NASA *video* clips when available, else stills + Ken Burns
    video_seek: float = 3.0      # skip N seconds into video clips (avoids NASA title cards)
    show_credits: bool = True    # burn a small source credit per clip


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
            data = yaml.safe_load(Path(path).read_text()) or {}

        cfg = cls(
            llm=LLMConfig(**(data.get("llm") or {})),
            script=ScriptConfig(**(data.get("script") or {})),
            funnel=FunnelConfig(**(data.get("funnel") or {})),
            tts=TTSConfig(**(data.get("tts") or {})),
            stt=STTConfig(**(data.get("stt") or {})),
            video=VideoConfig(**(data.get("video") or {})),
            captions=CaptionStyle(**(data.get("captions") or {})),
            publish=PublishConfig(**(data.get("publish") or {})),
            paths=PathsConfig(**(data.get("paths") or {})),
        )

        # Environment overrides win over the file.
        cfg.llm.host = os.getenv("OLLAMA_HOST", cfg.llm.host)
        cfg.llm.model = os.getenv("AVP_LLM_MODEL", cfg.llm.model)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
