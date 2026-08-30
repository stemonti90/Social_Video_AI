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
    model: str = "gemma4:12b-mlx"   # the ONE script model (Gemma 4, Apache-2.0, native Apple MLX)
    temperature: float = 0.7
    best_of: int = 1             # 1 = quality-via-depth (one script, refined); >1 generates N drafts (off by default)


@dataclass
class ScriptConfig:
    target_seconds: int = 50     # aim above the 60s TikTok Creator-Rewards minimum
    language: str = "en"         # en | it — language of the spoken narration
    subtitle_language: str | None = None   # if set & != language → translated phrase subtitles (e.g. EN audio, IT subs)
    refine_passes: int = 2       # critique→refine rounds on the ONE script (quality via depth, never regresses)
    normalize_numbers: bool = True   # rewrite digits → spoken words for TTS ("85%"→"ottantacinque per cento")
    # Adversarial fact-check by a stronger model before anything is rendered. The local writer is
    # fluent and unreliable — it invents mechanisms and writes about dead missions in the present
    # tense — and no amount of prompt tightening fixes that, because a 12B model cannot audit its own
    # knowledge. See avp/factcheck.py.
    #   "off"  — skip
    #   "flag" — check and report to factcheck.json, change nothing
    #   "fix"  — also apply corrections the checker is CONFIDENT about (never the uncertain ones)
    factcheck: str = "off"
    factcheck_model: str = "deepseek-chat"
    factcheck_key: str = ""      # prefer the DEEPSEEK_API_KEY env var; this is the fallback


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
    # What to do when the writer finds no honest link between the topic and astrophotography
    # (bridge_kind "none" — a cosmology or deep-theory video, say):
    #   "always" — still close on the app, using the generic funnel line. Maximum funnel.
    #   "honest" — close on the sky instead and show only the logo. Protects the channel's voice.
    # Anything the writer bridged as "shoot" or "principle" always closes on the app either way.
    bridge_policy: str = "always"


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
    # Visuals: archive = NASA/Wikimedia only (real, public-domain) · generate = local AI images
    # (mflux z-image-turbo, CLIP-picked) with the archives as fallback. NOTE: with `generate` the
    # visuals are synthetic — set publish.disclose_ai true so TikTok gets video_made_with_ai.
    # Archive matching: rank a shortlist by text, then let CLIP choose on the actual pixels — NASA
    # titles rarely contain a segment's keywords, so text alone picks filler surprisingly often.
    footage_clip: bool = True               # CLIP-rerank archive candidates (falls back to text)
    footage_clip_candidates: int = 3        # shortlist size downloaded for the rerank
    # CLIP cosine similarity lives on its OWN scale (ViT-B/32: ~0.30+ good, ~0.25 usable, <0.22 poor)
    # — do NOT rescale it into the text-relevance scale, or a mediocre match reads as a great one.
    footage_clip_floor: float = 0.25        # below this the winner is still wrong → next source
    footage_source: str = "archive"         # archive | generate
    image_venv: str = "vendor/mflux-venv"   # venv holding mflux-generate-z-image-turbo
    image_model: str = "vendor/zimage-q4"   # PRE-QUANTIZED model dir (mflux-save -q 4); on-the-fly
    #                                         quantizing peaks ~28GB and takes ~12min/image on 24GB
    image_candidates: int = 2               # candidates per segment; CLIP ranks them
    # How many of those candidates end up ON SCREEN, splitting the segment's duration with hard cuts.
    # 1 image per ~10s segment read as slow; 2 halves the shot length at zero extra generation cost
    # (the runner-up was already generated and thrown away).
    images_per_segment: int = 2
    image_steps: int = 8                    # z-image-turbo is a turbo model: 8 is its sweet spot
    image_width: int = 720                  # 720x1280 ≈ 74s/image; 512x896 ≈ 25s (upscaled at render)
    image_height: int = 1280
    image_seed: int = 100
    image_timeout: int = 900                # per-image cap, so a hung generator can't stall a build


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
    """Publishing to socials.

    Two backends. ``native`` (default) talks to TikTok / Meta / Google directly: no extra services to
    run, and it asks TikTok for three scopes instead of the six Postiz requires — which matters,
    because TikTok's review delays apps that request scopes they don't exercise. ``postiz`` keeps the
    old path for the other ~18 networks Postiz supports.
    """
    backend: str = "native"       # native | postiz
    # Per-platform OAuth app credentials. Env wins (AVP_TIKTOK_CLIENT_KEY, AVP_META_APP_ID, …); this is
    # the fallback for a config-driven setup. config.yaml is gitignored, but prefer the environment.
    apps: dict = field(default_factory=dict)      # {"tiktok": {"client_id": …, "client_secret": …}}
    # Instagram's API fetches the video from a URL instead of accepting an upload, so a Reel needs the
    # MP4 briefly reachable in public. {"ssh": …, "dir": …, "url": …} — see social/hosting.py.
    media_host: dict = field(default_factory=dict)
    postiz_url: str = "http://localhost:4007/api"   # self-host public API base = NEXT_PUBLIC_BACKEND_URL
    postiz_token: str = ""        # or set env AVP_POSTIZ_TOKEN
    platforms: list[str] = field(default_factory=lambda: ["youtube", "tiktok", "instagram"])
    integrations: dict = field(default_factory=dict)  # platform -> Postiz integration (channel) id; {} = auto-discover
    voice: str = "kokoro"         # which engine's video to publish
    privacy: str = "public"       # public | unlisted | private (mapped per platform)
    made_for_kids: bool = False   # YouTube selfDeclaredMadeForKids
    # TikTok's video_made_with_ai flag. The pipeline's footage is REAL (NASA/Wikimedia), but the voice
    # and script are AI-generated — set this true if you want to disclose AI on TikTok (your call).
    disclose_ai: bool = False
    short_link: bool = False      # let Postiz shorten links in the caption


@dataclass
class PathsConfig:
    projects_dir: str = "projects"
    footage_dir: str = "assets/footage"
    music_dir: str = "assets/music"
    # Per-project copy of the shareable outputs (final mp4 + metadata), for quick share/storage.
    # ~ is expanded; set to "" to disable. One subfolder per project: <export_dir>/<slug>/.
    export_dir: str = "~/Desktop/Social AstroStacker"


@dataclass
class AutoConfig:
    """Unattended daily pipeline: generate N videos and schedule them to Postiz at set local times.
    Topics come from a queue file (one per line); when it runs low the LLM proposes fresh, deduped
    ones. Publishing only happens for platforms whose channel is actually connected in Postiz —
    otherwise the videos are still built and left ready. Runs on Apple Silicon (the models are MLX)."""
    count: int = 3                       # videos generated per run
    platforms: list[str] = field(default_factory=lambda: ["tiktok", "instagram"])
    post_times: list[str] = field(default_factory=lambda: ["12:00", "18:00", "21:00"])  # local HH:MM
    timezone: str = "Europe/Rome"        # IANA tz the post_times are expressed in
    queue_path: str = "topics.txt"       # topic queue (one per line); relative paths → projects_dir
    refill_threshold: int = 6            # refill the queue when it holds fewer than this many topics
    refill_batch: int = 12               # how many topics the LLM proposes per refill
    theme: str = "space and astronomy"   # editorial theme the LLM brainstorms topics within


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
    auto: AutoConfig = field(default_factory=AutoConfig)

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
            auto=_section("auto", AutoConfig),
        )

        # Environment overrides win over the file (but an empty/unset env var must NOT blank a
        # good default — `OLLAMA_HOST=""` should fall back, not override with "").
        cfg.llm.host = os.getenv("OLLAMA_HOST") or cfg.llm.host
        cfg.llm.model = os.getenv("AVP_LLM_MODEL") or cfg.llm.model
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
