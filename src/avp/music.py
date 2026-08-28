"""Local **original** music generation via Stable Audio Open (diffusers).

Stability AI Community License → commercial use OK under $1M ARR. The output is original,
so it can never trigger a YouTube Content-ID claim. Heavy + optional: torch/diffusers are
imported lazily, only when music generation is actually requested (music_source='generate').
"""
from __future__ import annotations

from pathlib import Path

from .log import get_logger

log = get_logger("avp.music")

MODEL = "stabilityai/stable-audio-open-1.0"

# Mood → prompt. These are background BEDS under narration, so: instrumental, spacious, and (with the
# sole exception of 'tense', which gets a subtle low pulse) NO drums / NO vocals so they never fight
# the voice. Six moods cover the editorial range; `classify_mood` picks one from the script tone.
PROMPTS = {
    "ethereal": ("Ethereal ambient space music, warm analog synth pads, slow evolving drones, "
                 "deep soft sub bass, gentle twinkling celestial bells, vast peaceful cosmic "
                 "atmosphere, cinematic, smooth, no drums, no percussion, no vocals, warm and clean studio production, high fidelity, seamless slow evolution"),
    "cinematic": ("Cinematic orchestral space ambient, soaring string and choir pads, deep "
                  "cinematic drones, slow majestic build, awe and wonder, film score, lush, "
                  "no percussion, no drums, no vocals, polished film-score production, wide stereo image, high fidelity"),
    "dark": ("Dark mysterious deep-space drone, ominous low synth bass, slow shifting nebula "
             "textures, distant echoes, suspenseful and vast, ambient, no drums, no vocals, clean deep low end, controlled and polished, high fidelity"),
    "tense": ("Tense suspenseful ambient, urgent pulsing low synth bass, subtle ticking heartbeat "
              "pulse, rising dread, dark cinematic underscore, driving but restrained, building "
              "tension, no melody, no vocals, polished cinematic underscore, tight controlled low end, professional film mix, high fidelity"),
    "emotional": ("Emotional cinematic ambient, tender warm piano, soft swelling strings, "
                  "bittersweet hopeful pads, intimate and reflective, slow gentle build, "
                  "no percussion, no drums, no vocals, intimate studio recording quality, warm and clean, high fidelity"),
    "documentary": ("Calm neutral documentary underscore, soft minimal piano and warm pad, steady "
                    "understated, curious and clear, unobtrusive background bed, light, "
                    "no percussion, no drums, no vocals, clean unobtrusive studio production, high fidelity"),
}
NEGATIVE = ("low quality, noisy, distorted, clipping, harsh, lo-fi, abrupt, muddy, hiss, metallic artifacts, dissonant, out of tune, glitchy, sudden cuts")

# Per-mood musical character + a suggested loudnorm trim (dB) on the bed: tenser/epic beds sit a touch
# louder, gentle/neutral ones a touch quieter — always under the voice (the mix also sidechain-ducks).
MOOD_PARAMS = {
    "ethereal":    {"energy": "low",      "bpm": 70,  "percussion": False, "tension": "low",
                    "crescendo": False, "instruments": "synth pads, sub bass, bells", "gain_db": 0.0},
    "cinematic":   {"energy": "high",     "bpm": 90,  "percussion": False, "tension": "medium",
                    "crescendo": True,  "instruments": "strings, choir, drones",      "gain_db": 1.0},
    "dark":        {"energy": "low-med",  "bpm": 60,  "percussion": False, "tension": "high",
                    "crescendo": False, "instruments": "low drone, sub bass",         "gain_db": 0.0},
    "tense":       {"energy": "high",     "bpm": 110, "percussion": True,  "tension": "high",
                    "crescendo": True,  "instruments": "pulse, low bass, underscore", "gain_db": 1.5},
    "emotional":   {"energy": "low-med",  "bpm": 75,  "percussion": False, "tension": "medium",
                    "crescendo": True,  "instruments": "piano, strings, pads",        "gain_db": 0.5},
    "documentary": {"energy": "low",      "bpm": 80,  "percussion": False, "tension": "low",
                    "crescendo": False, "instruments": "minimal piano, pad",          "gain_db": -1.5},
}

# Keyword signals per mood (IT + EN), scored against the script. Tuned for the space channel but
# generic enough for news/crypto pivots. Order matters only as a tie-break (earlier = priority).
_MOOD_KEYWORDS = {
    "tense":       ("pericolo", "impatto", "collision", "collisione", "minaccia", "scontro", "veloce",
                    "corsa", "prima che", "ultima possibilità", "fuga", "emergenza", "rischio", "esplode",
                    "danger", "impact", "threat", "race", "urgent", "before it", "crash", "alert"),
    "dark":        ("buco nero", "black hole", "oscuro", "oscura", "invisibile", "mistero", "misterioso",
                    "nascost", "vuoto", "morte", "morta", "collasso", "divora", "inghiotte", "ombra",
                    "dark", "mystery", "void", "death", "collapse", "devour", "shadow", "sinister"),
    "cinematic":   ("immenso", "gigante", "gigantesca", "enorme", "miliardi", "supernova", "esplosione",
                    "vasto", "colossale", "potente", "spettacolare", "maestoso", "epico",
                    "vast", "giant", "colossal", "explosion", "billions", "epic", "majestic", "powerful"),
    "emotional":   ("casa", "umanità", "solitudine", "soli", "fragile", "perduto", "perso", "speranza",
                    "ultimo respiro", "destino", "home", "humanity", "lonely", "alone", "fragile",
                    "lost", "hope", "fate", "tiny"),
    "ethereal":    ("aurora", "nebulosa", "stella", "luce", "bellezza", "silenzio", "calma", "sogno",
                    "meraviglia", "delicat", "nebula", "aurora borealis", "starlight", "beauty",
                    "calm", "wonder", "gentle", "dream"),
}


def classify_mood(text: str, lang: str = "it", default: str = "documentary") -> dict:
    """Pick a music mood from the script tone by weighted keyword hits. Returns the chosen mood, a
    short rationale (the matched signals), the per-mood scores, and the mood's musical params — all
    deterministic and logged, so the choice is auditable. Falls back to a neutral documentary bed
    when no signal is strong (an inquietante script never gets a cheerful bed; a 'collision/impact'
    one gets tense)."""
    blob = (text or "").lower()
    scores: dict[str, int] = {}
    hits: dict[str, list[str]] = {}
    for mood, kws in _MOOD_KEYWORDS.items():
        matched = [k for k in kws if k in blob]
        if matched:
            scores[mood] = len(matched)
            hits[mood] = matched[:5]
    if not scores:
        chosen, rationale = default, "no strong tonal signal → neutral bed"
    else:
        chosen = max(scores, key=lambda m: (scores[m], -list(_MOOD_KEYWORDS).index(m)))
        rationale = f"matched {', '.join(repr(h) for h in hits[chosen])}"
    return {"mood": chosen, "rationale": rationale, "scores": scores,
            "params": MOOD_PARAMS.get(chosen, MOOD_PARAMS["documentary"])}

_PIPE = None


def _patch_torchsde() -> None:
    """Stable Audio's native scheduler uses torchsde, whose BrownianInterval recurses
    infinitely on Mac: in `_halfway_tree` mode `_split` repeatedly halves [start,end] trying
    to land the *rounded* midpoint exactly on the requested time, which float precision never
    reaches. The non-`halfway_tree` path (`_split_exact(midway)`) does a direct split with NO
    recursion and still yields valid Brownian motion — so we force it. This lets us keep the
    NATIVE cosine-SDE scheduler (correct sigma schedule = correct, coherent audio)."""
    try:
        from torchsde._brownian import brownian_interval as _bi
        if getattr(_bi._Interval._split, "_avp_patched", False):
            return

        def _split(self, midway):
            self._split_exact(midway)

        _split._avp_patched = True
        _bi._Interval._split = _split
        log.info("Patched torchsde BrownianInterval (no recursive split).")
    except Exception as e:  # noqa: BLE001
        log.warning("Could not patch torchsde (%s) — generation may fail.", e)


def _pipe(device: str = "mps", dtype: str = "float32"):
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import StableAudioPipeline
        _patch_torchsde()    # make Stable Audio's native cosine-SDE scheduler usable on Mac
        td = getattr(torch, dtype, torch.float32)   # "float16" halves the ~5GB footprint (A/B-tested)
        log.info("Loading Stable Audio Open (%s, first run downloads ~5 GB)…", dtype)
        _PIPE = StableAudioPipeline.from_pretrained(MODEL, torch_dtype=td).to(device)
        # Keep the NATIVE CosineDPMSolverMultistepScheduler: its cosine sigma schedule is what
        # the model was trained for → correct, coherent output. (Swapping to an EDM solver
        # dodged the torchsde crash but its mismatched schedule produced incoherent audio.)
    return _PIPE


def _free_pipe() -> None:
    """Release the ~5GB Stable Audio model after the bed is written. Music is generated once per
    project (cached to music.wav), so it's never needed twice in a build — and dropping it before
    assemble's ffmpeg-heavy mux/overlay lowers peak memory, cutting SIGSEGV risk on the 24GB budget."""
    global _PIPE
    if _PIPE is None:
        return
    _PIPE = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:  # noqa: BLE001
        log.warning("Could not free Stable Audio pipeline (%s).", e)


def generate_track(out_path: Path, prompt: str | None = None, mood: str = "ethereal",
                   seconds: float = 40.0, steps: int = 160, seed: int = 0,
                   device: str = "mps", dtype: str = "float32") -> Path:
    """Generate one original instrumental bed → out_path (wav). Looped under the voice later.
    dtype 'float16' halves the model footprint (compared via `avp ab`)."""
    import soundfile as sf
    import torch

    seconds = min(float(seconds), 47.0)   # Stable Audio Open hard limit ≈47.55s
    pipe = _pipe(device, dtype)
    text = prompt or PROMPTS.get(mood, PROMPTS["ethereal"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(text, negative_prompt=NEGATIVE, num_inference_steps=steps,
                  audio_end_in_s=float(seconds), num_waveforms_per_prompt=1, generator=gen)
    audio = result.audios[0].T.float().cpu().numpy()          # [samples, channels]
    sr = int(pipe.vae.config.sampling_rate)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr)
    log.info("Generated %.0fs music (%s) → %s (sr=%d)", seconds, mood, out_path, sr)
    _free_pipe()      # drop ~5GB before assemble's ffmpeg mux — music is cached, never needed twice
    return out_path
