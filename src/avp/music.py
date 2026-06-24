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

# Mood → prompt. These are background BEDS under narration, so: instrumental, slow, spacious,
# and explicitly NO drums / NO vocals so they never fight the voice.
PROMPTS = {
    "ethereal": ("Ethereal ambient space music, warm analog synth pads, slow evolving drones, "
                 "deep soft sub bass, gentle twinkling celestial bells, vast peaceful cosmic "
                 "atmosphere, cinematic, smooth, no drums, no percussion, no vocals"),
    "cinematic": ("Cinematic orchestral space ambient, soaring string and choir pads, deep "
                  "cinematic drones, slow majestic build, awe and wonder, film score, lush, "
                  "no percussion, no drums, no vocals"),
    "dark": ("Dark mysterious deep-space drone, ominous low synth bass, slow shifting nebula "
             "textures, distant echoes, suspenseful and vast, ambient, no drums, no vocals"),
}
NEGATIVE = "low quality, noisy, distorted, clipping, harsh, lo-fi, abrupt"

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


def _pipe(device: str = "mps"):
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import StableAudioPipeline
        _patch_torchsde()    # make Stable Audio's native cosine-SDE scheduler usable on Mac
        log.info("Loading Stable Audio Open (first run downloads ~5 GB)…")
        _PIPE = StableAudioPipeline.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)
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
                   device: str = "mps") -> Path:
    """Generate one original instrumental bed → out_path (wav). Looped under the voice later."""
    import soundfile as sf
    import torch

    seconds = min(float(seconds), 47.0)   # Stable Audio Open hard limit ≈47.55s
    pipe = _pipe(device)
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
