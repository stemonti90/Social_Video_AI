"""Local AI visuals: generate a segment's image with mflux (z-image-turbo on Apple MLX) and pick the
best of N candidates with CLIP.

Same engine Stefano's other sites use for article images, adapted to space/astronomy verticals. All
local, nothing leaves the Mac.

CRITICAL — always point --model at a PRE-QUANTIZED model on disk (`mflux-save -q 4 --path …`).
Quantizing on the fly loads full precision first: 27.8 GB peak / 11m49s per image on a 24 GB M5,
versus 9.8 GB / 74s from the saved Q4 model (measured 2026-08-27).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .log import get_logger
from .models import Script, Segment

log = get_logger("avp.imagegen")

# Visual registry per segment intent — the space equivalent of the article pipeline's REGISTRO.
# Keeps different subjects looking like one channel instead of a stock-image grab bag.
REGISTRO = {
    "planet": "a planet or moon filling much of the frame, seen from orbit, hard sunlight and deep shadow",
    "deep_sky": "a nebula or galaxy field, long-exposure astrophotography, faint gas structure",
    "surface": "a planetary or lunar surface up close, raking low light across terrain",
    "spacecraft": "a spacecraft or probe silhouetted against a planet, sunlit metal and antennas",
    "star": "a star or the Sun with its corona, extreme dynamic range, no lens flare clichés",
    "default": "a deep space scene, stars and cosmic structure, documentary framing",
}

# Style constants: what keeps segment-to-segment images coherent. Photographic, never illustration —
# for space, an "artist's impression" look reads as fake and undercuts the channel's credibility.
STILE = (
    "photorealistic astrophotography, cinematic wide shot, real telescope and probe imagery look, "
    "natural light and true-to-life color, fine detail and film grain, "
    "no text, no watermark, no captions, no logos"
)
NEGATIVI = ("illustration, 3d render, cgi, digital art, cartoon, anime, painting, oversaturated neon, "
            "fantasy, people, faces, text, watermark, ui elements")

# Keyword → registry bucket. First match wins; order matters (surface before planet).
_BUCKETS = (
    ("surface", ("surface", "crater", "terrain", "dune", "canyon", "regolith", "landscape", "ice sheet")),
    ("spacecraft", ("probe", "spacecraft", "satellite", "rover", "telescope", "mission", "lander",
                    "orbiter", "voyager", "cassini", "hubble", "webb")),
    ("deep_sky", ("nebula", "galaxy", "cluster", "supernova", "cosmic", "universe", "deep space",
                  "black hole", "quasar", "dark matter")),
    ("star", ("sun", "solar", "star", "corona", "flare", "betelgeuse", "supergiant")),
    ("planet", ("planet", "moon", "saturn", "jupiter", "mars", "venus", "mercury", "neptune",
                "uranus", "pluto", "titan", "europa", "enceladus", "ring")),
)


def _bucket(text: str) -> str:
    low = (text or "").lower()
    for name, words in _BUCKETS:
        if any(w in low for w in words):
            return name
    return "default"


def build_prompt(seg: Segment, script: Script) -> str:
    """A generation prompt for one segment: its own visual intent, in the channel's house style.
    Uses the script's VISUAL/KEYWORDS (what the writer meant to show), not the narration verbatim —
    narration is spoken language and makes muddy prompts."""
    kw = " ".join(str(k) for k in (seg.keywords or []) if k)
    subject = (seg.visual or kw or script.topic or "deep space").strip()
    context = REGISTRO[_bucket(f"{subject} {kw}")]
    return f"{subject}. Context: {context}. {STILE}. Avoid: {NEGATIVI}."


# --------------------------------------------------------------------------- mflux
def _binary(cfg) -> Path:
    root = Path(getattr(cfg.video, "image_venv", "vendor/mflux-venv")).expanduser()
    return root / "bin" / "mflux-generate-z-image-turbo"


def _model_path(cfg) -> Path:
    return Path(getattr(cfg.video, "image_model", "vendor/zimage-q4")).expanduser()


def available(cfg) -> bool:
    """True only if BOTH the mflux binary and the pre-quantized model are on disk. Never raises —
    callers fall back to the archive sources when this is False."""
    try:
        return _binary(cfg).exists() and _model_path(cfg).is_dir()
    except Exception:  # noqa: BLE001
        return False


def _run_mflux(prompt: str, out: Path, seed: int, cfg) -> bool:
    """One generation. Returns True if the image landed. Never raises."""
    w = int(getattr(cfg.video, "image_width", 720))
    h = int(getattr(cfg.video, "image_height", 1280))
    steps = int(getattr(cfg.video, "image_steps", 8))
    cmd = [str(_binary(cfg)), "--model", str(_model_path(cfg)), "--base-model", "z-image-turbo",
           "--steps", str(steps), "--seed", str(seed), "--height", str(h), "--width", str(w),
           "--prompt", prompt, "--output", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=int(getattr(cfg.video, "image_timeout", 900)))
    except Exception as e:  # noqa: BLE001 — a hung/missing generator must not kill the build
        log.warning("mflux failed to run (%s)", e)
        return False
    if r.returncode != 0 or not out.exists():
        log.warning("mflux returned %s for %s (%s)", r.returncode, out.name,
                    (r.stderr or "").strip()[-160:])
        return False
    return True


# --------------------------------------------------------------------------- CLIP selection
_CLIP: dict = {}


def _clip():
    """Lazily load CLIP (transformers, already a dependency). Cached; returns None if unavailable so
    selection degrades to 'first candidate' rather than failing the build."""
    if "state" in _CLIP:
        return _CLIP["state"]
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(name)
        proc = CLIPProcessor.from_pretrained(name)
        dev = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        model = model.to(dev).eval()
        _CLIP["state"] = (model, proc, dev, torch)
    except Exception as e:  # noqa: BLE001 — CLIP is a refinement, not a requirement
        log.warning("CLIP unavailable (%s) — picking the first candidate.", e)
        _CLIP["state"] = None
    return _CLIP["state"]


def clip_available() -> bool:
    """True if CLIP can actually score (model loadable). Used by the archive path to decide whether
    to rerank on the pixels; never raises."""
    return _clip() is not None


def clip_scores(images: list[Path], text: str) -> list[float]:
    """Cosine similarity of each image to `text` (0-1-ish, comparable within one call). Empty list if
    CLIP isn't available."""
    state = _clip()
    if not state or not images:
        return []
    model, proc, dev, torch = state
    try:
        from PIL import Image
        pil = [Image.open(p).convert("RGB") for p in images]
        # CLIP truncates at 77 tokens; the prompt's style tail is identical across candidates anyway.
        inputs = proc(text=[text[:300]], images=pil, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sims = (img @ txt.T).squeeze(-1)
        return [round(float(s), 4) for s in sims.cpu()]
    except Exception as e:  # noqa: BLE001
        log.warning("CLIP scoring failed (%s) — picking the first candidate.", e)
        return []


# --------------------------------------------------------------------------- public API
def generate_for_segment(seg: Segment, script: Script, cfg, dest: Path,
                         work_dir: Path | None = None) -> dict | None:
    """Generate the segment's visual: N candidates → CLIP picks the one closest to the segment's
    meaning → copied to `dest`. Returns a report dict, or None if nothing could be generated (the
    caller then falls back to the archive chain)."""
    if not available(cfg):
        return None
    n = max(1, int(getattr(cfg.video, "image_candidates", 2)))
    base_seed = int(getattr(cfg.video, "image_seed", 100))
    prompt = build_prompt(seg, script)
    work = work_dir or (dest.parent / "_gen")
    work.mkdir(parents=True, exist_ok=True)

    cands: list[Path] = []
    for i in range(n):
        c = work / f"{seg.index:02d}_c{i}.png"
        # Vary the seed per candidate AND per segment, so segment 2 never repeats segment 1's image.
        if _run_mflux(prompt, c, base_seed + i * 977 + seg.index * 31, cfg):
            cands.append(c)
    if not cands:
        return None

    # Score against the segment's MEANING (visual intent + keywords), not the styled prompt: every
    # candidate shares the style tail, so it carries no signal for choosing between them.
    meaning = (seg.visual or "") + " " + " ".join(str(k) for k in (seg.keywords or []) if k)
    scores = clip_scores(cands, meaning.strip() or script.topic or "deep space")
    if scores:
        best_i = max(range(len(cands)), key=lambda i: scores[i])
        method = "clip"
    else:
        best_i, method = 0, "first"
    shutil.copyfile(cands[best_i], dest)
    log.info("Segment %d ← generated image (%d candidate%s, %s pick%s)",
             seg.index, len(cands), "" if len(cands) == 1 else "s", method,
             f", score {scores[best_i]:.3f}" if scores else "")
    return {"prompt": prompt, "candidates": len(cands), "selection": method,
            "scores": scores, "chosen": best_i}
