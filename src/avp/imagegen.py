"""Local AI visuals: generate a segment's image with mflux (z-image-turbo on Apple MLX) and pick the
best of N candidates with CLIP.

Same engine Stefano's other sites use for article images, adapted to space/astronomy verticals. All
local, nothing leaves the Mac.

CRITICAL — always point --model at a PRE-QUANTIZED model on disk (`mflux-save -q 4 --path …`).
Quantizing on the fly loads full precision first: 27.8 GB peak / 11m49s per image on a 24 GB M5,
versus 9.8 GB / 74s from the saved Q4 model (measured 2026-08-27).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .llm import SECONDS_PER_IMAGE
from .log import get_logger
from .models import Script, Segment

log = get_logger("avp.imagegen")

# Visual registry per segment intent — the space equivalent of the article pipeline's REGISTRO.
# Keeps different subjects looking like one channel instead of a stock-image grab bag.
REGISTRO = {
    # LIGHTING AND PALETTE ONLY — deliberately no composition. An earlier version prescribed the shot
    # too ("the Sun ... photosphere with dark sunspots" = a full disc), which silently overrode what
    # the script asked for: a sunspot video whose five segments requested a close-up, a scale
    # comparison, erupting flares and a SOHO view came back as five identical full discs.
    # Say how it is LIT, never how it is FRAMED — the framing comes from the script and `SHOTS`.
    #
    # Also: not one word of negation here. These strings go into the POSITIVE prompt, where "no
    # orange cast" reads to the model as "orange cast" — the exact thing it was meant to prevent.
    # Everything to avoid lives in NEGATIVI, which is passed via --negative-prompt.
    "planet": ("in its own true colours under hard white sunlight, deep black shadow on the night side, "
               "pure black space behind it"),
    "deep_sky": ("cool blue and violet hydrogen glow with white starlight on a pure black sky, "
                 "long-exposure astrophotography"),
    # Two kinds of ground, and the sky is what separates them. "surface" describes an AIRLESS world
    # (the Moon, an asteroid): hard shadows, black sky right down to the horizon. Mars has an
    # atmosphere and looked wrong under that rule — an Opportunity build came back with the rover
    # sitting under open black space in four frames out of eight (2026-08-30). A dusty world scatters
    # its own light: butterscotch sky, soft shadows, haze that swallows the far horizon.
    "surface": ("grey regolith in its natural colour, raking white low light, hard black shadows, "
                "airless black sky meeting the horizon sharply"),
    "dusty_surface": ("rusty ochre soil under a hazy butterscotch sky, soft diffuse daylight, "
                      "fine dust softening the distant horizon"),
    "spacecraft": ("cold white sunlight glinting on grey metal and gold foil, the black void of deep "
                   "space around it, distant pinpoint stars"),
    "star": ("natural visible light, pale yellow-white surface with realistic granulation, "
             "black sky around it"),
    "default": ("white and blue starlight on a black sky, neutral documentary colour"),
}

# Shot scale, prepended so it lands in the first tokens — diffusion models weight the opening of the
# prompt far more heavily, which is why the old code's trailing "Alternate framing: …" was ignored
# outright (measured: the close-up variant produced another identical full disc).
SHOTS = (
    "",                                                     # whatever the script asked for
    "Extreme close-up, filling the frame with one small detail. ",
    "Wide establishing shot, the subject small and distant. ",
)

# Style constants: what keeps segment-to-segment images coherent. Photographic, never illustration —
# for space, an "artist's impression" look reads as fake and undercuts the channel's credibility.
# "real telescope imagery look" invited SDO-style FALSE-COLOR renders (deep orange suns, neon nebulae):
# striking for astrophotographers, but average viewers read them as a cheap render. Natural color wins.
STILE = (
    "photorealistic astrophotography, natural true-color palette exactly as the human eye would see "
    "it, realistic dynamic range, fine detail and subtle film grain"
)
# Passed to --negative-prompt, NOT appended to the positive one. mflux/z-image-turbo runs at guidance
# 3.5, so classifier-free guidance is active and these actually steer the model away. Appending them
# as "Avoid: …" text (what this used to do) does the opposite: "monochrome orange" and "false color"
# sitting in the positive prompt are a large part of why every video came out orange.
NEGATIVI = ("illustration, 3d render, cgi, digital art, cartoon, anime, painting, oversaturated neon, "
            "false color, infrared look, x-ray palette, monochrome orange, fantasy, people, faces, "
            "text, watermark, captions, logos, ui elements, lens flare, "
            # "people, faces" reads as portrait scale. The model kept adding the tiny silhouette
            # that landscape photography conventionally puts in for scale — three of the eight
            # Olympus Mons frames had a figure standing on a ridge, on a planet nobody has visited.
            "person, human figure, silhouette of a person, hiker, climber, tourist, astronaut, "
            "figure for scale, footprints, "
            # A writer who asks for a "diagram" gets garbled pseudo-text from this model, and under
            # footage_source: generate_only there is no archive left to rescue the frame.
            "diagram, chart, infographic, schematic, cutaway, labels, arrows, annotations, "
            "split screen, side by side comparison, collage, multi-panel")

# Keyword → registry bucket. First match wins; order matters (surface before planet).
_BUCKETS = (
    # ORBITAL FIRST. A cue can name a world and still be a shot of it from space — "the planet Mars
    # in the dark void" was matching `mars` and coming back with "rusty ochre soil under a hazy
    # butterscotch sky", so the registry described a desert while the shot asked for a globe against
    # black. The model obeyed the registry, which is the more concrete of the two. These phrases say
    # "we are outside the atmosphere looking in", whatever world follows.
    ("planet", ("in the dark void", "from space", "from orbit", "against the blackness",
                "against black space", "hanging in space", "seen from space", "full disc",
                "full disk", "the whole planet", "the globe of")),
    # Dusty worlds next: "Martian surface" must not fall into the airless-Moon bucket.
    ("dusty_surface", ("mars", "martian", "titan", "venus", "venusian", "dust storm", "sand dune")),
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


_DRAWN = re.compile(
    r"\b(diagram|chart|graph|infographic|schematic|cutaway|cross[- ]section|illustration|"
    r"annotated|labell?ed|side[- ]by[- ]side|split[- ]screen|comparison)\b", re.IGNORECASE)
_VS = re.compile(r"\s+(?:vs\.?|versus)\s+", re.IGNORECASE)
_SCALE = re.compile(r"^([^,]*\b(?:shot|frame|angle|close[- ]up)\b[^,]*),\s*(.+)$", re.IGNORECASE)
_DANGLING = re.compile(r"^(?:of|and|with|between|showing)\s+", re.IGNORECASE)


def _photographable(visual: str) -> str:
    """The part of a visual cue a camera could actually shoot. Empty if none of it is.

    The writer is told to describe a photographable scene, but a prompt is advice: it still
    occasionally asks for "medium shot, tectonic plate movement diagram vs static Martian crust". A
    negative prompt alone does not save that — the model is still handed "diagram" as the subject it
    must depict, and under `footage_source: generate_only` there is no archive left to rescue the
    frame. So it comes out of the positive prompt too:

    * "X vs Y" is a split-screen request in disguise; keep whichever side is not a drawn figure
      (here "static Martian crust" — the real photograph), preserving the shot scale that prefixes
      the whole cue,
    * the drawn-figure words themselves are removed from whatever remains,
    * a preposition left dangling by the removal goes with it ("annotated cutaway OF the magma
      chamber" must not become "of the magma chamber").
    """
    raw = (visual or "").strip()
    if not raw:
        return ""
    prefix, body = "", raw
    m = _SCALE.match(raw)                 # "wide shot, …" belongs to the cue, not to either side
    if m:
        prefix, body = m.group(1) + ", ", m.group(2)
    sides = _VS.split(body)
    if len(sides) > 1:
        shootable = [x for x in sides if not _DRAWN.search(x)]
        body = (shootable or sides)[0]
    body = _DANGLING.sub("", _DRAWN.sub("", body).strip(" ,;:-"))
    body = re.sub(r"\s{2,}", " ", body).strip(" ,;:-").strip()
    return f"{prefix}{body}" if body else ""


def images_for_segment(seg: Segment, cfg) -> int:
    """How many stills this segment is cut across, from its MEASURED narration length.

    This used to be a flat 2 for every segment, which made the cut rhythm a hostage of how the
    writer happened to split the text. Measured across twelve builds, the writer delivers anywhere
    from 4 to 8 segments for the same 50s target, so "2 images each" landed between 2.4s and 5.7s
    per image — the 2.4s end is the "everything flies past, I cannot even read it" the channel owner
    reported. The voice stage has already measured every segment by the time footage runs, so the
    editor can simply ask for the number of images that puts each one near SECONDS_PER_IMAGE, and
    the writer's segmentation stops mattering.

    Rounding to nearest is deliberate: a 5.5s segment holds ONE still for 5.5s rather than cutting
    to two at 2.75s, because 5.5s is the closer miss and a slow shot reads far better than a flash.
    """
    per = SECONDS_PER_IMAGE
    cap = max(1, int(getattr(cfg.video, "max_images_per_segment", 3) or 3))
    dur = float(getattr(seg, "duration", 0) or 0)
    if dur <= 0:                      # voice stage skipped — fall back to the planning assumption
        return max(1, min(cap, int(getattr(cfg.video, "images_per_segment", 2) or 2)))
    return max(1, min(cap, round(dur / per)))


def build_prompt(seg: Segment, script: Script, shot: str = "") -> str:
    """A generation prompt for one segment: its own visual intent, in the channel's house style.

    Uses the script's VISUAL/KEYWORDS (what the writer meant to show), not the narration verbatim —
    narration is spoken language and makes muddy prompts. `shot` is prepended, never appended: the
    opening tokens are what the model actually obeys.

    Returns the POSITIVE prompt only. What to avoid goes to --negative-prompt (see NEGATIVI)."""
    kw = " ".join(str(k) for k in (seg.keywords or []) if k)
    # Falls through to the keywords when the visual was nothing but a drawn-figure request.
    subject = _photographable(seg.visual or "") or kw or (script.topic or "").strip() or "deep space"
    context = REGISTRO[_bucket(f"{subject} {kw}")]
    return f"{shot}{subject}, {context}. {STILE}."


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
           "--prompt", prompt, "--negative-prompt", NEGATIVI, "--output", str(out)]
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


# NOTE — false-color suns cannot be fixed in post (tried 2026-08-28, reverted). z-image-turbo always
# renders solar subjects in saturated SDO orange and no prompt wording overrides it. Three corrections
# were built and measured on real frames: a linear per-channel gain (blew the disc to flat white), a
# gamma curve (no effect — B is crushed to ~0 on the disc and 0**x is still 0), and a blend toward a
# target built from R (works, but yields a washed sepia/grey sun — the blue detail simply is not in the
# file). Post-processing cannot invent the missing channel; the vivid orange is the better shipping
# choice. Fixing this for real needs a different image model, not a filter.

# --------------------------------------------------------------------------- what NOT to generate
# Two subject families where this model is measurably WRONG, not merely weaker. For these the caller
# asks the archives first and only falls back to generation.
#
#   "deep_sky"  — z-image-turbo has exactly ONE canonical deep-sky image: a face-on spiral galaxy.
#                 Measured 2026-08-30 on the Orion Nebula with five prompts at a fixed seed: the full
#                 pipeline prompt; the same without the shot-scale prefix; that plus a negative prompt
#                 naming "spiral galaxy, galactic disc, spiral arms, whirlpool"; a bare "The Orion
#                 Nebula, M42, photograph"; and an explicit "emission nebula, chaotic cloud of gas and
#                 dust, NO spiral structure, four bright young stars at its heart". All five returned
#                 the same spiral galaxy. The prior does not bend — and a nebula rendered as a galaxy
#                 is not a stylistic miss, it is the wrong object in an astronomy video.
#   "star"      — always the saturated false-colour SDO orange disc (documented above, 2026-08-28).
#
# Everything else still generates: probes in flight, planetary surfaces and conceptual shots are
# either fine or have no archive equivalent. Revisit this set when the image model changes — it is a
# statement about z-image-turbo, not about generation in general.
ARCHIVE_FIRST = frozenset({"deep_sky", "star"})


def prefers_archive(seg: Segment, script: Script) -> bool:
    """True when real archive footage should be tried BEFORE generating this segment."""
    kw = " ".join(str(k) for k in (seg.keywords or []) if k)
    subject = (seg.visual or kw or script.topic or "")
    return _bucket(f"{subject} {kw}") in ARCHIVE_FIRST


# --------------------------------------------------------------------------- people in the frame
_DET: dict = {}
PEOPLE_THRESHOLD = 0.9    # measured: every real figure scored >= 0.991, every clean frame <= 0.637


def _person_detector():
    """torchvision's COCO Faster R-CNN — already installed, BSD-licensed, no new dependency. Loaded
    lazily and cached on success (a failure is not cached: see _clip for why). min_size is large
    because the figures we chase are ~15px tall in a 1280px frame; at that size CLIP could not see
    them at all (measured at four tile grids), this model separates them perfectly."""
    if "model" in _DET:
        return _DET["model"], _DET["person"]
    try:
        from torchvision.models import detection as D
        w = D.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = D.fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.05,
                                             min_size=1280, max_size=1600).eval()
        _DET["model"], _DET["person"] = model, w.meta["categories"].index("person")
        return model, _DET["person"]
    except Exception as e:  # noqa: BLE001 — no detector means no filtering, never a failed build
        log.warning("Person detector unavailable (%s) — candidates will not be screened for people.", e)
        return None, None


def person_scores(paths: list) -> list[float] | None:
    """Highest 'person' confidence in each image (0..1), or None when the detector is unavailable.

    Why this exists: this image model puts a lone figure on a ridge "for scale" in roughly a quarter
    of its landscape frames — on a faceless channel, about worlds nobody has stood on. Nothing in the
    prompt removes it (negative and positive phrasing both measured at zero effect, same seeds, same
    figures). But a clean candidate existed in every pair generated, so the fix is not to ask
    better; it is to LOOK, and pick the candidate without the person."""
    model, person = _person_detector()
    if model is None:
        return None
    try:
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        out: list[float] = []
        with torch.no_grad():
            for p in paths:
                res = model([to_tensor(Image.open(p).convert("RGB"))])[0]
                sc = [float(s) for s, l in zip(res["scores"], res["labels"]) if int(l) == person]
                out.append(max(sc) if sc else 0.0)
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("Person screening failed (%s) — candidates not screened.", e)
        return None


# --------------------------------------------------------------------------- CLIP selection
_CLIP: dict = {}
_CLIP_TRIES = 3          # give up ranking only after this many genuine load failures


def _clip():
    """Lazily load CLIP (transformers, already a dependency). Returns None if unavailable so selection
    degrades to 'first candidate' rather than failing the build.

    A SUCCESS is cached forever; a FAILURE is not. The generator holds ~9.8 GB while it runs, and on a
    24 GB machine CLIP's own ~600 MB can fail to load in that window — observed mid-build, free memory
    down to tens of megabytes. Caching that failure switched ranking off for the whole video even
    though memory frees up between generations, so we retry (at most `_CLIP_TRIES` times, to avoid
    paying the load cost on every segment when CLIP is genuinely absent)."""
    if _CLIP.get("state") is not None:
        return _CLIP["state"]
    if _CLIP.get("fails", 0) >= _CLIP_TRIES:
        return None
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        # Cache FIRST, network only if the cache is empty. With the weights already on disk,
        # transformers still phoned HuggingFace to look for a safetensors variant, and one night the
        # connection simply hung: 0% CPU, ten minutes, a build that would have sat there until the
        # stage watchdog shot it. A model that is on disk must never depend on a server to load.
        try:
            model = CLIPModel.from_pretrained(name, local_files_only=True)
            proc = CLIPProcessor.from_pretrained(name, local_files_only=True)
        except Exception:  # noqa: BLE001 — not cached yet: the one time the network is allowed
            log.info("CLIP not in the local cache — downloading once.")
            model = CLIPModel.from_pretrained(name)
            proc = CLIPProcessor.from_pretrained(name)
        dev = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        model = model.to(dev).eval()
        _CLIP["state"] = (model, proc, dev, torch)
    except Exception as e:  # noqa: BLE001 — CLIP is a refinement, not a requirement
        _CLIP["fails"] = _CLIP.get("fails", 0) + 1
        log.warning("CLIP unavailable (%s) — picking the first candidate%s.", e,
                    "" if _CLIP["fails"] >= _CLIP_TRIES else "; will retry next segment")
        return None
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
    """Generate the segment's visuals: N candidates → CLIP ranks them against the segment's meaning →
    the top `images_per_segment` go on screen (best first at `dest`, runners-up as NN_2.png, …), each
    taking a slice of the segment's duration. Returns a report dict, or None if nothing generated."""
    if not available(cfg):
        return None
    keep = images_for_segment(seg, cfg)
    n = max(keep, int(getattr(cfg.video, "image_candidates", 2)))
    base_seed = int(getattr(cfg.video, "image_seed", 100))
    work = work_dir or (dest.parent / "_gen")
    work.mkdir(parents=True, exist_ok=True)
    for stale in dest.parent.glob(f"{seg.index:02d}_*.png"):
        stale.unlink(missing_ok=True)            # old runners-up must not leak into this build

    # Candidate shots: same subject, genuinely different SCALE — so the mid-segment cut changes the
    # picture instead of showing the same disc with different grain (seed alone barely varies it).
    # Offset by segment index so consecutive segments don't open on the same scale either.
    cands: list[Path] = []
    prompt = build_prompt(seg, script)          # reported in the result; candidate 0 uses it as-is
    for i in range(n):
        c = work / f"{seg.index:02d}_c{i}.png"
        shot = SHOTS[(i + seg.index) % len(SHOTS)]
        # Vary seed per candidate AND per segment, so segment 2 never repeats segment 1's image.
        if _run_mflux(build_prompt(seg, script, shot), c,
                      base_seed + i * 977 + seg.index * 31, cfg):
            cands.append(c)
    if not cands:
        return None

    # Screen for people BEFORE ranking. A candidate with a figure in it is not "a bit worse" — on a
    # faceless channel about uninhabited worlds it is wrong — so it is excluded, and if that leaves
    # too few, more candidates are generated with fresh seeds (measured: a clean one existed in
    # every pair). Only if every attempt has a person do we fall back to the least-peopled one,
    # and say so loudly.
    people: list[float] = []
    if getattr(cfg.video, "image_reject_people", True):
        thr = float(getattr(cfg.video, "image_people_threshold", PEOPLE_THRESHOLD))
        retries = int(getattr(cfg.video, "image_people_retries", 2))
        people = person_scores(cands) or []
        extra = 0
        while people and sum(1 for x in people if x < thr) < keep and extra < retries:
            j = n + extra
            c = work / f"{seg.index:02d}_c{j}.png"
            shot = SHOTS[(j + seg.index) % len(SHOTS)]
            log.info("Segment %d: %d of %d candidates show a person — generating another",
                     seg.index, sum(1 for x in people if x >= thr), len(cands))
            if _run_mflux(build_prompt(seg, script, shot), c, base_seed + j * 977 + seg.index * 31, cfg):
                cands.append(c)
                people += person_scores([c]) or [0.0]
            extra += 1
        if people:
            clean = [i for i, x in enumerate(people) if x < thr]
            if len(clean) >= keep:
                if len(clean) < len(cands):
                    log.info("Segment %d: dropped %d candidate(s) with a person in frame.",
                             seg.index, len(cands) - len(clean))
                cands = [cands[i] for i in clean]
                people = [people[i] for i in clean]
            else:
                order_p = sorted(range(len(cands)), key=lambda i: people[i])
                log.warning("Segment %d: every candidate shows a person (best p=%.2f) — keeping the "
                            "least peopled; consider regenerating this segment.",
                            seg.index, people[order_p[0]])
                cands = [cands[i] for i in order_p]
                people = [people[i] for i in order_p]

    # Score against the segment's MEANING (visual intent + keywords), not the styled prompt: every
    # candidate shares the style tail, so it carries no signal for choosing between them.
    meaning = (seg.visual or "") + " " + " ".join(str(k) for k in (seg.keywords or []) if k)
    scores = clip_scores(cands, meaning.strip() or script.topic or "deep space")
    if scores:
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        method = "clip"
    else:
        order, method = list(range(len(cands))), "first"
    kept = order[:min(keep, len(cands))]
    shutil.copyfile(cands[kept[0]], dest)
    for pos, idx in enumerate(kept[1:], start=2):
        shutil.copyfile(cands[idx], dest.parent / f"{seg.index:02d}_{pos}.png")
    log.info("Segment %d ← %d generated image%s of %d candidates (%s rank%s)",
             seg.index, len(kept), "" if len(kept) == 1 else "s", len(cands), method,
             f", best {scores[kept[0]]:.3f}" if scores else "")
    return {"prompt": prompt, "candidates": len(cands), "kept": len(kept),
            "selection": method, "scores": scores, "chosen": kept[0],
            "people": people or None}
