"""The three daily lanes — Discovery, Value/Education, Product/Community — and what each changes.

Three videos a day must not be three variations of the same video. Each daily slot belongs to a lane
with its own job: DISCOVERY reaches people who do not know the app (astronomy wonder, the channel so
far); EDUCATION earns saves, shares and follows by teaching one concrete mobile-astrophotography
technique; PRODUCT turns interest into a relationship with AstroStackerPro (a feature, an update, a
result, a story from building it). A lane is chosen from the posting slot the run is serving, stored
in the project's manifest, and read by the stages that behave differently: the topic queue and its
refill theme, the writer's brief, the fact sheet's focus, the polish rules, the hashtag bank, the
rotating CTA, and the footage source (PRODUCT needs the app's real pictures — see `assets`).

PRODUCT is honest about its needs: it runs only when `auto.product_assets` holds real screenshots or
screen recordings of the app and its own topic queue has entries. Until then the slot falls back to
EDUCATION — a weaker third video beats a video that fakes an app that exists.
"""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_LANES = ["discovery", "education", "product"]
MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".mp4", ".mov")

SPECS: dict[str, dict] = {
    "discovery": {
        "label": "Discovery",
        "queue": None,                      # the channel's original queue (auto.queue_path)
        "theme": None,                      # auto.theme
        "writer": "",
        "brief_focus": "",
        "polish": "",
        "hashtags": {},
        "cta": {
            "instagram": ["Follow for the sky, explained one wonder at a time.",
                          "Which world should we visit next? Tell us below.",
                          "Send this to someone who still looks up.",
                          "Save this for your next clear night."],
            "tiktok": ["Which world next?", "Send this to a night-sky friend.",
                       "Follow for one cosmic fact a day.", "Save it for a clear night."],
        },
    },
    "education": {
        "label": "Value / Education",
        "queue": "topics.education.txt",
        "theme": ("mobile astrophotography, taught honestly: stacking and why it beats one long exposure, "
                  "exposure and ISO on a phone, focusing on stars, noise and how to kill it, planning a "
                  "night (Moon phase, light pollution), tripods and stability, shooting the Moon, planets, "
                  "the Milky Way and deep-sky objects with a phone, calibration frames, common mistakes, "
                  "single shot vs stack comparisons, what a phone can and cannot capture"),
        "writer": (
            "\n\nLANE: VALUE / EDUCATION. This video TEACHES one concrete thing about photographing the sky "
            "with a phone: a technique, a setting, a mechanism, a mistake to avoid. Open with the surprising "
            "claim (what the viewer did not know was possible or was doing wrong), then the how or the why in "
            "steps a beginner can follow tonight, then the result they will see. Every segment is actionable or "
            "explanatory; numbers (seconds, frames, ISO) come from the fact sheet. Name tools honestly — a phone, "
            "a tripod, a stacking app — and name AstroStackerPro only where it genuinely does the job described."
        ),
        "brief_focus": (
            "Focus: this is a MOBILE ASTROPHOTOGRAPHY TECHNIQUE topic. Give the facts of how it works, the typical "
            "numbers (exposure times, ISO ranges, how many frames to stack, focal lengths, what phones can do), "
            "the common mistakes and their fix, and what result a beginner can realistically expect."
        ),
        "polish": (
            "\n- THIS IS A TEACHING VIDEO: one step or one reason per segment, in the order a beginner would do "
            "it; the surprise is in the claim, the value is in the steps; the last content line states the result "
            "the viewer gets. Never vague: a number from the sheet beats an adjective."
        ),
        "hashtags": {
            "instagram": {
                "broad": ["#astrophotography", "#photography", "#space", "#astronomy", "#nightphotography"],
                "mid": ["#mobilephotography", "#smartphonephotography", "#shotoniphone", "#photographytips",
                        "#nightsky", "#milkyway", "#moonphotography"],
                "community": ["#astrophotographytips", "#stargazing", "#astronomylovers", "#amateurastronomy",
                              "#backyardastronomy", "#iphonephotography"],
            },
            "tiktok": {"core": ["#LearnOnTikTok", "#PhotographyTips", "#astrophotography", "#mobilephotography",
                                "#nightphotography", "#space"]},
        },
        "cta": {
            "instagram": ["Save this for your next clear night.", "Would you try this with your phone?",
                          "Which subject should we shoot next?", "Follow for one mobile astrophotography tip a day."],
            "tiktok": ["Save this for a clear night.", "Would you try it with your phone?",
                       "What should we shoot next?", "Follow for daily phone astrophotography tips."],
        },
    },
    "product": {
        "label": "Product / Community",
        "queue": "topics.product.txt",
        "theme": ("AstroStackerPro, the mobile astrophotography app: one feature at a time (stacking, RAW capture, "
                  "plate solving, calibration frames, drizzle, planner), an update, a result, a decision from "
                  "building it, a question to the community"),
        "writer": (
            "\n\nLANE: PRODUCT / COMMUNITY. This video shows ONE thing about AstroStackerPro — a feature, an "
            "update, a user result, a story from building it — as a story: the problem the viewer has, what the "
            "app does about it, what they see on screen. Concrete and honest; no marketing adjectives; claim "
            "only what the topic line states about the app; invite the viewer to try it or to answer a question."
        ),
        "brief_focus": (
            "Focus: this is about a FEATURE of a mobile astrophotography app. Give the facts of what such a "
            "feature does in general (the technique behind it), why it matters for a phone, and the numbers a "
            "user can expect. Do not invent specifics about the app itself beyond the topic line."
        ),
        "polish": (
            "\n- THIS IS A PRODUCT STORY: problem → what the app does → what the viewer sees. Plain and honest; "
            "the app is named where it acts; the last content line invites the viewer to try or to answer."
        ),
        "hashtags": {
            "instagram": {
                "broad": ["#astrophotography", "#photography", "#app", "#space", "#astronomy"],
                "mid": ["#mobilephotography", "#smartphonephotography", "#photographyapp", "#iphoneapps",
                        "#androidapps", "#nightphotography", "#buildinpublic"],
                "community": ["#astrophotographytips", "#astronomylovers", "#stargazing", "#indiedev",
                              "#iphonephotography"],
            },
            "tiktok": {"core": ["#LearnOnTikTok", "#PhotographyTips", "#astrophotography", "#app",
                                "#mobilephotography", "#buildinpublic"]},
        },
        "cta": {
            "instagram": ["Would you use this feature?", "What should we build next?",
                          "Try it on your next clear night — link in bio.", "Show us your result."],
            "tiktok": ["Would you use this?", "What should we build next?", "Link in bio — try it tonight.",
                       "Show us your stack."],
        },
    },
}


def spec(lane: str) -> dict:
    return SPECS.get((lane or "discovery").lower(), SPECS["discovery"])


def of(project) -> str:
    """The lane a project was created in (manifest), Discovery for everything older."""
    try:
        return str(project.manifest.data.get("lane") or "discovery").lower()
    except Exception:  # noqa: BLE001
        return "discovery"


def for_slot(now: datetime, post_times: list[str], lanes: list[str] | None = None) -> str:
    """The lane of the posting slot this run serves: the first slot later than `now` today, else the
    first slot (a run after the last slot builds tomorrow's first video). Lanes map to slots in order
    and repeat if there are more slots than lanes."""
    lanes = [x for x in (lanes or DEFAULT_LANES) if x] or DEFAULT_LANES
    parsed = []
    for t in post_times or []:
        try:
            hh, mm = (int(x) for x in str(t).split(":")[:2])
            parsed.append(hh * 60 + mm)
        except Exception:  # noqa: BLE001
            continue
    parsed.sort()
    minute = now.hour * 60 + now.minute
    idx = next((i for i, m in enumerate(parsed) if m > minute), 0)
    return lanes[idx % len(lanes)]


def assets(cfg) -> list[Path]:
    """The app's real pictures and screen recordings for the Product lane, if any."""
    d = Path(str(getattr(cfg.auto, "product_assets", "assets/app") or "assets/app")).expanduser()
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in MEDIA_SUFFIXES and not p.name.startswith("."))


def effective(lane: str, cfg, queue_has_topics: bool = True) -> str:
    """PRODUCT only when it can be honest: real app assets AND its own topics. Otherwise EDUCATION."""
    if (lane or "").lower() == "product" and (not assets(cfg) or not queue_has_topics):
        log.warning("Product lane not ready (%s) — this slot runs the Education lane instead.",
                    "no app assets in auto.product_assets" if not assets(cfg) else "no product topics queued")
        return "education"
    return (lane or "discovery").lower()


def cta(lane: str, platform: str, slug: str) -> str:
    """One CTA from the lane's bank per platform, rotated by the video's slug — never the same line
    on every post, always the same line for the same video."""
    bank = spec(lane).get("cta", {}).get(platform) or spec("discovery")["cta"].get(platform) or []
    if not bank:
        return ""
    i = int(hashlib.md5(f"{slug}:{platform}".encode()).hexdigest(), 16) % len(bank)
    return bank[i]


def apply_cta(meta: dict, lane: str, slug: str) -> dict:
    """Put the rotating CTA into the captions: Instagram before the blank line and the tags, TikTok
    right after the hook. A caption that already asks something is left alone."""
    for plat in ("instagram", "tiktok"):
        d = meta.get(plat)
        if not isinstance(d, dict) or not d.get("caption"):
            continue
        line = cta(lane, plat, slug)
        if not line:
            continue
        cap = d["caption"]
        body, sep, tags = cap.partition("\n\n") if plat == "instagram" else (cap, "", "")
        head = body if plat == "instagram" else re.split(r"\s#", body, 1)[0]
        if "?" in head or line.lower() in cap.lower():
            continue
        if plat == "instagram":
            d["caption"] = f"{body.rstrip()}\n{line}{sep}{tags}" if sep else f"{body.rstrip()}\n{line}"
        else:
            rest = body[len(head):]
            d["caption"] = f"{head.rstrip()} {line}{rest}"
    return meta


def place_product_assets(project, cfg, n_segments: int) -> int:
    """Copy the app's assets into the project as manual footage (NN.ext), round-robin by slug, so the
    footage stage uses real screens instead of generating. Returns how many were placed."""
    files = assets(cfg)
    if not files or n_segments <= 0:
        return 0
    start = int(hashlib.md5(project.root.name.encode()).hexdigest(), 16) % len(files)
    project.footage_dir.mkdir(parents=True, exist_ok=True)
    placed = 0
    for i in range(n_segments):
        src = files[(start + i) % len(files)]
        dest = project.footage_dir / f"{i + 1:02d}{src.suffix.lower()}"
        if not dest.exists():
            shutil.copyfile(src, dest)
        placed += 1
    log.info("Product lane: %d app asset(s) placed as footage from %s", placed, files[0].parent)
    return placed
