"""Domain models — the data that flows through the pipeline."""
from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field
from typing import Any


def dedupe_segments(segments: list["Segment"]) -> list["Segment"]:
    """Drop segments whose narration repeats an earlier one (verbatim or near-verbatim, ratio
    >= 0.9) and re-index. The local model sometimes pads to the requested segment count by
    repeating its 'payoff' line (e.g. identical segments 8 & 9). Keeps the first occurrence."""
    kept: list[Segment] = []
    keys: list[str] = []
    for seg in segments:
        norm = re.sub(r"[^a-z0-9]+", " ", (seg.narration or "").lower()).strip()
        if norm and any(difflib.SequenceMatcher(None, norm, k).ratio() >= 0.9 for k in keys):
            continue
        keys.append(norm)
        kept.append(seg)
    for i, seg in enumerate(kept):
        seg.index = i + 1
    return kept


@dataclass
class Segment:
    """One beat of the video: a spoken line + the visual that should accompany it."""
    index: int
    narration: str
    visual: str = ""                      # human/AI cue describing the ideal footage
    keywords: list[str] = field(default_factory=list)  # search terms for archives
    footage: str | None = None            # resolved file name, relative to project footage/
    duration: float | None = None         # seconds, measured after TTS
    kind: str = "content"                 # "content" | "cta" (app endcard)
    credit: str = ""                      # source credit to display (e.g. "NASA/JPL-Caltech")


@dataclass
class Script:
    title: str
    segments: list[Segment]
    target_seconds: int = 60
    disclosure_ai: bool = False           # True only if AI-generated *realistic* visuals are used
    topic: str = ""
    notes: str = ""
    # One spoken sentence the LLM writes to BRIDGE from this topic into the app CTA (e.g. "Want to
    # capture Saturn's rings with your own phone?"). Empty on old scripts → generic funnel line.
    cta_bridge: str = ""

    @property
    def narration(self) -> str:
        # the CTA endcard is shown, not spoken → keep it out of the spoken/captioned text
        return " ".join(s.narration.strip() for s in self.segments
                        if s.narration.strip() and s.kind != "cta")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Script":
        segs = [Segment(**s) for s in d.get("segments", [])]
        rest = {k: v for k, v in d.items() if k != "segments"}
        return cls(segments=segs, **rest)


@dataclass
class Attribution:
    """A required credit, captured so the build can honor licenses (CC BY etc.)."""
    source: str                  # "NASA", "ESA/Hubble", ...
    credit: str                  # exact credit line to show
    license: str                 # "Public Domain", "CC BY 4.0", ...
    url: str = ""
    asset_id: str = ""
    requires_onscreen: bool = False   # CC BY -> must be burned on screen, not just described
