"""Domain models — the data that flows through the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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

    @property
    def narration(self) -> str:
        return " ".join(s.narration.strip() for s in self.segments if s.narration.strip())

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
