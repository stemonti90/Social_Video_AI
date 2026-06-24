"""Per-video project state: a folder under projects/<slug>/ with a manifest.json.

The manifest tracks stage status (so runs are resumable), the models used, the
license attributions owed, and whether AI-disclosure is required at publish time.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import Config
from .log import get_logger
from .models import Attribution

log = get_logger("avp.manifest")

STAGES = ["script", "voice", "footage", "captions", "assemble", "metadata"]


class Manifest:
    def __init__(self, path: Path, data: dict | None = None):
        self.path = path
        self.data = data or {
            "slug": path.parent.name,
            "title": "",
            "topic": "",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "disclosure_ai": False,
            "models": {},
            "stages": {s: {"state": "pending"} for s in STAGES},
            "attributions": [],
        }

    @classmethod
    def load_or_create(cls, path: Path) -> "Manifest":
        if path.exists():
            try:
                return cls(path, json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError) as e:
                # A crash mid-write (SIGSEGV/OOM/Ctrl-C — the exact conditions this pipeline is built
                # to survive) can truncate manifest.json. Don't make the project unrecoverable: fall
                # back to a fresh manifest (artifacts on disk stay intact; stages just re-verify).
                log.warning("manifest %s unreadable (%s) — starting a fresh one.", path, e)
        m = cls(path)
        m.save()
        return m

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: a crash mid-write must never truncate the live manifest. Write a sibling tmp
        # then rename — os.replace is atomic on the same filesystem.
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    def mark(self, stage: str, state: str, **info) -> None:
        entry = {"state": state, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        entry.update(info)
        self.data["stages"][stage] = entry
        self.save()

    def state(self, stage: str) -> str:
        return self.data["stages"].get(stage, {}).get("state", "pending")

    def is_done(self, stage: str) -> bool:
        return self.state(stage) == "done"

    def add_attribution(self, att: Attribution) -> None:
        self.data["attributions"].append(att.__dict__)
        if att.requires_onscreen:
            self.data.setdefault("onscreen_credits", []).append(att.credit)
        self.save()


class VideoProject:
    """Filesystem layout + manifest for one video."""

    def __init__(self, slug: str, cfg: Config):
        self.slug = slug
        self.cfg = cfg
        self.root = Path(cfg.paths.projects_dir) / slug
        self.manifest = Manifest.load_or_create(self.root / "manifest.json")

    @classmethod
    def create(cls, slug: str, cfg: Config) -> "VideoProject":
        p = cls(slug, cfg)
        p.root.mkdir(parents=True, exist_ok=True)
        return p

    # --- artifact paths ---
    @property
    def script_md(self) -> Path:
        return self.root / "script.md"

    @property
    def script_json(self) -> Path:
        return self.root / "script.json"

    @property
    def footage_dir(self) -> Path:
        d = self.root / "footage"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def audio_dir(self) -> Path:
        d = self.root / "audio"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def captions_ass(self) -> Path:
        return self.root / "captions.ass"

    @property
    def output(self) -> Path:
        return self.root / f"{self.slug}.mp4"

    def output_for(self, engine: str) -> Path:
        return self.root / f"{self.slug}.{engine}.mp4"

    @property
    def log_file(self) -> Path:
        return self.root / "build.log"
