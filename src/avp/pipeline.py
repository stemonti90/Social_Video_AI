"""Orchestrator. The 'build' phase runs everything *after* the human-reviewed script,
skipping stages already marked done (unless force=True)."""
from __future__ import annotations

from pathlib import Path

from . import stages
from .config import Config
from .log import get_logger
from .manifest import VideoProject

log = get_logger("avp.pipeline")

BUILD_STAGES = ["voice", "footage", "captions", "assemble", "metadata"]

_DISPATCH = {
    "voice": stages.stage_voice,
    "footage": stages.stage_footage,
    "captions": stages.stage_captions,
    "assemble": stages.stage_assemble,
    "metadata": stages.stage_metadata,
}


def build(project: VideoProject, cfg: Config, force: bool = False) -> Path:
    for name in BUILD_STAGES:
        if project.manifest.is_done(name) and not force:
            log.info("• skip %s (already done)", name)
            continue
        log.info("▶ %s", name)
        try:
            _DISPATCH[name](project, cfg)
        except Exception:
            project.manifest.mark(name, "failed")
            raise
    return project.output
