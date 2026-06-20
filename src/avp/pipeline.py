"""Orchestrator. The 'build' phase runs everything *after* the human-reviewed script,
skipping stages already marked done (unless force=True)."""
import os
import subprocess
import sys

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


def _run_stage_subprocess(name: str, slug: str, config_path: str, verbose: bool) -> int:
    """Run one stage as a SEPARATE `avp` process. Critical on memory-constrained machines: the
    voice stage loads ~GB of TTS models (kokoro/torch/spaCy) that Python won't return to the OS
    in-process — so if assemble ran in the same process it would starve ffmpeg, which SIGSEGVs
    under memory pressure. A fresh process per stage reclaims all of it between stages."""
    src_dir = str(Path(__file__).resolve().parent.parent)          # …/src (so `import avp` works)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src_dir, env.get("PYTHONPATH", "")) if p)
    cmd = [sys.executable, "-m", "avp.cli", name, slug, "--config", config_path]
    if verbose:
        cmd.append("-v")
    return subprocess.run(cmd, env=env).returncode


def build(project: VideoProject, cfg: Config, force: bool = False,
          config_path: str = "config.yaml", verbose: bool = False) -> Path:
    for name in BUILD_STAGES:
        if project.manifest.is_done(name) and not force:
            log.info("• skip %s (already done)", name)
            continue
        log.info("▶ %s", name)
        rc = _run_stage_subprocess(name, project.slug, config_path, verbose)
        if rc != 0:
            project.manifest.mark(name, "failed")
            raise RuntimeError(f"stage {name!r} failed (exit {rc}) — see the project log.")
    return project.output
