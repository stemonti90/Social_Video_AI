"""Orchestrator. The 'build' phase runs everything *after* the human-reviewed script,
skipping stages already marked done (unless force=True)."""
import os
import subprocess
import sys

from pathlib import Path

from .config import Config
from .log import get_logger
from .manifest import VideoProject

log = get_logger("avp.pipeline")

# Stage order is RAM-choreographed for a 24GB Mac. metadata runs BEFORE captions so the Ollama model
# (~7GB, still warm from the script stage) serves BOTH metadata and captions' subtitle translation
# with no cold reload. captions THEN evicts that model before its STT aligner (parakeet/MLX) runs —
# the aligner needs the RAM, and if the model is still resident the aligner's subprocess fails under
# memory pressure and silently falls back to even timing. assemble runs last with the model already
# gone, giving ffmpeg headroom (it SIGSEGVs under memory pressure). metadata only needs script.json.
BUILD_STAGES = ["voice", "footage", "metadata", "captions", "assemble"]


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
            # Mark via a FRESH manifest read, not the parent's stale in-memory copy: the stage
            # subprocess wrote its own manifest (incl. license attributions) this run, and saving
            # the parent's pre-loop snapshot here would clobber those records.
            VideoProject(project.slug, cfg).manifest.mark(name, "failed")
            raise RuntimeError(f"stage {name!r} failed (exit {rc}) — see the project log.")
    return project.output
