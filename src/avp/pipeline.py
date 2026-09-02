"""Orchestrator. The 'build' phase runs everything *after* the human-reviewed script,
skipping stages already marked done (unless force=True)."""
import os
import signal
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
    # OWN PROCESS GROUP + wall-clock watchdog. Both halves earned their place the same night:
    #
    #  * A script stage sat for 12h50m on an Ollama call that never answered. The client's own 10-min
    #    read timeout did not fire — the Mac had slept, and a socket blocked across a suspend does not
    #    re-arm it. Per-request timeouts cannot be the only defence for an unattended daily run: a
    #    hang like that stops the channel with no error for anyone to see.
    #  * Killing the parent left the stage, and mflux under it, still running and still writing into
    #    the project directory — two builds then raced over the same folder. A separate session means
    #    one killpg reaches the whole tree.
    #
    # The limits are wall-clock and deliberately loose (roughly 2x the slowest measured run): they
    # exist to catch a HANG, never to hurry a slow but working stage.
    limit = STAGE_TIMEOUTS.get(name, DEFAULT_STAGE_TIMEOUT)
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        return proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        log.error("stage %r exceeded its %d-minute watchdog and was killed — it was not making "
                  "progress (a hung model call looks exactly like this).", name, limit // 60)
        _kill_tree(proc)
        return 124                                   # the conventional "timed out" exit code
    except KeyboardInterrupt:
        _kill_tree(proc)                             # Ctrl-C must not leave the stage orphaned
        raise


STAGE_TIMEOUTS = {          # wall-clock seconds; ~2x the slowest measured run for each stage
    "script": 2700,         # best-of-3 + 2 refine passes measured ~16 min, and the length loop can
                            # add up to 3 more model calls on top of that
    "voice": 1800,
    "footage": 3600,        # up to 18 generated images at ~2 min each
    "metadata": 1200,
    "captions": 1800,
    "assemble": 1800,       # includes generating the music bed
}
DEFAULT_STAGE_TIMEOUT = 1800


def _kill_tree(proc: subprocess.Popen) -> None:
    """SIGTERM the stage's whole process group, then SIGKILL what ignores it. The group is what
    matters: a stage spawns mflux and ffmpeg, and signalling only the direct child leaves those
    running against the project directory."""
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


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
