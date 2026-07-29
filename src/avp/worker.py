"""Mac GPU worker for the distributed factory.

Claims a job from the control plane (server/control.py), generates the video with the local
Apple-Silicon pipeline, and uploads the finished mp4 + metadata back. The server then schedules it to
Postiz. This is the ONLY half that needs the GPU (MLX/Metal) — the server just orchestrates.

    avp worker --server http://192.168.1.184:8770 --token <CONTROL_TOKEN>
    avp worker --once            # process a single job and exit (for testing / launchd one-shots)
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from .auto import _unique_slug, slugify
from .config import Config
from .log import get_logger
from .manifest import VideoProject

log = get_logger("avp.worker")


def _video_path(project, cfg: Config) -> Path:
    eng = cfg.publish.voice or "kokoro"
    v = project.output_for(eng)
    return v if v.exists() else project.output


def _claim(server: str, headers: dict, name: str):
    r = requests.post(f"{server}/api/jobs/claim", json={"worker": name}, headers=headers, timeout=30)
    if r.status_code == 204 or not (r.text or "").strip():
        return None
    r.raise_for_status()
    return r.json()


def _render_and_upload(job: dict, server: str, headers: dict, cfg: Config, projects_dir: Path,
                       config_path: str) -> None:
    from . import pipeline, stages
    jid, topic = job["id"], job["topic"]
    slug = _unique_slug(slugify(topic), projects_dir)
    project = VideoProject.create(slug, cfg)
    stages.stage_script(project, cfg, topic)
    pipeline.build(project, cfg, config_path=config_path)
    meta_path = project.root / "metadata.json"
    meta = meta_path.read_text() if meta_path.exists() else "{}"
    requests.post(f"{server}/api/jobs/{jid}/metadata", data=meta.encode(),
                  headers={**headers, "Content-Type": "application/json"}, timeout=60).raise_for_status()
    video = _video_path(project, cfg)
    with open(video, "rb") as vf:
        requests.put(f"{server}/api/jobs/{jid}/video", data=vf,
                     headers={**headers, "Content-Type": "video/mp4"}, timeout=1800).raise_for_status()
    log.info("worker: uploaded %s (%s)", jid, video.name)


def run_worker(cfg: Config, server: str, token: str, once: bool = False, poll: int = 60,
               name: str = "mac-worker", config_path: str = "config.yaml") -> None:
    server = server.rstrip("/")
    headers = {"Authorization": token}
    projects_dir = Path(cfg.paths.projects_dir).expanduser()
    log.info("worker %r polling %s every %ds", name, server, poll)
    while True:
        try:
            job = _claim(server, headers, name)
        except Exception as e:  # noqa: BLE001 — a transient control-server hiccup shouldn't kill the worker
            log.warning("worker: claim failed (%s) — retrying in %ds", e, poll)
            job = None
        if not job:
            if once:
                return
            time.sleep(poll)
            continue
        log.info("worker: claimed %s — %r", job["id"], job["topic"])
        try:
            _render_and_upload(job, server, headers, cfg, projects_dir, config_path)
        except Exception as e:  # noqa: BLE001 — one bad job is reported and the worker moves on
            log.error("worker: job %s failed: %s", job["id"], e)
            try:
                requests.post(f"{server}/api/jobs/{job['id']}/fail", json={"error": str(e)[:500]},
                              headers=headers, timeout=15)
            except Exception:  # noqa: BLE001
                pass
        if once:
            return
