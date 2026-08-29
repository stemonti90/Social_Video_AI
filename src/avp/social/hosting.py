"""Temporarily expose a rendered video at a public URL.

Instagram's Graph API has no file-upload path for Reels: ``POST /{ig-user}/media`` takes a
``video_url`` and Meta's servers fetch it themselves. So the MP4 has to be reachable from the public
internet for the length of one publish — a real constraint, not a design choice.

We push it to a folder the existing site already serves, hand Meta the URL, then delete it. The file
lives under a random name so nothing is guessable, and it is removed in a `finally` so a failed publish
does not leave a video sitting on a public host.

The target folder must NOT be tracked by git: the site repo deploys by `git pull --ff-only` and a
committed 20 MB video would end up in its history forever.
"""
from __future__ import annotations

import secrets
import subprocess
from pathlib import Path

from ..log import get_logger

log = get_logger("avp.social.hosting")


def _conf(cfg) -> dict:
    host = getattr(cfg.publish, "media_host", None) or {}
    missing = [k for k in ("ssh", "dir", "url") if not host.get(k)]
    if missing:
        raise RuntimeError(
            "Instagram needs a public URL for the video, but publish.media_host is not configured "
            f"(missing: {', '.join(missing)}). Set it to e.g. "
            '{ssh: titan-prod, dir: /opt/astrostackerpro-site/media, '
            'url: "https://www.astrostackerpro.com/media"}')
    return host


class PublicCopy:
    """Context manager: a video reachable at ``.url`` for as long as the block runs."""

    def __init__(self, video: Path, cfg):
        self.video, self.cfg = video, cfg
        self.name = f"{secrets.token_urlsafe(16)}.mp4"
        self.url = ""

    def __enter__(self) -> "PublicCopy":
        h = _conf(self.cfg)
        remote = f"{h['ssh']}:{h['dir'].rstrip('/')}/{self.name}"
        subprocess.run(["ssh", h["ssh"], "mkdir", "-p", h["dir"]], check=True, capture_output=True)
        subprocess.run(["scp", "-q", str(self.video), remote], check=True, capture_output=True)
        self.url = f"{h['url'].rstrip('/')}/{self.name}"
        log.info("Video staged publicly for Instagram (%s)", self.url)
        return self

    def __exit__(self, *exc) -> None:
        h = _conf(self.cfg)
        try:
            subprocess.run(["ssh", h["ssh"], "rm", "-f", f"{h['dir'].rstrip('/')}/{self.name}"],
                           check=True, capture_output=True, timeout=60)
            log.info("Removed the staged copy.")
        except Exception as e:  # noqa: BLE001 — never mask the real publishing error with a cleanup one
            log.warning("Could not remove the staged video %s (%s) — delete it by hand.", self.url, e)
