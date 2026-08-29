"""Shared machinery for the native social publishers.

Design notes worth keeping:

* **The redirect URI is a STATIC page on the main site.** TikTok and Meta both refuse plain-http
  localhost redirects, and standing up an HTTPS callback service means a new subdomain, a new
  Cloudflare tunnel route and a second domain verification. Instead the callback lands on
  ``https://www.astrostackerpro.com/connect/<platform>.html`` — already inside the URL prefix TikTok
  verified — and that page just shows the authorisation code plus the exact command to paste back.
  One paste, once a year (TikTok refresh tokens last 365 days). No service to run or keep patched.

* **App credentials come from the environment first.** ``config.yaml`` is gitignored, but env vars keep
  secrets out of files entirely and let the same checkout run against sandbox or production creds.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import requests

from ..log import get_logger
from . import tokens

log = get_logger("avp.social")

# Where the OAuth callback pages live. Overridable so a fork can host them elsewhere.
CALLBACK_BASE = os.environ.get("AVP_CALLBACK_BASE", "https://www.astrostackerpro.com/connect")

# Generous: an upload is a long request over a home connection, and a timeout mid-chunk means redoing
# the whole post. Read timeout only — connect stays short so a dead host fails fast.
TIMEOUT = (10, 300)


def redirect_uri(platform: str) -> str:
    return f"{CALLBACK_BASE}/{platform}.html"


def app_credentials(platform: str, cfg=None) -> tuple[str, str]:
    """(client_id, client_secret) for a platform. Env wins over config so the same checkout can be
    pointed at sandbox credentials without editing a file."""
    env = {"tiktok": ("AVP_TIKTOK_CLIENT_KEY", "AVP_TIKTOK_CLIENT_SECRET"),
           "instagram": ("AVP_META_APP_ID", "AVP_META_APP_SECRET"),
           "youtube": ("AVP_GOOGLE_CLIENT_ID", "AVP_GOOGLE_CLIENT_SECRET")}[platform]
    cid, sec = os.environ.get(env[0], ""), os.environ.get(env[1], "")
    if (not cid or not sec) and cfg is not None:
        apps = getattr(cfg.publish, "apps", None) or {}
        entry = apps.get(platform) or {}
        cid = cid or entry.get("client_id", "")
        sec = sec or entry.get("client_secret", "")
    if not cid or not sec:
        raise RuntimeError(
            f"No {platform} app credentials. Set {env[0]} and {env[1]} in the environment "
            f"(or publish.apps.{platform} in config.yaml).")
    return cid, sec


def http(method: str, url: str, **kw) -> dict:
    """One HTTP call returning parsed JSON, raising with the RESPONSE BODY in the message.

    Platform APIs put the real reason in the body and a generic reason in the status line, so a bare
    `raise_for_status()` throws away the only useful information — the difference between "fix your
    scope" and "your video is 3 seconds too long"."""
    kw.setdefault("timeout", TIMEOUT)
    r = requests.request(method, url, **kw)
    try:
        data = r.json()
    except Exception:  # noqa: BLE001 — some endpoints answer 204/empty on success
        data = {}
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {url.split('?')[0]} → {r.status_code}: {(r.text or '')[:400]}")
    return data


class Publisher(ABC):
    """One social platform. Publishers are stateless: everything durable lives in the token store."""

    platform: str = ""
    scopes: tuple[str, ...] = ()

    # ---------------------------------------------------------------- connecting
    @abstractmethod
    def authorize_url(self, state: str, cfg=None) -> str:
        """The URL the creator opens to grant access."""

    @abstractmethod
    def exchange(self, code: str, cfg=None) -> dict:
        """Trade an authorisation code for a token record to store."""

    def refresh(self, record: dict, cfg=None) -> dict:
        """Renew an expired access token. Platforms whose tokens don't expire may keep the default."""
        return record

    # ---------------------------------------------------------------- posting
    @abstractmethod
    def post(self, video: Path, caption: str, meta: dict, cfg, token: str,
             record: dict, disclose_ai: bool) -> dict:
        """Upload and publish. Returns a small result dict; raises on failure."""

    # ---------------------------------------------------------------- shared
    def access_token(self, cfg=None) -> tuple[str, dict]:
        """A valid access token for this platform, refreshing it first if it is at or near expiry.
        Raises with a pointer to `avp connect` when the account was never linked."""
        rec = tokens.get(self.platform)
        if not rec:
            raise RuntimeError(f"{self.platform} is not connected — run `avp connect {self.platform}`.")
        if not tokens.is_fresh(rec):
            log.info("%s access token expired — refreshing.", self.platform)
            rec = self.refresh(rec, cfg)
            tokens.put(self.platform, rec)
            rec = tokens.get(self.platform) or rec
        return rec["access_token"], rec

    @staticmethod
    def poll(fn, done, *, tries: int = 60, every: float = 5.0, what: str = "job") -> dict:
        """Call `fn()` until `done(result)`, then return that result.

        Both TikTok and Instagram accept a post and then transcode it asynchronously, so "the API
        returned 200" is not "the video is live" — the failure usually arrives during processing.
        Polling to a terminal state is what turns a hopeful post into a verified one."""
        last: dict = {}
        for _ in range(tries):
            last = fn()
            if done(last):
                return last
            time.sleep(every)
        raise RuntimeError(f"{what} did not finish within {int(tries * every)}s — last state: {last}")
