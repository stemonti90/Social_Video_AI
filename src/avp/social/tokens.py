"""Where the social access tokens live.

One small JSON file, mode 0600, outside the repo. Not a database: this pipeline serves ONE creator
posting to their OWN accounts, so a keyring-grade store would be ceremony without a threat model it
actually answers. What matters is that the file never lands in git and never gets logged.

Tokens are refreshed lazily, at the moment a publisher needs one — a daily-cron design must not assume
the process was alive when the token expired.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..log import get_logger

log = get_logger("avp.social.tokens")

# Refresh this many seconds BEFORE the stated expiry. An upload takes a while and a token that dies
# mid-chunk fails the whole post, so we never cut it fine.
SKEW_SECONDS = 300


def store_path() -> Path:
    return Path(os.environ.get("AVP_TOKEN_STORE", Path.home() / ".avp" / "social_tokens.json"))


def _load_all() -> dict:
    p = store_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001 — a corrupt store must not crash a scheduled run
        log.warning("Token store at %s is unreadable (%s) — treating it as empty.", p, e)
        return {}


def _save_all(data: dict) -> None:
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write via a private temp file then rename: an interrupted write must never leave a half-file
    # that costs the user a re-authorisation.
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(p)


def get(platform: str) -> dict | None:
    """The stored record for a platform, or None if that account was never connected."""
    return _load_all().get(platform)


def put(platform: str, record: dict) -> None:
    """Store (or replace) a platform's record, stamping `expires_at` from `expires_in` when present."""
    rec = dict(record)
    # `expires_at: None` must count as absent, not as "never expires": a refresh returns a fresh
    # expires_in alongside a cleared expires_at, and treating None as a real value would make the
    # token look permanently fresh and 401 on the first post after the real expiry.
    if rec.get("expires_at") is None:
        rec.pop("expires_at", None)
        if "expires_in" in rec:
            rec["expires_at"] = int(time.time()) + int(rec.pop("expires_in"))
    rec.pop("expires_in", None)
    data = _load_all()
    data[platform] = {**data.get(platform, {}), **rec}
    _save_all(data)
    log.info("Saved %s credentials to %s", platform, store_path())


def forget(platform: str) -> bool:
    data = _load_all()
    if platform not in data:
        return False
    del data[platform]
    _save_all(data)
    return True


def connected() -> dict[str, dict]:
    """Every connected platform → a redacted summary, safe to print. Never returns raw tokens:
    this feeds `avp accounts`, and a token that reaches a terminal reaches the scrollback too."""
    out = {}
    for plat, rec in _load_all().items():
        exp = rec.get("expires_at")
        out[plat] = {
            "account": rec.get("account") or rec.get("open_id") or "(unknown)",
            "scope": rec.get("scope", ""),
            "expires_in": (int(exp - time.time()) if exp else None),
            "expired": bool(exp and exp <= time.time()),
        }
    return out


def is_fresh(record: dict | None) -> bool:
    """True if the access token is still usable (with the safety skew applied)."""
    if not record or not record.get("access_token"):
        return False
    exp = record.get("expires_at")
    return not exp or exp - SKEW_SECONDS > time.time()
