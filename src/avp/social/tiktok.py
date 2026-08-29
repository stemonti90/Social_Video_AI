"""TikTok Content Posting API — direct post of a rendered video.

Contract verified against a working client (Postiz's provider, read 2026-08-29) plus TikTok's docs:

  1. POST /v2/post/publish/video/init/   {post_info, source_info} → {publish_id, upload_url}
  2. PUT  upload_url                     the bytes, Content-Range per chunk; 200/201/**206** all mean OK
  3. POST /v2/post/publish/status/fetch/ {publish_id} → poll to PUBLISH_COMPLETE

Two things that bite:

* **206 is success.** A chunked PUT answers "206 Partial Content"; an HTTP client that treats only 2xx
  <300 as success will report a perfectly good upload as failed.
* **The AI-disclosure field is ``is_aigc``.** Not ``video_made_with_ai`` — that name belongs to Postiz's
  own settings object and TikTok silently ignores it, so the video ships undisclosed.

Only three scopes are needed here — ``user.info.basic``, ``video.upload``, ``video.publish``. TikTok's
review explicitly delays apps that request scopes they don't exercise, so we ask for nothing else.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import requests

from ..log import get_logger
from .base import Publisher, app_credentials, http, redirect_uri

log = get_logger("avp.social.tiktok")

API = "https://open.tiktokapis.com/v2"

# A chunk must be 5-64 MB. Anything that fits in one chunk goes in one request; past that we cut 10 MB
# chunks and let the last one carry the remainder (TikTok allows the final chunk to exceed chunk_size).
MAX_SINGLE_CHUNK = 64 * 1024 * 1024
CHUNK_SIZE = 10 * 1024 * 1024

_PRIVACY = {"public": "PUBLIC_TO_EVERYONE",
            "unlisted": "FOLLOWER_OF_CREATOR",
            "private": "SELF_ONLY"}


def chunk_plan(size: int) -> tuple[int, int]:
    """(chunk_size, total_chunk_count) for a file of `size` bytes."""
    if size <= MAX_SINGLE_CHUNK:
        return size, 1
    return CHUNK_SIZE, size // CHUNK_SIZE


class TikTok(Publisher):
    platform = "tiktok"
    scopes = ("user.info.basic", "video.upload", "video.publish")

    # ---------------------------------------------------------------- connecting
    def authorize_url(self, state: str, cfg=None) -> str:
        key, _ = app_credentials(self.platform, cfg)
        return "https://www.tiktok.com/v2/auth/authorize/?" + urlencode({
            "client_key": key,
            "scope": ",".join(self.scopes),
            "response_type": "code",
            "redirect_uri": redirect_uri(self.platform),
            "state": state,
            # Never auto-skip the consent screen on re-auth: the person connecting should always
            # see what they are granting (and the app-review demo has to show this screen).
            "disable_auto_auth": "1",
        })

    def _token_call(self, form: dict, cfg) -> dict:
        key, secret = app_credentials(self.platform, cfg)
        data = http("POST", f"{API}/oauth/token/",
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={"client_key": key, "client_secret": secret, **form})
        if "access_token" not in data:
            raise RuntimeError(f"TikTok refused the token request: {data}")
        return data

    def exchange(self, code: str, cfg=None) -> dict:
        d = self._token_call({"code": code, "grant_type": "authorization_code",
                              "redirect_uri": redirect_uri(self.platform)}, cfg)
        rec = {"access_token": d["access_token"], "refresh_token": d.get("refresh_token", ""),
               "expires_in": d.get("expires_in", 86400), "open_id": d.get("open_id", ""),
               "scope": d.get("scope", "")}
        try:
            # Only fields covered by user.info.basic: adding e.g. `username` (a user.info.profile
            # field) makes TikTok 401 the WHOLE request with scope_not_authorized (seen 2026-08-29).
            info = http("GET", f"{API}/user/info/?fields=display_name",
                        headers={"Authorization": f"Bearer {rec['access_token']}"})
            rec["account"] = (info.get("data", {}).get("user", {}).get("display_name") or "")
        except Exception as e:  # noqa: BLE001 — a nice label is not worth failing the connection over
            log.debug("Could not read the TikTok display name (%s)", e)
        return rec

    def refresh(self, record: dict, cfg=None) -> dict:
        if not record.get("refresh_token"):
            raise RuntimeError("No TikTok refresh token — run `avp connect tiktok` again.")
        d = self._token_call({"grant_type": "refresh_token",
                              "refresh_token": record["refresh_token"]}, cfg)
        return {**record, "access_token": d["access_token"],
                "refresh_token": d.get("refresh_token", record["refresh_token"]),
                "expires_in": d.get("expires_in", 86400), "expires_at": None}

    # ---------------------------------------------------------------- posting
    def creator_info(self, token: str) -> dict:
        """What the creator's account currently allows. Queried BEFORE posting because TikTok requires
        the privacy level we send to be one the account actually offers — a private or under-16 account
        rejects PUBLIC_TO_EVERYONE, and the error arrives only at init time."""
        return http("POST", f"{API}/post/publish/creator_info/query/",
                    headers={"Authorization": f"Bearer {token}",
                             "Content-Type": "application/json; charset=UTF-8"}).get("data", {})

    def _upload_bytes(self, upload_url: str, video: Path, size: int) -> None:
        chunk, total = chunk_plan(size)
        with video.open("rb") as fh:
            for i in range(total):
                start = i * chunk
                end = size - 1 if i == total - 1 else start + chunk - 1
                fh.seek(start)
                body = fh.read(end - start + 1)
                headers = {
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(body)),
                    "Content-Range": f"bytes {start}-{end}/{size}",
                }
                # Scalar timeout, NOT a (connect, read) tuple: with the tuple, sending a large body
                # hit "The write operation timed out" after ~10s — the connect value was applied to
                # the socket writes (seen 2026-08-29 pushing 22 MB). One retry per chunk: the
                # upload_url stays valid and re-PUTting the same byte range is idempotent.
                last_err: Exception | None = None
                for attempt in (1, 2):
                    try:
                        r = requests.put(upload_url, data=body, timeout=900, headers=headers)
                        break
                    except Exception as e:  # noqa: BLE001 — network hiccup; retry once then surface
                        last_err = e
                        log.warning("Chunk %d/%d attempt %d failed (%s)%s", i + 1, total, attempt, e,
                                    " — retrying" if attempt == 1 else "")
                else:
                    raise RuntimeError(f"TikTok upload failed on chunk {i + 1}/{total}: {last_err}")
                # 206 = "chunk accepted, send the next one". Not an error.
                if r.status_code not in (200, 201, 206):
                    raise RuntimeError(f"TikTok rejected chunk {i + 1}/{total} "
                                       f"({r.status_code}): {(r.text or '')[:300]}")
                log.info("TikTok upload %d/%d (%.1f MB)", i + 1, total, len(body) / 1e6)

    def post(self, video: Path, caption: str, meta: dict, cfg, token: str,
             record: dict, disclose_ai: bool) -> dict:
        size = video.stat().st_size
        if size == 0:
            raise RuntimeError(f"{video} is empty — nothing to upload.")
        chunk, total = chunk_plan(size)
        want = _PRIVACY.get(getattr(cfg.publish, "privacy", "public"), "PUBLIC_TO_EVERYONE")

        allowed = self.creator_info(token).get("privacy_level_options") or []
        if allowed and want not in allowed:
            # Better a private post than a failed one: the creator can flip it in the app.
            log.warning("TikTok account does not allow %s (allows %s) — posting as %s.",
                        want, ", ".join(allowed), allowed[0])
            want = allowed[0]

        def _init(privacy: str) -> dict:
            return http("POST", f"{API}/post/publish/video/init/",
                        headers={"Authorization": f"Bearer {token}",
                                 "Content-Type": "application/json; charset=UTF-8"},
                        json={
                            "post_info": {
                                "title": caption[:2200],
                                "privacy_level": privacy,
                                "disable_duet": False,
                                "disable_comment": False,
                                "disable_stitch": False,
                                "is_aigc": bool(disclose_ai),
                                "brand_content_toggle": False,
                                "brand_organic_toggle": False,
                            },
                            "source_info": {"source": "FILE_UPLOAD", "video_size": size,
                                            "chunk_size": chunk, "total_chunk_count": total},
                        }).get("data", {})

        try:
            init = _init(want)
        except RuntimeError as e:
            # Until the app passes TikTok's audit, direct post is allowed ONLY as SELF_ONLY on a
            # private account — creator_info does not advertise this, the init call just 403s
            # (verified 2026-08-29: a private account offering FOLLOWER_OF_CREATOR still got the
            # unaudited_client error until we sent SELF_ONLY). Post rather than fail: the creator
            # can flip visibility in the app, and after the audit this branch simply stops firing.
            if "unaudited_client_can_only_post_to_private_accounts" in str(e) and want != "SELF_ONLY":
                log.warning("TikTok app not audited yet — posting as SELF_ONLY "
                            "(only the account owner can see it).")
                want = "SELF_ONLY"
                init = _init(want)
            else:
                raise
        publish_id, upload_url = init.get("publish_id"), init.get("upload_url")
        if not publish_id or not upload_url:
            raise RuntimeError(f"TikTok did not return an upload target: {init}")

        self._upload_bytes(upload_url, video, size)

        def status() -> dict:
            return http("POST", f"{API}/post/publish/status/fetch/",
                        headers={"Authorization": f"Bearer {token}",
                                 "Content-Type": "application/json; charset=UTF-8"},
                        json={"publish_id": publish_id}).get("data", {})

        final = self.poll(status, lambda d: d.get("status") in ("PUBLISH_COMPLETE", "FAILED"),
                          what="TikTok publish")
        if final.get("status") != "PUBLISH_COMPLETE":
            raise RuntimeError(f"TikTok publish failed: {final.get('fail_reason') or final}")
        ids = final.get("publicaly_available_post_id") or final.get("publicly_available_post_id") or []
        return {"publish_id": publish_id, "post_id": (ids[0] if ids else None),
                "privacy": want, "disclosed_ai": bool(disclose_ai)}
