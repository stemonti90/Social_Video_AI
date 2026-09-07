"""Posting slots, timezone and topic identity — the one copy of the scheduling rules.

Until 7/9 two stacks scheduled videos — `avp auto` (launchd on the Mac, a text-file topic queue) and
a control plane + GPU worker (Docker, a sqlite queue, Postiz) — and each carried its own `post_slots`,
`zone`, `iso_utc` and its own idea of when two topics are "the same": character for character the
same code, so a fix landed in one place and not the other. The rules were extracted here first; the
second stack, never used in production, was then retired. Stdlib-only, so anything can import it.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except Exception:  # noqa: BLE001 — no tz database on this interpreter
    ZoneInfo = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

DEFAULT_TIMES: tuple[tuple[int, int], ...] = ((12, 0), (18, 0), (21, 0))
MAX_DAYS_AHEAD = 15               # rolling bound for slot search — a safety net, never reached in practice


def zone(tz: str):
    """A tzinfo for `tz` (IANA name), UTC when the name is unknown or zoneinfo is unavailable."""
    if ZoneInfo is not None and tz:
        try:
            return ZoneInfo(tz)
        except Exception:  # noqa: BLE001 — bad tz name
            log.warning("Unknown timezone %r — using UTC.", tz)
    return timezone.utc


def iso_utc(dt: datetime) -> str:
    """The Postiz/Meta-style UTC timestamp with millisecond zeros: 2026-09-07T10:00:00.000Z."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def parse_times(times) -> list[tuple[int, int]]:
    """"HH:MM" strings → (hour, minute) pairs; malformed entries are skipped; empty → DEFAULT_TIMES."""
    parsed: list[tuple[int, int]] = []
    for t in times or []:
        try:
            hh, mm = (int(x) for x in str(t).split(":")[:2])
        except Exception:  # noqa: BLE001 — skip a malformed time entry
            continue
        if 0 <= hh < 24 and 0 <= mm < 60:
            parsed.append((hh, mm))
    return parsed or list(DEFAULT_TIMES)


def post_slots(now: datetime, times, tz: str, count: int) -> list[datetime]:
    """The next `count` posting datetimes (tz-aware) from the daily `times` (HH:MM), rolling into
    following days once today's remaining slots are used up. Only slots strictly in the future."""
    z = zone(tz)
    now = now.astimezone(z)
    parsed = parse_times(times)
    slots: list[datetime] = []
    for day in range(0, MAX_DAYS_AHEAD):
        base = now + timedelta(days=day)
        for hh, mm in parsed:
            cand = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if cand > now:
                slots.append(cand)
                if len(slots) >= count:
                    return slots
    return slots


def topic_key(s: str) -> str:
    """Topic identity for de-duplication: lowercase, punctuation folded to single spaces.
    'The Great Red Spot!' and 'the great  red spot' are the same topic."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
