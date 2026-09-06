"""A fact sheet from the stronger model BEFORE the writer starts.

Why this exists. The writer (gemma4:12b, local) is good at rhythm and does not know the facts of a
less-travelled topic. Left alone on "Philae's bouncing landing" it wrote about ghost-soil and a
perpetual pirouette; the fact-check then flagged six lines out of seven and repaired them into flat,
true, dead sentences. Checking after the fact cannot put back what the writer never had.

So the same strong model that audits the script is asked, first, for the facts: ten to fourteen
precise, verifiable statements with their numbers, dates and names, the three most surprising
angles, and the misconceptions the writer must not repeat. The writer receives that sheet as the
ONLY fact base it may draw on. It still chooses, phrases and orders; it no longer invents. The
fact-check still runs afterwards — the sheet narrows the writer, it does not replace the audit.

Fail-soft on purpose: no key, no network, a malformed reply → no sheet, a warning, and the writer
proceeds exactly as before. A missing brief must never stop the daily video.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from . import factcheck

log = logging.getLogger(__name__)

SYSTEM = (
    "You are the research desk of a short-form astronomy/space video channel. Given a topic, return "
    "the facts a scriptwriter may rely on. Only include what you are confident is true and can be "
    "checked in standard references (space agency pages, peer-reviewed results, Wikipedia). Prefer "
    "numbers, dates, sizes, speeds, durations, names of missions and instruments, and firsts. Where a "
    "figure is disputed or has been revised, give the current accepted value and say so briefly. "
    "Never invent; if the topic is thin, return fewer facts. Return STRICT JSON only."
)

USER = """Topic: {topic}

Return JSON exactly:
{{
  "facts": ["10-14 one-sentence facts, each with its number/date/name, most surprising first"],
  "wonder": ["3 short angles that make a reader stop scrolling — true, concrete, no adjectives"],
  "avoid": ["3-6 common errors or myths about this topic the script must NOT repeat"],
  "see_it": "one sentence: can a person see or photograph this (or its host body) with eyes, binoculars, a small telescope or a phone? If so how; if not, say so"
}}"""


def _mode(cfg) -> str:
    return str(getattr(getattr(cfg, "script", None), "brief", "auto") or "auto").strip().lower()


def render(data: dict) -> str:
    """The sheet as the writer sees it: numbered facts, the angles, the don'ts."""
    facts = [str(f).strip() for f in (data.get("facts") or []) if str(f).strip()]
    if not facts:
        return ""
    lines = ["FACT SHEET — verified by a stronger model. These are the ONLY facts you may use:"]
    lines += [f"  {i}. {f}" for i, f in enumerate(facts, 1)]
    wonder = [str(w).strip() for w in (data.get("wonder") or []) if str(w).strip()]
    if wonder:
        lines.append("SURPRISING ANGLES (build the hook from one of these):")
        lines += [f"  - {w}" for w in wonder]
    avoid = [str(a).strip() for a in (data.get("avoid") or []) if str(a).strip()]
    if avoid:
        lines.append("DO NOT CLAIM (common errors):")
        lines += [f"  - {a}" for a in avoid]
    see = str(data.get("see_it") or "").strip()
    if see:
        lines.append(f"CAN THE VIEWER SEE IT: {see}")
    return "\n".join(lines)


def build(topic: str, cfg, out_dir: Path | None = None) -> str | None:
    """The fact sheet for `topic`, or None when the brief is off, unconfigured or failed.

    `script.brief`: "auto" (default) = on whenever a fact-check key exists; "on" = required (warns
    loudly when it cannot run); "off" = never. The result is cached in <out_dir>/brief.json so a
    re-run of the script stage does not pay for the same research twice."""
    mode = _mode(cfg)
    if mode == "off":
        return None
    key = factcheck._api_key(cfg)
    if not key:
        if mode == "on":
            log.warning("Fact brief requested but no API key (DEEPSEEK_API_KEY or script.factcheck_key).")
        return None
    cache = (out_dir / "brief.json") if out_dir else None
    if cache and cache.exists():
        try:
            old = json.loads(cache.read_text())
            if old.get("topic") == topic and render(old):
                log.info("Fact brief: reusing %s", cache)
                return render(old)
        except Exception:  # noqa: BLE001 — a bad cache is just re-researched
            pass
    model = (str(getattr(cfg.script, "brief_model", "") or "").strip()
             or str(getattr(cfg.script, "factcheck_model", "deepseek-chat") or "deepseek-chat"))
    try:
        r = requests.post(
            factcheck.DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": USER.format(topic=topic)}],
                  "temperature": 0.0,
                  "response_format": {"type": "json_object"}},
            timeout=getattr(factcheck, "TIMEOUT", (30, 180)),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {(r.text or '')[:200]}")
        data = factcheck._extract_json(r.json()["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001 — the brief is an aid, never a gate
        log.warning("Fact brief failed (%s) — the writer proceeds without a sheet.", e)
        return None
    text = render(data if isinstance(data, dict) else {})
    if not text:
        log.warning("Fact brief returned no facts — the writer proceeds without a sheet.")
        return None
    data["topic"] = topic
    data["model"] = model
    if out_dir:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "brief.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
            (out_dir / "brief.md").write_text(f"# Fact sheet — {topic}\n\n{text}\n")
        except OSError as e:
            log.debug("could not save the brief (%s)", e)
    log.info("Fact brief: %d facts from %s.", len(data.get("facts") or []), model)
    return text
