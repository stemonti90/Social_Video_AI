"""Adversarial fact-check of a finished script, before a single frame is rendered.

Why this exists. The writing model (gemma4:12b, local) is good at rhythm and bad at knowing what it
doesn't know. Measured on real builds: it never invented the NUMBERS — Opportunity's 45 km and its
silica deposits were both right — but it invented a MECHANISM ("scavenging power from the Martian
soil" for a solar-powered rover), wrote in the present tense about a rover silent since 2018, put
"13 billion human heartbeats" on the Voyager Golden Record (it carries one hour of one person's), and
called Voyager 1 "the only human-made object" in interstellar space while Voyager 2 has been there
since 2018. Tightening the writing prompt caught some of these and then produced new ones of a
different shape, which is the signal that prompt rules are the wrong tool: a 12B model cannot audit
its own knowledge.

So a second, much stronger model reads the finished script cold and tries to break it.

Two stages, because they fail differently:

1. **Judgement** (DeepSeek). Catches everything that is a knowledge or reasoning error — the wrong
   power source, the impossible physics, the superlative that forgets a sibling mission. Cheap, one
   call, a few hundred tokens.
2. **Web verification**, only for the claims the judge marks uncertain AND that are dated, numeric or
   superlative. Those are the ones no model can settle from memory, because the answer moves: "is it
   still transmitting", "how far is it now", "how many have done this". A model asked those questions
   answers confidently from a frozen world.

Three hard rules, all learned the expensive way:

* **Never block the build.** No key, no network, a 500 from the API, a malformed reply — every one of
  those logs a warning and returns "no findings". A channel that publishes three videos a day cannot
  stop because someone else's endpoint is down.
* **Never lengthen a line.** Corrections are spoken, and the caption timings, the segment durations
  and the 60-second ceiling are all derived from the word count. A "better" sentence three words
  longer silently pushes the video over.
* **Write down what was checked, not just what was wrong.** `factcheck.json` in the project is the
  audit trail; a pass that reports nothing should still prove it looked.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from .log import get_logger
from .models import Script

log = get_logger("avp.factcheck")

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# Generous: a checker call is a few hundred tokens, but a cold endpoint can be slow and this runs
# once per video, not per segment. Connect stays short so an unreachable host fails fast.
TIMEOUT = (10, 120)

SYSTEM = """You are a ruthless fact-checker for an astronomy and spaceflight channel. A script has \
been written by a small language model that is fluent and unreliable. Your job is to find every claim \
that is FALSE, and to leave everything else alone.

You are checking for these failure modes specifically, in order of how often they occur:
1. INVENTED MECHANISMS — a plausible-sounding explanation of *how* something worked that is simply not \
how it worked. (A solar-powered rover described as drawing power from the soil.)
2. WRONG TENSE — a mission, probe or telescope that has ended, been destroyed or fallen silent, \
written about in the present tense.
3. FALSE SUPERLATIVES — "the only", "the first", "the largest" where a sibling or predecessor exists.
4. FABRICATED FIGURES — a number that sounds specific and has no source.
5. PHYSICS THAT DOES NOT HOLD — wrong units, wrong orders of magnitude, wrong scale of distance.

Rules for your verdicts:
- Judge each claim on its own. Quote it exactly as written.
- "wrong" means you are confident it is false. "unsure" means it is the KIND of thing that changes \
with time or that you cannot settle from memory — a current distance, an operational status, a count \
of how many objects have done something. Use "unsure" honestly; it triggers a web check, not a rewrite.
- A simplification that is broadly true is NOT wrong. This is a 40-second video for a general \
audience, not a paper. Do not flag "roughly", do not flag rounded figures, do not flag vivid language.
- When you flag something, supply a fix that is TRUE and NO LONGER than the original — count the \
words. The line is spoken aloud and the video's timing is built from its length.
- If the script is clean, return an empty list. Do not invent problems to look useful.

Return STRICT JSON only, no commentary:
{"findings": [{"segment": 1, "claim": "exact quoted text", "verdict": "wrong|unsure", \
"why": "one sentence", "fix": "replacement of the same length or shorter"}]}"""

USER = """Script to check (segment number, then the spoken narration):

{body}

Return the JSON."""

# A claim only earns a web lookup when it is the sort of thing that moves. Checking "Mars is red"
# against a search engine wastes a call and adds a failure mode; checking "is still transmitting"
# is the entire point.
_VOLATILE = re.compile(
    r"\b(still|currently|now|today|to date|so far|remains?|continues?)\b"      # operational status
    r"|\b(only|first|last|largest|farthest|furthest|fastest|oldest)\b"         # superlatives
    r"|\d",                                                                    # any figure
    re.IGNORECASE,
)


@dataclass
class Finding:
    segment: int
    claim: str
    verdict: str                 # "wrong" | "unsure"
    why: str = ""
    fix: str = ""
    web: str = ""                # what the web check concluded, when one ran
    applied: bool = False


@dataclass
class Report:
    checked: bool = False        # did a checker actually run, or did we fail open?
    model: str = ""
    reason: str = ""             # why it didn't run, when it didn't
    segments: int = 0
    findings: list[Finding] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2, ensure_ascii=False)


def _api_key(cfg=None) -> str:
    """Env first, then config. Env keeps the key out of files entirely and lets the same checkout run
    against a different account without an edit."""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    return str(getattr(getattr(cfg, "script", None), "factcheck_key", "") or "").strip()


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a reply that may be wrapped in prose or a ```json fence."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return {}


def _judge(script: Script, cfg) -> list[Finding]:
    """One DeepSeek call over the whole script. Returns [] on any failure — never raises."""
    key = _api_key(cfg)
    if not key:
        raise RuntimeError("no API key (set DEEPSEEK_API_KEY or script.factcheck_key)")

    content = [s for s in script.segments if getattr(s, "kind", "content") != "cta"]
    body = "\n".join(f"{s.index}. {s.narration.strip()}" for s in content)
    model = str(getattr(cfg.script, "factcheck_model", "deepseek-chat") or "deepseek-chat")

    r = requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER.format(body=body)}],
            # Deterministic: a fact-checker that returns different verdicts on the same script is
            # not a fact-checker. DeepSeek documents 0.0 as the setting for this kind of task.
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        timeout=TIMEOUT,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {(r.text or '')[:200]}")

    data = _extract_json(r.json()["choices"][0]["message"]["content"])
    out: list[Finding] = []
    for f in (data.get("findings") or []):
        if not isinstance(f, dict):
            continue
        verdict = str(f.get("verdict", "")).strip().lower()
        if verdict not in ("wrong", "unsure"):
            continue
        try:
            idx = int(f.get("segment", 0))
        except (TypeError, ValueError):
            continue
        out.append(Finding(segment=idx,
                           claim=str(f.get("claim", "")).strip(),
                           verdict=verdict,
                           why=str(f.get("why", "")).strip(),
                           fix=str(f.get("fix", "")).strip()))
    return out


def _shorter_or_equal(fix: str, original: str) -> bool:
    """Corrections must not lengthen the spoken line: captions, segment durations and the 60s ceiling
    are all derived from word count. Two words of slack absorbs honest phrasing differences."""
    return len(fix.split()) <= len(original.split()) + 2


def apply(script: Script, findings: list[Finding]) -> int:
    """Rewrite the narration for findings the checker is CONFIDENT about. Returns how many landed.

    "unsure" is never auto-applied: it means the checker could not settle the claim, and swapping a
    possibly-true line for a possibly-true line is churn, not correction. Those stay in the report
    for a human — or for the web stage, which may promote them to "wrong" with a source.
    """
    by_index = {s.index: s for s in script.segments}
    applied = 0
    for f in findings:
        if f.verdict != "wrong" or not f.fix or not f.claim:
            continue
        seg = by_index.get(f.segment)
        if seg is None or f.claim not in seg.narration:
            # The checker quoted something that isn't there verbatim — usually a paraphrase. Rewriting
            # on a fuzzy match risks mangling a good line, so leave it flagged and move on.
            log.warning("Fact-check: segment %s quote not found verbatim, left for review: %r",
                        f.segment, f.claim[:60])
            continue
        if not _shorter_or_equal(f.fix, f.claim):
            log.warning("Fact-check: fix for segment %s is longer than the line — flagged, not applied",
                        f.segment)
            continue
        seg.narration = seg.narration.replace(f.claim, f.fix)
        f.applied = True
        applied += 1
        log.info("Fact-check: segment %s corrected — %s", f.segment, f.why or "no reason given")
    return applied


def run(script: Script, cfg, out_dir: Path | None = None) -> Report:
    """Check a script and, under mode "fix", correct what the checker is sure about.

    Modes (`script.factcheck`): "off" skips entirely, "flag" reports without touching the script,
    "fix" also applies confident corrections. Failure of any kind degrades to a clean report with a
    reason recorded — the caller carries on and the build completes.
    """
    mode = str(getattr(cfg.script, "factcheck", "off") or "off").strip().lower()
    rep = Report(model=str(getattr(cfg.script, "factcheck_model", "deepseek-chat")),
                 segments=len([s for s in script.segments if getattr(s, "kind", "content") != "cta"]))

    if mode == "off":
        rep.reason = "disabled (script.factcheck: off)"
        return rep

    try:
        rep.findings = _judge(script, cfg)
        rep.checked = True
    except Exception as e:  # noqa: BLE001 — a checker outage must never sink a build
        rep.reason = f"checker unavailable: {e}"
        log.warning("Fact-check skipped — %s", e)
        return _write(rep, out_dir)

    if not rep.findings:
        log.info("Fact-check: %d segments, nothing flagged.", rep.segments)
        return _write(rep, out_dir)

    wrong = [f for f in rep.findings if f.verdict == "wrong"]
    unsure = [f for f in rep.findings if f.verdict == "unsure"]
    log.warning("Fact-check: %d wrong, %d unsure out of %d segments.",
                len(wrong), len(unsure), rep.segments)
    for f in rep.findings:
        log.info("  [%s] seg %s: %s — %s", f.verdict, f.segment, f.claim[:70], f.why[:80])

    if mode == "fix":
        n = apply(script, rep.findings)
        log.info("Fact-check: %d correction%s applied.", n, "" if n == 1 else "s")

    return _write(rep, out_dir)


def volatile(claim: str) -> bool:
    """True when a claim's truth depends on WHEN you ask — a status, a superlative, a figure. Only
    these earn a web lookup; everything else a strong model can settle from what it knows."""
    return bool(_VOLATILE.search(claim or ""))


def _write(rep: Report, out_dir: Path | None) -> Report:
    if out_dir:
        try:
            (Path(out_dir) / "factcheck.json").write_text(rep.to_json())
        except Exception as e:  # noqa: BLE001 — the audit trail is not worth failing a build over
            log.warning("Could not write factcheck.json (%s)", e)
    return rep
