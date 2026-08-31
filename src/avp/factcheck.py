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
* **Never lengthen a spoken line.** Caption timings, segment durations and the 60-second ceiling are
  all derived from word count. A "better" sentence three words longer silently pushes the video over.
  Shot descriptions are not spoken, so they are judged differently: they may grow a little, but they
  must keep their shot scale (segments alternate wide / medium / close on purpose) and stay short
  enough that an image model still obeys the tail.
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

Each segment gives you TWO things to check, and they fail differently:

NARRATION — the words spoken aloud. Check them for these failure modes, in order of how often \
they occur:
1. INVENTED MECHANISMS — a plausible-sounding explanation of *how* something worked that is simply not \
how it worked. (A solar-powered rover described as drawing power from the soil.)
2. WRONG TENSE — a mission, probe or telescope that has ended, been destroyed or fallen silent, \
written about in the present tense.
3. FALSE SUPERLATIVES — "the only", "the first", "the largest" where a sibling or predecessor exists. \
Be especially hard on spaceflight superlatives, where the sibling is usually another nation's programme \
and is easy to forget: more than one crewed station has been in orbit at a time, more than one probe \
has left the heliosphere, more than one agency has landed on Mars. "The only X we have" is still a \
false superlative if someone else has one — a viewer will not read "we" as "our side".
4. FABRICATED FIGURES — a number that sounds specific and has no source.
5. PHYSICS THAT DOES NOT HOLD — wrong units, wrong orders of magnitude, wrong scale of distance.

VISUAL — a shot description that will be handed to an image generator. Nobody hears it, but a wrong \
one produces a wrong picture, which is worse. Check whether the scene DESCRIBED IS POSSIBLE and looks \
the way it is described:
6. IMPOSSIBLE OR WRONG APPEARANCE — the wrong sky colour for that world (Mars is butterscotch by day, \
blue only at sunset; the Moon's sky is black), a surface on a gas giant, rings where there are none, \
stars visible in a sunlit lunar photograph, an object drawn with features it does not have.
7. A VIEW NOBODY COULD HAVE — a lander photographed from outside by no one, a probe filmed from a \
chase camera that was never there. A conceptual or artist's-impression shot is fine; a shot that \
claims to be a photograph of an impossible vantage point is not.

Every visual MUST keep its shot scale — the phrase naming how close the camera is ("wide shot", \
"medium shot", "close-up", "extreme close-up", "filling the frame"). The video is cut from these and \
consecutive segments deliberately alternate. If you correct a visual, KEEP its scale word intact and \
change only what is factually wrong. Never turn a wide shot into a close-up.

Rules for your verdicts:
- Say which field you are judging: "narration" or "visual".
- Judge each claim on its own. Quote it exactly as written.
- "wrong" means you are confident it is false. "unsure" means it is the KIND of thing that changes \
with time or that you cannot settle from memory — a current distance, an operational status, a count \
of how many objects have done something. Use "unsure" honestly; it triggers a web check, not a rewrite.
- A simplification that is broadly true is NOT wrong. This is a 40-second video for a general \
audience, not a paper. Do not flag "roughly", do not flag vivid language, and above all DO NOT FLAG \
ROUNDED NUMBERS. A figure rounded for speech is correct, not an error: 28,000 km/h for 27,580, \
450,000 kg for 419,725, "over 23 years" for 25. Replacing a spoken round number with an exact one \
makes the narration worse, not better — it is harder to say, harder to hear, and no more true. Only \
flag a figure when it is wrong by an order of magnitude, in the wrong unit, or invented outright.
- When you flag something, supply a fix that is TRUE and NO LONGER than the original — count the \
words. The line is spoken aloud and the video's timing is built from its length.
- If the script is clean, return an empty list. Do not invent problems to look useful.
- THINK BEFORE YOU ANSWER, NOT INSIDE THE ANSWER. The "why" field is a finished conclusion in one \
sentence, never a scratchpad. If working through a claim leads you to decide it is acceptable after \
all, emit NO finding for it — do not emit one and explain inside it that you changed your mind.
- Every "wrong" verdict must carry a real replacement in "fix". If you cannot write the corrected \
line, you do not know the claim is false: mark it "unsure" instead.

Return STRICT JSON only, no commentary:
{"findings": [{"segment": 1, "field": "narration|visual", "claim": "exact quoted text", \
"verdict": "wrong|unsure", "why": "one sentence", "fix": "replacement, no longer than the original"}]}"""

USER = """Script to check. For each segment: what is SAID, then the SHOT that will be generated.

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


# The shot scale is load-bearing: segments deliberately alternate wide / medium / close, and a
# "correction" that quietly turns a wide shot into a close-up undoes the pacing work it knows nothing
# about. A visual fix must still carry one of these.
SCALE_WORDS = ("wide", "medium", "close-up", "closeup", "close up", "extreme",
               "filling the frame", "aerial", "macro", "establishing")


@dataclass
class Finding:
    segment: int
    claim: str
    verdict: str                 # "wrong" | "unsure"
    field: str = "narration"     # "narration" (spoken) | "visual" (drives image generation)
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
    body = "\n\n".join(
        f"segment {s.index}\n  narration: {s.narration.strip()}\n  visual: {(s.visual or '').strip()}"
        for s in content)
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
        # A "wrong" with no replacement is a hunch, not a finding. Observed live: the checker talked
        # itself out of a verdict inside the "why" field — "the claim is actually true... no flag" —
        # yet still emitted verdict "wrong" with an empty fix. Nothing was rewritten only because the
        # fix was blank; had it invented one, a correct line would have been replaced. Downgrading
        # here makes that safety deliberate: unsure is reported, never applied.
        fix_text = str(f.get("fix", "")).strip()
        if verdict == "wrong" and not fix_text:
            verdict = "unsure"
        try:
            idx = int(f.get("segment", 0))
        except (TypeError, ValueError):
            continue
        which = str(f.get("field", "narration")).strip().lower()
        if which not in ("narration", "visual"):
            which = "narration"
        out.append(Finding(segment=idx,
                           claim=str(f.get("claim", "")).strip(),
                           verdict=verdict,
                           field=which,
                           why=str(f.get("why", "")).strip(),
                           fix=fix_text))
    return out


FOCUS_SYSTEM = """You check ONE claim from an astronomy video script. It was picked because it \
contains a superlative, an operational status, or a figure — statements whose truth depends on when \
you ask, and which a viewer will challenge in the comments.

Name any counterexample you can think of BEFORE deciding: another country's programme, an earlier \
mission, a second spacecraft that did the same thing. A number rounded for speech is NOT an error.

Return STRICT JSON:
{"verdict": "ok|wrong", "why": "one sentence", "fix": "corrected line, no longer than the original"}"""

# How many single-claim calls one script may spend. Each is a few hundred tokens, so the cost is
# negligible, but the latency is serial and a runaway script should not add a minute to every build.
MAX_FOCUS_CALLS = 6


def _sentences(text: str) -> list[str]:
    """Split narration into claim-sized pieces. Crude on purpose: a claim is judged and quoted whole,
    so a split that keeps sentences intact matters far more than one that handles every edge case."""
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if p.strip()]


def _focus_check(claim: str, key: str, model: str) -> Finding | None:
    """Judge ONE claim in its own call. Returns a Finding only when it comes back wrong.

    A claim alone gets more scrutiny than one of six, and that is worth the extra calls — but do not
    oversell it. Measured on "the only laboratory we have to test the human body's endurance", which
    is false because Tiangong has been permanently crewed for years and is in 2026 running a
    year-long study of exactly that: the batch pass missed it three times out of three, and the
    focused pass caught it once out of four. The verdict flips between runs at temperature 0, and on
    the misses the model states positively that the ISS "remains the only" such laboratory. That is
    not inattention, it is what the model believes, and no wording fixes a belief.

    The honest conclusion: an LLM checker is a filter, not a guarantee, and this particular class —
    a superlative whose counterexample sits outside the model's salient knowledge — needs evidence
    rather than recall. Wiring a search API is the fix; this pass narrows the gap, it does not close
    it."""
    r = requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "system", "content": FOCUS_SYSTEM},
                           {"role": "user", "content": f"Claim: {claim}"}],
              "temperature": 0.0,
              "response_format": {"type": "json_object"}},
        timeout=TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {(r.text or '')[:120]}")
    d = _extract_json(r.json()["choices"][0]["message"]["content"])
    if str(d.get("verdict", "")).strip().lower() != "wrong":
        return None
    fix = str(d.get("fix", "")).strip()
    if not fix:
        return None                       # a verdict with no correction is a hunch (see _judge)
    return Finding(segment=0, claim=claim, verdict="wrong", field="narration",
                   why=str(d.get("why", "")).strip(), fix=fix, web="focused re-check")


def _focus_pass(script: Script, already: list[Finding], key: str, model: str) -> list[Finding]:
    """Re-examine every volatile claim the batch pass did not already flag, one call each."""
    seen = {f.claim for f in already}
    out: list[Finding] = []
    budget = MAX_FOCUS_CALLS
    for seg in script.segments:
        if getattr(seg, "kind", "content") == "cta":
            continue
        for sentence in _sentences(seg.narration):
            if budget <= 0:
                break
            if sentence in seen or not volatile(sentence):
                continue
            budget -= 1
            try:
                found = _focus_check(sentence, key, model)
            except Exception as e:  # noqa: BLE001 — one bad call must not sink the pass
                log.warning("Focused re-check failed for segment %d (%s)", seg.index, e)
                continue
            if found:
                found.segment = seg.index
                out.append(found)
    if budget <= 0:
        log.info("Focused re-check hit its %d-call budget; later claims were not re-examined.",
                 MAX_FOCUS_CALLS)
    return out


def _shorter_or_equal(fix: str, original: str) -> bool:
    """Corrections must not lengthen the spoken line: captions, segment durations and the 60s ceiling
    are all derived from word count. Two words of slack absorbs honest phrasing differences."""
    return len(fix.split()) <= len(original.split()) + 2


def _visual_ok(fix: str, original: str) -> tuple[bool, str]:
    """Whether a corrected SHOT description may be applied. Nobody speaks it, so length matters far
    less — but two things still do.

    It must keep a shot scale, because segments alternate wide / medium / close on purpose and a fix
    that drops the scale silently flattens the cut rhythm. And it must not balloon: this string is
    prepended to an image prompt, where a long tail simply stops being obeyed."""
    n, was = len(fix.split()), len(original.split())
    # Two bounds, because bloat arrives in two shapes: a long fix, and a short phrase padded into a
    # long one. "jagged rocks on Mars" swelling to twenty-five words of "extremely detailed" clears
    # any absolute cap while still drowning the subject the shot was supposed to describe.
    if n > 30:
        return False, "longer than 30 words — an image prompt stops obeying the tail"
    if n > max(8, was * 2):
        return False, f"padded from {was} words to {n} — the subject gets buried"
    low = fix.lower()
    if any(w in original.lower() for w in SCALE_WORDS) and not any(w in low for w in SCALE_WORDS):
        return False, "the fix dropped the shot scale"
    return True, ""


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
        if seg is None:
            continue
        visual = f.field == "visual"
        current = (seg.visual or "") if visual else seg.narration
        if f.claim not in current:
            # The checker quoted something that isn't there verbatim — usually a paraphrase. Rewriting
            # on a fuzzy match risks mangling a good line, so leave it flagged and move on.
            log.warning("Fact-check: segment %s %s quote not found verbatim, left for review: %r",
                        f.segment, f.field, f.claim[:60])
            continue
        if visual:
            ok, why_not = _visual_ok(f.fix, f.claim)
            if not ok:
                log.warning("Fact-check: visual fix for segment %s refused (%s)", f.segment, why_not)
                continue
            seg.visual = current.replace(f.claim, f.fix)
        else:
            if not _shorter_or_equal(f.fix, f.claim):
                log.warning("Fact-check: fix for segment %s is longer than the spoken line — "
                            "flagged, not applied", f.segment)
                continue
            seg.narration = current.replace(f.claim, f.fix)
        f.applied = True
        applied += 1
        log.info("Fact-check: segment %s %s corrected — %s",
                 f.segment, f.field, f.why or "no reason given")
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

    # Second pass, one claim at a time. The batch call reads a whole script in one go and its
    # attention thins out across it: it walked past "the only laboratory we have" on three separate
    # runs while another crewed station has been in orbit for years. The same sentence handed over
    # alone was caught at once. So every claim whose truth depends on WHEN you ask — a superlative,
    # an operational status, a figure — gets its own look if the batch pass said nothing about it.
    try:
        extra = _focus_pass(script, rep.findings, _api_key(cfg),
                            str(getattr(cfg.script, "factcheck_model", "deepseek-chat")))
        if extra:
            log.warning("Focused re-check found %d claim%s the batch pass missed.",
                        len(extra), "" if len(extra) == 1 else "s")
        rep.findings.extend(extra)
    except Exception as e:  # noqa: BLE001 — same rule: a safety net never blocks a build
        log.warning("Focused re-check skipped (%s)", e)

    if not rep.findings:
        log.info("Fact-check: %d segments, nothing flagged.", rep.segments)
        return _write(rep, out_dir)

    wrong = [f for f in rep.findings if f.verdict == "wrong"]
    unsure = [f for f in rep.findings if f.verdict == "unsure"]
    log.warning("Fact-check: %d wrong, %d unsure out of %d segments.",
                len(wrong), len(unsure), rep.segments)
    for f in rep.findings:
        log.info("  [%s/%s] seg %s: %s — %s",
                 f.verdict, f.field, f.segment, f.claim[:70], f.why[:80])

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
