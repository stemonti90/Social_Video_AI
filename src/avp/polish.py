"""The strong model's final pass over the script: the channel's voice, locked to the fact sheet.

Why. With the fact sheet in its prompt the local writer STILL reached past it — "a hidden river of
gas ten kilometres thick" on Pluto, where the sheet said solid nitrogen ice — and the fact-check
then did its job and repaired all six lines into sentences that are true and dead ("A frozen plain
of nitrogen ice spreads across a world where temperatures hover near -230 degrees Celsius"). Two
models each doing what they are good at left the video with the worst of both: the small one's
invention removed, the big one's correction without any voice.

So the strong model gets one more turn, as head writer: here is the sheet, here is the correct but
flat script, rewrite the title and every narration in the channel's voice — wonder, a reveal per
segment, escalation — using no fact that is not on the sheet, keeping the segment count, the order,
the visuals and roughly the length. The result is then checked exactly like a draft: morbid words,
copied exemplar lines, a length that drifted — any of those and the original stays. The fact-check
runs AFTER this pass, so a rewrite that reached past the sheet is still caught.

Fail-soft, like the brief and the fact-check: no key, no network, a malformed reply → the script
is left as it was.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from . import factcheck
from .models import Script

log = logging.getLogger(__name__)

SYSTEM = (
    "You are the head writer of a faceless short-form astronomy channel (TikTok / Reels / Shorts). "
    "You rewrite a checked but flat script into the channel's voice without adding a single fact that "
    "is not on the fact sheet you are given. Return STRICT JSON only."
)

VOICE = """THE VOICE:
- The tone is WONDER, not the morgue. Curious, confident, a little playful — someone who cannot wait
  to tell you this. Stakes come from awe, scale, strangeness, things happening right now.
- Segment 1 is the HOOK: its first words RENAME the subject as something that sounds impossible, a
  secret, a scale that breaks intuition — what it is made of, what it does, what it would do to you.
  NO digit and NO unit in the first six words (a number is evidence: it lands in segment 2), never
  the subject's name first, never a question, never "Imagine", "Picture this", "In the vast expanse".
  A description tells ("a vast basin of ice on a cold world"); a hook is a claim the viewer did not
  know was possible, in plain words, the number held back for the next line.
- EVERY segment is a REVEAL: one new, specific fact from the sheet, delivered as a claim that surprises,
  then its consequence in one concrete image. If a line could sit in a textbook unchanged, it is wrong.
- ESCALATE: each segment raises the stakes; the penultimate one is the biggest turn; the last content
  segment lands with a line that would work as the title.
- No EXPLAINER sentences: NO line may begin with "This", "These", "It" or "Its"; never "ensuring",
  "allowing", "creating", "resulting in"; never restate an earlier line; no padding.
- BANNED words: {morbid}; also mind-blowing, incredible, literally, breathtaking, journey, unlock,
  delve, "did you know". No markdown, no emojis.
- These channel lines are OFF-LIMITS verbatim (match the energy, never the words): {exemplars}.
- Metric units. Numbers, dates and names ONLY from the fact sheet; never invent a mechanism; if the
  sheet does not say how, say what happened, not how.
- Spoken cadence, one or two sentences per segment, about the word count given for each (±15%).
- PUNCTUATE FOR THE VOICE: the pauses are part of the message. One idea per sentence, at most 18
  words; a comma exactly where a listener needs a breath; a full stop before the reveal, so the pause
  gives it weight; an em dash only for the twist ("The Moon is leaving us — four centimetres a year");
  never a semicolon, never a parenthesis, never three clauses in one breath.
"""

USER = """{facts}

CURRENT SCRIPT — every claim in it has been checked; the WORDING is what needs your hand:
{script}

Rewrite in the channel's voice, under these constraints:
- Same number of segments, same order, same VISUAL and KEYWORDS (do not return them).
- Per-segment word budget (±15%): {budgets}.
- TITLE: 3-8 words, surprising, no morbid word, no colon-heavy SEO shape.
- cta_bridge: ONE honest sentence of at most 15 words that links THIS topic to looking at or photographing the sky with
  the equipment it really takes (the sheet's last line tells you what a viewer can see), METRIC
  units only (centimetres, never inches). It is
  followed by "Get {app} — link in bio.", so do not write that part.

Return JSON exactly:
{{"title": "...", "segments": [{{"index": 1, "narration": "..."}}, ...], "cta_bridge": "..."}}"""


def _mode(cfg) -> str:
    return str(getattr(getattr(cfg, "script", None), "polish", "auto") or "auto").strip().lower()


def _words(text: str) -> int:
    return len((text or "").split())


def _chat(key: str, model: str, system: str, user: str) -> dict:
    r = requests.post(
        factcheck.DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
              "temperature": 0.7,
              "response_format": {"type": "json_object"}},
        timeout=getattr(factcheck, "TIMEOUT", (30, 180)),
    )
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {(r.text or '')[:200]}")
    return factcheck._extract_json(r.json()["choices"][0]["message"]["content"])


def apply(script: Script, data: dict) -> tuple[Script | None, str]:
    """A new Script from the model's reply, or (None, reason) when the reply fails a guard."""
    from .llm import copied_exemplar, morbid_in_script
    content = [s for s in script.segments if s.kind != "cta"]
    segs = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(segs, list) or len(segs) != len(content):
        return None, f"segment count {len(segs) if isinstance(segs, list) else '?'} != {len(content)}"
    new_lines = []
    for s in segs:
        line = " ".join(str((s or {}).get("narration", "")).split()) if isinstance(s, dict) else ""
        if not line:
            return None, "empty narration"
        new_lines.append(line)
    import re
    head = " ".join(new_lines[0].split()[:6]).lower()
    if re.search(r"\d", head) or re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|dozen|hundred|thousand|"
                                            r"million|billion|trillion)\b", head):
        return None, "hook opens on a number"          # "A gold veil a hundred atoms thick" is a measurement, not a hook
    for x in new_lines:
        if re.match(r"^(This|These|It|Its)\b", x):
            return None, f"explainer opener {x[:32]!r}"
        for sent in re.split(r"(?<=[.!?])\s+", x):
            if _words(sent) > 26:
                return None, f"run-on sentence ({_words(sent)} words): {sent[:40]!r}"
    before = sum(_words(s.narration) for s in content)
    after = sum(_words(x) for x in new_lines)
    if not (0.75 * before <= after <= 1.25 * before):      # ±25% ≈ ±5 s on a 50 s video; the fit already sized it
        return None, f"length drifted {before} → {after} words"
    title = " ".join(str(data.get("title") or "").split()).rstrip(".")
    bridge = " ".join(str(data.get("cta_bridge") or "").split())
    if _words(bridge) > 22:              # a two-sentence bridge made the Moon's CTA 14.9 s and the video 63 s
        return None, f"bridge too long ({_words(bridge)} words)"
    probe = {"title": title or script.title,
             "segments": [{"narration": x} for x in new_lines],
             "cta_bridge": bridge or script.cta_bridge}
    if (mw := morbid_in_script(probe)):
        return None, f"morbid word {mw!r}"
    if (ph := copied_exemplar(probe)):
        return None, f"copied exemplar {ph!r}"
    out = Script.from_dict(script.to_dict())
    kept = [s for s in out.segments if s.kind != "cta"]
    for seg, line in zip(kept, new_lines):
        seg.narration = line
    if title:
        out.title = title
    if bridge:
        out.cta_bridge = bridge
        out.bridge_kind = out.bridge_kind or "shoot"
    return out, "ok"


def run(script: Script, facts: str | None, cfg, out_dir: Path | None = None) -> Script:
    """The polished script, or the input unchanged when the pass is off, unconfigured or fails a guard."""
    mode = _mode(cfg)
    if mode == "off":
        return script
    key = factcheck._api_key(cfg)
    if not key:
        if mode == "on":
            log.warning("Polish requested but no API key (DEEPSEEK_API_KEY or script.factcheck_key).")
        return script
    from .llm import EXEMPLAR_PHRASES, MORBID_WORDS
    model = (str(getattr(cfg.script, "brief_model", "") or "").strip()
             or str(getattr(cfg.script, "factcheck_model", "deepseek-chat") or "deepseek-chat"))
    content = [s for s in script.segments if s.kind != "cta"]
    if not content:
        return script
    current = {"title": script.title,
               "segments": [{"index": s.index, "narration": s.narration, "visual": s.visual} for s in content],
               "cta_bridge": script.cta_bridge}
    budgets = ", ".join(f"seg {s.index}: {_words(s.narration)} words" for s in content)
    facts_block = facts.strip() if facts and facts.strip() else \
        "FACT SHEET: none available — use ONLY facts already present in the current script."
    system = SYSTEM + "\n\n" + VOICE.format(
        morbid=", ".join(MORBID_WORDS[:24]) + ", …",
        exemplars="; ".join(f'"{p}"' for p in EXEMPLAR_PHRASES[:6]))
    user = USER.format(facts=facts_block, script=json.dumps(current, ensure_ascii=False, indent=1),
                       budgets=budgets, app=getattr(getattr(cfg, "funnel", None), "app_name", "the app"))
    note = ""
    reasons: list[str] = []
    for attempt in range(5):        # each guard trips once; the retry carries EVERY earlier reason
        try:
            data = _chat(key, model, system, user + note)
        except Exception as e:  # noqa: BLE001 — a polish must never sink a build
            log.warning("Polish pass failed (%s) — keeping the script as written.", e)
            return script
        out, why = apply(script, data)
        if out is not None:
            log.info("Polish: script rewritten in the channel's voice by %s (%d segments).", model, len(content))
            if out_dir:
                try:
                    (out_dir / "polish.json").write_text(json.dumps(
                        {"model": model, "before": current,
                         "after": {"title": out.title, "cta_bridge": out.cta_bridge,
                                   "segments": [{"index": s.index, "narration": s.narration}
                                                for s in out.segments if s.kind != "cta"]}},
                        indent=2, ensure_ascii=False))
                except OSError:
                    pass
            return out
        log.warning("Polish attempt %d rejected (%s).", attempt + 1, why)
        reasons.append(why)
        note = ("\n\nYour previous attempts were REJECTED for these reasons — avoid ALL of them at once: "
                + "; ".join(f"({i}) {r}" for i, r in enumerate(reasons, 1))
                + ". Same number of segments, no digit or number word in the hook's first six words, no line "
                "starting with This/These/It/Its, one to two sentences of at most 18 words each.")
    return script
