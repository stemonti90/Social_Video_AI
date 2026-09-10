"""The editorial engine — the machine that chooses and develops a story BEFORE anyone writes it.

Third revision of the architecture (10/09/2026), built on the skeleton of branch
feat/editorial-machine-v3. The old script stage asked a model to write a video about a topic and then
checked the prose; this one asks first whether there is a story worth telling, and why that one.

    subject + curated facts + channel history
      → EDITORIAL DIRECTOR   ten candidate angles ranked with the channel's THEORY OF VALUE, the obvious
                             and the unverifiable killed; an independent editor picks and explains
      → EDITORIAL BRIEF      angle, question, tension, takeaway, facts in/out, misconception, comparison
      → NARRATIVE DESIGN     three arcs of the SAME story, every beat with its fact and what is seen;
                             the editor picks one
      → WRITERS EN + IT      two independent proses from the same beats — never a translation
      → HYGIENE              the mechanical nets (facts against the sheet, lint, glossary, names,
                             competitors, morbid lexicon, budgets) — they catch errors, not quality
      → EDITORIAL REVIEW     a different model diagnoses with the POETICS and the annotated BENCHMARK:
                             ordinal rubric, weak sentences with reasons, publish / rewrite / reject story
      → REWRITE + REVIEW 2   the second review compares v1 and v2: better, or merely compliant?
      → APPROVED, or the story is rejected and the director tries another angle; then the stage fails.

The documents that define "magazine level" live in editorial/ (theory_of_value.md, poetics.md,
benchmark.json) and are loaded into the prompts: the judge's competence is built into the system, not
assumed from the model. The editor can be a different provider (AVP_EDITOR_URL / _MODEL / _API_KEY or
script.editor_*); until then DeepSeek plays both roles with separate prompts — a stated compromise.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import requests

from . import brief as brief_mod
from . import factcheck, italian, polish
from .llm import competitor_mentions, copied_exemplar, morbid_in_script, names_subject
from .models import Script, Segment, dedupe_segments
from .subtitles import _backend, dropped_names, italian_lint

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
WORDS_PER_SECOND = 2.5           # Kokoro, measured: 109 words → 44 s
CTA_SECONDS = 9                  # spoken CTA + silent tail, worst case (see stages.stage_script)
SUBTITLE_READING_FACTOR = 0.92   # the voice waits for the Italian reader (stages.reading_pause)
MIN_BEATS, MAX_BEATS = 5, 7
MIN_WORDS_PER_BEAT, MAX_WORDS_PER_BEAT = 9, 32
RUN_ON_WORDS = 26
_NUMBER_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "dozen", "hundred",
                 "thousand", "million", "billion", "trillion")
_IMPERIAL = re.compile(r"\b(inch(es)?|miles?|feet|foot|yards?|pounds?|fahrenheit)\b|°F", re.I)


class EditorialError(RuntimeError):
    """The engine could not bring a story to the standard — the video is not made."""


class StoryRejected(EditorialError):
    """The editor rejected the STORY (not the prose): the director must find another angle."""

    def __init__(self, angle_id: int | None, why: str):
        super().__init__(f"story rejected: {why}")
        self.angle_id, self.why = angle_id, why


# ----------------------------------------------------------------------------- standards
def standard(name: str) -> str:
    """A document from editorial/ — the theory of value, the poetics, the benchmark. Empty if absent."""
    for base in (ROOT / "editorial", Path.cwd() / "editorial"):
        p = base / name
        if p.exists():
            try:
                return p.read_text().strip()
            except OSError:
                continue
    return ""


def benchmark_text() -> str:
    raw = standard("benchmark.json")
    if not raw:
        return "No annotated benchmark available yet: judge against the poetics alone."
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, indent=1)
    except ValueError:
        return raw


# ----------------------------------------------------------------------------- the two models
def _endpoint(cfg, editor: bool) -> tuple[str, str, str]:
    """(key, url, model). The EDITOR may be another provider: env AVP_EDITOR_* or script.editor_*;
    it falls back to the writer's model so the engine runs everywhere, and says so in the report."""
    sc = cfg.script
    writer_model = (str(getattr(sc, "brief_model", "") or "").strip()
                    or str(getattr(sc, "factcheck_model", "deepseek-chat") or "deepseek-chat"))
    if not editor:
        return factcheck._api_key(cfg), factcheck.DEEPSEEK_URL, writer_model
    key = (os.getenv("AVP_EDITOR_API_KEY", "").strip() or str(getattr(sc, "editor_api_key", "") or "").strip()
           or factcheck._api_key(cfg))
    url = (os.getenv("AVP_EDITOR_URL", "").strip() or str(getattr(sc, "editor_url", "") or "").strip()
           or factcheck.DEEPSEEK_URL)
    model = (os.getenv("AVP_EDITOR_MODEL", "").strip() or str(getattr(sc, "editor_model", "") or "").strip()
             or writer_model)
    return key, url, model


def independent_editor(cfg) -> bool:
    """True when the editor is a different model or provider than the writer."""
    _, wu, wm = _endpoint(cfg, False)
    _, eu, em = _endpoint(cfg, True)
    return (eu, em) != (wu, wm)


def _call(cfg, system: str, user: str, editor: bool = False, temperature: float = 0.2, max_tokens: int = 3000) -> dict:
    key, url, model = _endpoint(cfg, editor)
    if not key:
        raise EditorialError("no API key for the editorial engine (DEEPSEEK_API_KEY or script.factcheck_key; "
                             "AVP_EDITOR_API_KEY for an independent editor)")
    r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
                            "response_format": {"type": "json_object"},
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                      timeout=(10, 240))
    if r.status_code >= 400:
        raise EditorialError(f"editorial model HTTP {r.status_code}: {(r.text or '')[:300]}")
    return factcheck._extract_json(r.json()["choices"][0]["message"]["content"])


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=1)


# ----------------------------------------------------------------------------- prompts
DIRECTOR_SYSTEM = """You are the editorial director of a serious astronomy magazine planning 48-second videos for an
educated audience that already knows basic astronomy. You do not write prose. You decide whether a story worth
telling exists in a subject, and which one — and you can say that none does today.

THE CHANNEL'S THEORY OF VALUE (apply it literally, quote its criteria when you rank or kill):
{theory}

Return STRICT JSON only."""

DIRECTOR_USER = """SUBJECT: {topic}

FACT BASE (the only facts that exist for this video; an angle whose strongest fact is not here is dead):
{facts}

RECENT CHANNEL HISTORY (topic | title — do not retell these):
{history}
{avoid}
Generate 10 genuinely different editorial angles on the subject. For each return: "id" (1-10), "angle" (one
sentence), "central_question", "value_sources" (which of the theory's sources it draws on, by name),
"strongest_fact" (quoted or closely paraphrased FROM THE FACT BASE), "visual_idea" (what the viewer would see),
"risk" (why it might fail), "killed" (true/false) and "kill_reason" (which kill criterion, or null).
Then rank the surviving angles best to worst as "ranking" (list of ids) and write "comparison": two or three
sentences on why the top angle beats the runners-up. Return {{"angles": [...], "ranking": [...], "comparison": "..."}}"""

SELECT_SYSTEM = """You are an independent senior editor at a serious astronomy magazine. You judge story IDEAS, not
prose, with the channel's theory of value:
{theory}
Correctness alone is nothing; surprise without a verifiable fact is worse than nothing. Return STRICT JSON only."""

SELECT_USER = """SUBJECT: {topic}
CANDIDATE ANGLES (the director's ranking and comparison are opinions, not verdicts):
{candidates}

Choose the strongest angle. Verifiability comes first: an angle whose strongest_fact does not appear in the fact
base cannot win. Then: real tension or reversal, a fact the audience does not already hold, a number or mechanism
that makes it tangible, tellable in 48 seconds with generated images (no people), distance from the channel's
recent history. Return {{"winner": <id>, "why_this_story": "compare it with the two runners-up in three sentences",
"rejected": [{{"id": <id>, "reason": "..."}}]}}"""

BRIEF_SYSTEM = """You are the editorial desk of a serious astronomy magazine. Turn the chosen angle into an EDITORIAL
BRIEF — not a script. Make explicit what the story is, what it is not, and what the viewer should understand.
Use only the fact base. Return STRICT JSON only."""

BRIEF_USER = """SUBJECT: {topic}
CHOSEN ANGLE:
{winner}
WHY THIS STORY (the editor's comparison):
{why}
FACT BASE:
{facts}

Return {{"editorial_angle": "...", "central_question": "...", "central_tension": "...", "audience_takeaway": "...",
"key_facts": ["..."], "optional_facts": ["..."], "excluded_facts": ["true facts that would distract here"],
"misconception": "... or null", "quantitative_comparison": "... or null", "opening_type": "counterintuitive fact |
question | comparison | scene", "closing_type": "answer | honest open question | perspective | change of view",
"why_this_story": "...", "what_not_to_do": ["..."]}}"""

NARRATIVE_SYSTEM = """You are a narrative designer for a serious astronomy magazine's short videos. From the brief,
design THREE genuinely different narrative arcs for the SAME story (for example: from the number, from the open
question, from the misconception). Each arc has {n_beats} beats. A beat carries only: the fact it rests on, what the
viewer SEES while it is said (a concrete, generatable image: no people, no realistic historical scenes, no detailed
nebulae), and its role (opening | build | turn | peak | close). No prose. The arc must escalate and end by resolving
or honestly opening the central question. Return STRICT JSON only."""

NARRATIVE_USER = """SUBJECT: {topic}
BRIEF:
{brief}
FACT BASE:
{facts}

Return {{"arcs": [{{"id": 1, "thesis": "...", "beats": [{{"beat": 1, "fact": "...", "visual": "...", "role": "opening"}}],
"why": "..."}}, {{"id": 2, ...}}, {{"id": 3, ...}}]}}"""

ARC_SELECT_SYSTEM = """You are a senior narrative editor at a serious astronomy magazine. Choose the strongest arc for
a {seconds}-second video with generated images. Judge: coherence and escalation, distinct beats, scientific
integrity, visual feasibility (no people, no realistic historical scenes, no detailed nebulae), an ending that
belongs to the story, distance from the channel's recent videos. Return STRICT JSON only."""

ARC_SELECT_USER = """SUBJECT: {topic}
BRIEF:
{brief}
ARCS:
{arcs}
Return {{"winner": <id>, "why": "three sentences comparing the arcs"}}"""

WRITE_SYSTEM = """You are a staff writer at a serious astronomy magazine, writing the spoken text of a short video for an
audience that knows basic astronomy. You realise an editorial brief and a chosen arc — beat by beat, one segment
per beat, in order. This is not school science, not generic YouTube copy, not advertising.

THE POETICS (the qualities you optimise, with weak/strong pairs; the forbidden tendencies are absolute):
{poetics}

Contract: every sentence informs, contextualises or advances the story; use the brief's key facts, at most its
optional facts, never its excluded facts; never invent a mechanism, number, date, status, superlative or
attribution; never repeat a fact; numbers as digits; no metaphor unless the next sentence cashes it in; at most one
in the whole text. Never name another app or product. Return STRICT JSON only."""

WRITE_USER = """LANGUAGE: {language}
SUBJECT: {topic}
BRIEF:
{brief}
CHOSEN ARC (one segment per beat, same order, {n_beats} segments):
{arc}
FACT BASE:
{facts}

Budget of the medium (boundaries, not style): about {words} spoken words in total — never more than {hi}, never
fewer than {lo} — so about {per} words per segment ({min_w}-{max_w}), in one or two sentences; a viewer reads a
subtitle card of this text, so a sentence never runs past {run_on} words. Count your words before answering.
Segment 1 does not open with a number. {language_note}

Return {{"title": "...", "segments": [{{"narration": "...", "visual": "the beat's visual, refined", "keywords": ["..."]}}],
"bridge_kind": "shoot | principle | none", "cta_bridge": "one honest sentence (at most 15 words) that links THIS story
to looking at or photographing the sky tonight, in {language}; empty string with bridge_kind none when no honest link
exists — never force one"}}
Write independently in {language} from the facts and the beats. Do not translate any other language."""

LANG_NOTES = {
    "English": "Spoken register: concrete nouns, active verbs, a rhythm a voice can carry.",
    "Italian": ("Italiano da redazione scientifica, parlato: soggetto esplicito in ogni frase, mai un ordine dove si "
                "afferma, passato prossimo, niente calchi (mai 'mai' per ever, niente 'realizzare' per capire), "
                "terminologia esatta (obiettivo, messa a fuoco, cursore, tacca, bande di polvere, chilometri), "
                "numeri in cifre in formato italiano (8,87 millimetri; 26.000 anni)."),
}

REVISE_NOTE = """

You are REVISING after an editorial diagnosis (below). Fix every diagnosed weakness without flattening the voice,
changing the chosen story, or padding. Keep the beat count. The second review will ask whether the text became
BETTER, not whether it complied.

DIAGNOSIS:
{diagnosis}"""

TIGHTEN_SYSTEM = "You are a copy editor cutting a spoken script to length. You remove words, never facts. Return STRICT JSON only."
TIGHTEN_USER = """This {language} script does not fit the medium:
{reasons}

Cut it to fit WITHOUT losing a beat, a fact, a number or a name: remove asides, doubled adjectives, repeated
context, throat-clearing; split any sentence longer than {run_on} words. Same number of segments, same order,
same title, same cta_bridge and bridge_kind, visuals untouched. {limits}

SCRIPT:
{script}

Return the same JSON shape: {{"title": "...", "segments": [{{"narration": "...", "visual": "...", "keywords": ["..."]}}],
"bridge_kind": "...", "cta_bridge": "..."}}"""

REVIEW_SYSTEM = """You are a ruthless independent editor at a serious astronomy magazine. You DIAGNOSE a script; you do
not rewrite it. A script can be factually correct and editorially weak — correctness and quality are two dimensions.
You judge with the channel's poetics and its annotated benchmark:

POETICS:
{poetics}

BENCHMARK (gold texts and their annotations, near-misses and mediocre versions):
{benchmark}

Rate each dimension with one word — weak | solid | strong | exceptional — and ai_smell with none | mild | strong |
grave. Quote every sentence that prevents a stronger result, with a precise reason (vague, textbook, unearned
adjective, low density, wrong emphasis, betrays the brief's angle, cliché, order where a statement was meant,
literal phrasing in {language}). Flag fact risks, repetitions, promotional turns and misleading visuals. Decide:
"publish" when nothing is weak and the idea is at least strong; "rewrite" when the idea holds but the prose does not;
"reject_story" when the idea or the originality is weak — no rewrite saves a story that does not hold.
Return STRICT JSON only."""

REVIEW_USER = """LANGUAGE OF THE SCRIPT: {language}
SUBJECT: {topic}
BRIEF:
{brief}
SCRIPT (segments in order; the cta_bridge is the closing sentence before the app card):
{script}

Return {{"dimensions": {{"idea": "...", "specificity": "...", "density": "...", "originality": "...", "narration": "...",
"language": "...", "ai_smell": "..."}}, "weak_sentences": [{{"segment": 1, "quote": "...", "reason": "..."}}],
"fact_risks": ["..."], "decision": "publish | rewrite | reject_story", "summary": "two sentences"}}"""

REVIEW2_USER = """LANGUAGE OF THE SCRIPT: {language}
SUBJECT: {topic}
BRIEF:
{brief}
VERSION 1 (diagnosed):
{v1}
YOUR DIAGNOSIS OF VERSION 1:
{diagnosis}
VERSION 2 (rewritten with the diagnosis in hand):
{v2}

Rate VERSION 2 with the same rubric. Then answer the question that matters: did the text become a BETTER text — idea,
rhythm, density, voice — or did it merely comply with the corrections (adjectives removed, a number added, still no
idea and no rhythm)? Return {{"dimensions": {{...}}, "weak_sentences": [...], "fact_risks": [...],
"decision": "publish | rewrite | reject_story", "improved": true|false, "improvement_note": "one sentence",
"summary": "two sentences"}}"""


# ----------------------------------------------------------------------------- helpers
def word_budget(cfg, n_beats: int) -> tuple[int, int, int]:
    """(target, low, high) spoken words for the content beats, from the video target and the medium."""
    content = int(getattr(cfg.script, "target_seconds", 48) or 48) - (CTA_SECONDS if cfg.funnel.enabled else 0)
    sub = getattr(cfg.script, "subtitle_language", None)
    if sub and sub != getattr(cfg.script, "language", "en"):
        content = int(round(content * SUBTITLE_READING_FACTOR))
    words = int(round(max(20, content) * WORDS_PER_SECOND))
    return words, int(round(words * 0.85)), int(round(words * 1.15))


def _words(text: str) -> int:
    return len((text or "").split())


def _history(cfg) -> str:
    root = Path(cfg.paths.projects_dir).expanduser()
    rows: list[str] = []
    for man in sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)[-60:]:
        try:
            d = json.loads(man.read_text())
            if d.get("topic") or d.get("title"):
                rows.append(f"- {d.get('topic', '')} | {d.get('title', '')}")
        except Exception:  # noqa: BLE001
            continue
    return "\n".join(rows) or "(none)"


def _to_script(data: dict, topic: str, target: int) -> Script:
    segs: list[Segment] = []
    for i, row in enumerate(data.get("segments") or [], 1):
        if not isinstance(row, dict):
            continue
        narration = " ".join(str(row.get("narration", "")).split())
        if not narration:
            continue
        segs.append(Segment(index=i, narration=narration, visual=" ".join(str(row.get("visual", "")).split()),
                            keywords=[str(k).strip() for k in (row.get("keywords") or []) if str(k).strip()]))
    segs = dedupe_segments(segs)
    if not MIN_BEATS <= len(segs) <= MAX_BEATS:
        raise EditorialError(f"the writer returned {len(segs)} segments, the arc needs {MIN_BEATS}-{MAX_BEATS}")
    kind = str(data.get("bridge_kind") or "none").strip().lower()
    if kind not in ("shoot", "principle", "none"):
        kind = "none"
    return Script(title=" ".join(str(data.get("title") or topic).split()), segments=segs, target_seconds=target,
                  topic=topic, cta_bridge=" ".join(str(data.get("cta_bridge") or "").split()), bridge_kind=kind)


def hygiene_en(script: Script, cfg, budget: tuple[int, int, int]) -> list[str]:
    """The mechanical nets on the English — errors, not quality. Each string is a reason for the writer."""
    reasons: list[str] = []
    content = [s for s in script.segments if s.kind != "cta"]
    total = sum(_words(s.narration) for s in content)
    words, lo, hi = budget
    if total < lo or total > hi:
        reasons.append(f"total {total} spoken words, the budget is {lo}-{hi} (about {words}): "
                       + ("cut" if total > hi else "add substance, not padding"))
    head = " ".join(content[0].narration.split()[:6]).lower() if content else ""
    if re.search(r"\d", head) or any(f" {w} " in f" {head} " for w in _NUMBER_WORDS):
        reasons.append("segment 1 opens on a number — a listener cannot hold a number cold; open on the thing")
    for s in content:
        n = _words(s.narration)
        if n < MIN_WORDS_PER_BEAT:
            reasons.append(f"segment {s.index} has {n} words (< {MIN_WORDS_PER_BEAT}): a flash, not a beat")
        if n > MAX_WORDS_PER_BEAT:
            reasons.append(f"segment {s.index} has {n} words (> {MAX_WORDS_PER_BEAT}): the card cannot be read in time")
        for sent in re.split(r"(?<=[.!?])\s+", s.narration):
            if _words(sent) > RUN_ON_WORDS:
                reasons.append(f"segment {s.index}: run-on sentence ({_words(sent)} words) — {sent[:50]!r}")
        if _IMPERIAL.search(s.narration):
            reasons.append(f"segment {s.index}: imperial units — the channel speaks metric")
    if script.topic and content and not names_subject(" ".join(s.narration for s in content[:2]), script.topic):
        reasons.append("the subject is not named in plain words by segment 2 — the viewer has no context")
    probe = {"title": script.title, "segments": [{"narration": s.narration} for s in content], "cta_bridge": script.cta_bridge}
    if (mw := morbid_in_script(probe)):
        reasons.append(f"morbid word {mw!r} — banned on this channel")
    if (ph := copied_exemplar(probe)):
        reasons.append(f"copied exemplar phrase {ph!r}")
    for text in [script.title, script.cta_bridge] + [s.narration for s in content]:
        if (hit := competitor_mentions(text)):
            reasons.append(f"names another app ({hit[0]}) — never; say 'a camera app with manual focus'")
            break
    if _words(script.cta_bridge) > 22:
        reasons.append(f"cta_bridge has {_words(script.cta_bridge)} words (> 22)")
    return reasons


def hygiene_it(en: Script, it: Script) -> list[str]:
    """The mechanical nets on the Italian, beat by beat against the English of the same beat."""
    reasons: list[str] = []
    ec = [s for s in en.segments if s.kind != "cta"]
    ic = [s for s in it.segments if s.kind != "cta"]
    if len(ec) != len(ic):
        return [f"the Italian has {len(ic)} segments, the English {len(ec)}: same beats, same count"]
    for e, i in zip(ec, ic):
        for p in italian_lint(i.narration):
            reasons.append(f"segmento {i.index}: italiano scorretto — {p}")
        for w in italian.bad_sense(i.narration):
            reasons.append(f"segmento {i.index}: termine nel senso sbagliato {w!r} (tacca non scatto; obiettivo non lente; niente miglia; bande di polvere)")
        if (hit := competitor_mentions(i.narration)):
            reasons.append(f"segmento {i.index}: nomina un'altra app ({hit[0]}) — mai")
        lost = dropped_names(e.narration, i.narration)
        if lost:
            reasons.append(f"segmento {i.index}: manca il nome che la battuta inglese ha — {', '.join(lost)}")
        ratio = len(i.narration) / max(1, len(e.narration))
        if ratio > 1.4:
            reasons.append(f"segmento {i.index}: {ratio:.1f}× i caratteri dell'inglese — il lettore non arriva in fondo; stessa battuta, più asciutta")
        elif ratio < 0.7:
            reasons.append(f"segmento {i.index}: {ratio:.1f}× i caratteri dell'inglese — manca contenuto della battuta")
    for p in italian_lint(it.cta_bridge or ""):
        reasons.append(f"ponte finale: italiano scorretto — {p}")
    if (hit := competitor_mentions(it.cta_bridge or "")):
        reasons.append(f"ponte finale: nomina un'altra app ({hit[0]})")
    return reasons


def _ok(word: str, floor: tuple[str, ...]) -> bool:
    return str(word or "").strip().lower() in floor


def publishable(review: dict) -> bool:
    d = review.get("dimensions") or {}
    return (str(review.get("decision", "")).lower() == "publish"
            and all(_ok(d.get(k), ("solid", "strong", "exceptional")) for k in ("specificity", "density", "narration", "language"))
            and _ok(d.get("idea"), ("strong", "exceptional"))
            and _ok(d.get("originality"), ("solid", "strong", "exceptional"))
            and _ok(d.get("ai_smell"), ("none", "mild")))


def story_rejected(review: dict) -> bool:
    d = review.get("dimensions") or {}
    return (str(review.get("decision", "")).lower() == "reject_story"
            or _ok(d.get("idea"), ("weak",)) or _ok(d.get("originality"), ("weak",)))


# ----------------------------------------------------------------------------- the pass
def _write(cfg, language: str, topic: str, brief: dict, arc: dict, facts: str, budget: tuple[int, int, int],
           n_beats: int, poetics: str, target: int, diagnosis: dict | None = None, notes: list[str] | None = None,
           temperature: float = 0.55) -> Script:
    words, lo, hi = budget
    system = WRITE_SYSTEM.format(poetics=poetics)
    user = WRITE_USER.format(language=language, topic=topic, brief=_j(brief), arc=_j(arc), facts=facts, words=words,
                             lo=lo, hi=hi, per=max(MIN_WORDS_PER_BEAT, words // max(1, n_beats)), n_beats=n_beats,
                             min_w=MIN_WORDS_PER_BEAT, max_w=MAX_WORDS_PER_BEAT,
                             run_on=RUN_ON_WORDS, language_note=LANG_NOTES.get(language, ""))
    if diagnosis:
        user += REVISE_NOTE.format(diagnosis=_j(diagnosis))
    if notes:
        user += "\n\nHYGIENE NOTES from the previous attempt — every one must be resolved:\n- " + "\n- ".join(notes)
    return _to_script(_call(cfg, system, user, editor=False, temperature=temperature, max_tokens=3200), topic, target)


_LENGTH_MARKERS = ("spoken words", "run-on", "words (>", "caratteri dell'inglese")


def _length_only(reasons: list[str]) -> bool:
    return bool(reasons) and all(any(m in r for m in _LENGTH_MARKERS) for r in reasons)


def _tighten(cfg, language: str, script: Script, reasons: list[str], budget: tuple[int, int, int], target: int) -> Script:
    words, lo, hi = budget
    limits = (f"Total spoken words between {lo} and {hi} (about {words})." if language == "English"
              else "Per battuta, tra il 90% e il 125% dei caratteri della battuta inglese.")
    data = {"title": script.title, "segments": [{"narration": x.narration, "visual": x.visual, "keywords": x.keywords}
                                                for x in script.segments if x.kind != "cta"],
            "bridge_kind": script.bridge_kind, "cta_bridge": script.cta_bridge}
    out = _to_script(_call(cfg, TIGHTEN_SYSTEM, TIGHTEN_USER.format(language=language, reasons="- " + "\n- ".join(reasons),
                                                                    run_on=RUN_ON_WORDS, limits=limits, script=_j(data)),
                           editor=False, temperature=0.2, max_tokens=3200), script.topic, target)
    for a, b in zip([x for x in out.segments], [x for x in script.segments if x.kind != "cta"]):
        if not a.visual:
            a.visual, a.keywords = b.visual, list(b.keywords)
    return out


def _fit(cfg, language: str, topic: str, brief: dict, arc: dict, facts: str, budget: tuple[int, int, int], n_beats: int,
         poetics: str, target: int, script: Script, check, drafts: list) -> Script:
    """Two corrective passes at most: a rewrite with the notes when the problems are of substance, a cut to
    length when they are only of length; then the nets are final."""
    for attempt in (1, 2):
        reasons = check(script)
        drafts.append({"language": language, "attempt": attempt, "reasons": reasons, "words": sum(_words(x.narration) for x in script.segments),
                       "segments": [x.narration for x in script.segments]})
        if not reasons:
            return script
        log.info("Hygiene %s (%d): %s", language, attempt, "; ".join(reasons))
        if _length_only(reasons):
            script = _tighten(cfg, language, script, reasons, budget, target)
        else:
            script = _write(cfg, language, topic, brief, arc, facts, budget, n_beats, poetics, target, notes=reasons, temperature=0.4)
    reasons = check(script)
    drafts.append({"language": language, "attempt": 3, "reasons": reasons, "words": sum(_words(x.narration) for x in script.segments),
                   "segments": [x.narration for x in script.segments]})
    if reasons:
        raise EditorialError(f"{language} script still fails the hygiene nets: " + "; ".join(reasons))
    return script


def _review(cfg, language: str, topic: str, brief: dict, script: Script, poetics: str, benchmark: str) -> dict:
    return _call(cfg, REVIEW_SYSTEM.format(poetics=poetics, benchmark=benchmark, language=language),
                 REVIEW_USER.format(language=language, topic=topic, brief=_j(brief), script=_j(_script_view(script))),
                 editor=True, temperature=0.05, max_tokens=2800)


def _review2(cfg, language: str, topic: str, brief: dict, v1: Script, diagnosis: dict, v2: Script, poetics: str,
             benchmark: str) -> dict:
    return _call(cfg, REVIEW_SYSTEM.format(poetics=poetics, benchmark=benchmark, language=language),
                 REVIEW2_USER.format(language=language, topic=topic, brief=_j(brief), v1=_j(_script_view(v1)),
                                     diagnosis=_j(diagnosis), v2=_j(_script_view(v2))),
                 editor=True, temperature=0.05, max_tokens=2800)


def _script_view(s: Script) -> dict:
    return {"title": s.title, "segments": [{"index": x.index, "narration": x.narration, "visual": x.visual}
                                           for x in s.segments if x.kind != "cta"],
            "cta_bridge": s.cta_bridge, "bridge_kind": s.bridge_kind}


def _fix_facts_and_proof(cfg, en: Script, it: Script, facts: str, out_dir: Path) -> list[str]:
    """Hygiene corrections, applied silently: the sheet is the ground truth for both languages; a cold
    copy editor fixes grammar only; the Italian proofreader sees the English for the sense. Then the
    back-translation compare returns the beats whose Italian does not say what the English says."""
    notes: list[str] = []
    try:
        rep = factcheck.run(en, cfg, out_dir=out_dir, facts=facts)
        if rep.findings:
            notes.append(f"fact-check EN: {len(rep.findings)} finding(s)")
    except Exception as e:  # noqa: BLE001
        log.warning("fact-check skipped (%s)", e)
    try:
        polish.proofread(en, cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("English proofread skipped (%s)", e)
    texts = {s.index: s.narration for s in it.segments if s.kind != "cta"}
    en_by = {s.index: s.narration for s in en.segments if s.kind != "cta"}
    try:
        backend, _ = _backend(cfg)
        n = italian._proof(texts, en_by, backend)
        if n:
            notes.append(f"correttore IT: {n} schede corrette")
        notes += italian._factcheck(texts, en, facts, cfg)
        for s in it.segments:
            if s.index in texts:
                s.narration = texts[s.index]
        diffs = italian._meaning(texts, en_by, backend)
        for i, why in diffs.items():
            notes.append(f"segmento {i}: {why}")
    except Exception as e:  # noqa: BLE001
        log.warning("Italian proof/compare skipped (%s)", e)
    return notes


def _pass(cfg, topic: str, project, facts: str, history: str, attempt: int, avoid: list[str]) -> tuple[Script, Script, dict, dict]:
    theory = standard("theory_of_value.md") or "Prefer tension, reversal, tangible scale, mechanism, honest open questions, human traces, a change of view; kill the obvious, the unverifiable, the unvisualisable, the recently told."
    poetics = standard("poetics.md") or "Precision, information density, depth, rhythm, naturalness, voice; no declared enthusiasm, no riddles, no unearned adjectives, no textbook explanations, no promotional closers."
    benchmark = benchmark_text()
    target = int(getattr(cfg.script, "target_seconds", 48) or 48)
    root: Path = project.root

    # 1 · director
    avoid_note = ("\nANGLES ALREADY REJECTED BY THE EDITOR THIS RUN (do not propose them again):\n- " + "\n- ".join(avoid) + "\n") if avoid else ""
    director = _call(cfg, DIRECTOR_SYSTEM.format(theory=theory),
                     DIRECTOR_USER.format(topic=topic, facts=facts, history=history, avoid=avoid_note),
                     editor=False, temperature=0.55, max_tokens=3600)
    angles = [a for a in (director.get("angles") or []) if isinstance(a, dict)]
    alive = [a for a in angles if not a.get("killed")]
    if len(alive) < 2:
        raise EditorialError(f"the director found no story worth telling in {topic!r} today "
                             f"({len(angles)} angles, {len(alive)} alive)")
    # 2 · independent selection
    selection = _call(cfg, SELECT_SYSTEM.format(theory=theory),
                      SELECT_USER.format(topic=topic, candidates=_j({"angles": alive, "director_ranking": director.get("ranking"),
                                                                     "director_comparison": director.get("comparison")})),
                      editor=True, temperature=0.1, max_tokens=1800)
    try:
        wid = int(selection.get("winner"))
    except (TypeError, ValueError):
        wid = int(alive[0].get("id", 1))
    winner = next((a for a in alive if int(a.get("id", -1)) == wid), alive[0])
    (root / "editorial_director.json").write_text(_j({"attempt": attempt, "avoid": avoid, "angles": angles,
                                                       "director_ranking": director.get("ranking"),
                                                       "director_comparison": director.get("comparison"),
                                                       "selection": selection}))
    # 3 · brief
    brief = _call(cfg, BRIEF_SYSTEM, BRIEF_USER.format(topic=topic, winner=_j(winner), why=selection.get("why_this_story", ""), facts=facts),
                  editor=False, temperature=0.1, max_tokens=2600)
    brief.setdefault("why_this_story", selection.get("why_this_story", ""))
    (root / "editorial_brief.json").write_text(_j(brief))
    # 4 · narrative design + independent arc selection
    n_beats = 6
    arcs_data = _call(cfg, NARRATIVE_SYSTEM.format(n_beats=n_beats), NARRATIVE_USER.format(topic=topic, brief=_j(brief), facts=facts),
                      editor=False, temperature=0.5, max_tokens=3600)
    arcs = [a for a in (arcs_data.get("arcs") or []) if isinstance(a, dict) and a.get("beats")]
    if len(arcs) < 2:
        raise EditorialError("the narrative designer returned fewer than two arcs")
    arc_pick = _call(cfg, ARC_SELECT_SYSTEM.format(seconds=target), ARC_SELECT_USER.format(topic=topic, brief=_j(brief), arcs=_j(arcs)),
                     editor=True, temperature=0.1, max_tokens=900)
    try:
        aid = int(arc_pick.get("winner"))
    except (TypeError, ValueError):
        aid = int(arcs[0].get("id", 1))
    arc = next((a for a in arcs if int(a.get("id", -1)) == aid), arcs[0])
    n_beats = max(MIN_BEATS, min(MAX_BEATS, len(arc.get("beats") or [])))
    (root / "narrative_arcs.json").write_text(_j({"arcs": arcs, "selection": arc_pick}))
    # 5 · two writers, then the hygiene nets (rewrite with notes, or cut to length; two passes at most)
    budget = word_budget(cfg, n_beats)
    drafts: list[dict] = []
    try:
        en = _write(cfg, "English", topic, brief, arc, facts, budget, n_beats, poetics, target)
        en = _fit(cfg, "English", topic, brief, arc, facts, budget, n_beats, poetics, target, en,
                  lambda s: hygiene_en(s, cfg, budget), drafts)
        it = _write(cfg, "Italian", topic, brief, arc, facts, budget, n_beats, poetics, target)
        it = _fit(cfg, "Italian", topic, brief, arc, facts, budget, n_beats, poetics, target, it,
                  lambda s: hygiene_it(en, s), drafts)
    finally:
        (root / "editorial_drafts.json").write_text(_j(drafts))
    # 6 · facts and proof (corrections), back-translation compare (one Italian pass if beats disagree)
    notes = _fix_facts_and_proof(cfg, en, it, facts, root)
    diffs = [n for n in notes if "senso diverso" in n]
    if diffs:
        log.info("Compare IT/EN: %s", "; ".join(diffs))
        it = _write(cfg, "Italian", topic, brief, arc, facts, budget, n_beats, poetics, target, notes=diffs + hygiene_it(en, it), temperature=0.4)
        if hygiene_it(en, it):
            it = _tighten(cfg, "Italian", it, hygiene_it(en, it), budget, target) if _length_only(hygiene_it(en, it)) else it
        if hygiene_it(en, it):
            raise EditorialError("Italian script fails the hygiene nets after the compare pass: " + "; ".join(hygiene_it(en, it)))
    # 7 · editorial review, rewrite, second review
    reviews: dict = {}
    for lang, s in (("English", en), ("Italian", it)):
        r1 = _review(cfg, lang, topic, brief, s, poetics, benchmark)
        suffix = "en" if lang == "English" else "it"
        (root / f"editorial_review_{suffix}_v1.json").write_text(_j(r1))
        reviews[f"{suffix}_v1"] = r1
        if story_rejected(r1):
            raise StoryRejected(wid, f"{lang}: {r1.get('summary', '')}")
        if publishable(r1):
            continue
        v2 = _write(cfg, lang, topic, brief, arc, facts, budget, n_beats, poetics, target, diagnosis=r1, temperature=0.35)
        nets = hygiene_en(v2, cfg, budget) if lang == "English" else hygiene_it(en if lang == "Italian" else v2, v2)
        if nets:
            raise EditorialError(f"{lang} rewrite fails the hygiene nets: " + "; ".join(nets))
        r2 = _review2(cfg, lang, topic, brief, s, r1, v2, poetics, benchmark)
        (root / f"editorial_review_{suffix}_v2.json").write_text(_j(r2))
        reviews[f"{suffix}_v2"] = r2
        if r2.get("improved") is False or story_rejected(r2):
            raise StoryRejected(wid, f"{lang}: the rewrite merely complied — {r2.get('improvement_note') or r2.get('summary', '')}")
        if not publishable(r2):
            raise EditorialError(f"{lang} script below the editorial standard after the rewrite: {r2.get('summary', '')}")
        s.title, s.segments, s.cta_bridge, s.bridge_kind = v2.title, v2.segments, v2.cta_bridge, v2.bridge_kind
        if lang == "English":
            en = s
        else:
            it = s
    # 8 · one structure for the pipeline: the English script carries the Italian card of every beat
    for a, b in zip([x for x in en.segments if x.kind != "cta"], [x for x in it.segments if x.kind != "cta"]):
        a.italian = b.narration
        b.visual, b.keywords = a.visual, list(a.keywords)
    if cfg.funnel.enabled:
        from . import stages
        en.segments.append(Segment(index=len(en.segments) + 1, narration=stages._cta_narration(en, cfg),
                                   visual="App endcard", keywords=[], kind="cta",
                                   italian=(it.cta_bridge if en.cta_bridge and it.cta_bridge else "")))
    evidence = {"winner": winner, "why_this_story": selection.get("why_this_story", ""), "arc": arc.get("thesis", ""),
                "arc_why": arc_pick.get("why", ""), "reviews": reviews, "hygiene_notes": notes}
    return en, it, brief, evidence


def run(project, cfg, topic: str | None) -> Script:
    """The whole editorial machine for one video. Writes script.json / script.md (English with the Italian
    card of every beat), italian_script.json and the evidence files; raises when no story reaches the
    standard — no video is better than a mediocre one."""
    topic = (topic or str(project.manifest.data.get("topic") or "")).strip()
    if not topic:
        raise ValueError("No topic given and no existing one. Pass --topic.")
    facts = brief_mod.build(topic, cfg, out_dir=project.root)
    if not facts:
        raise EditorialError("no audited fact sheet — the engine refuses to invent facts (set script.brief and a key)")
    history = _history(cfg)
    avoid: list[str] = []
    last: Exception | None = None
    for attempt in (1, 2):
        try:
            en, it, brief, evidence = _pass(cfg, topic, project, facts, history, attempt, avoid)
        except StoryRejected as e:
            last = e
            log.warning("Attempt %d: %s — the director looks for another angle", attempt, e)
            avoid.append(f"angle id {e.angle_id}: {e.why[:160]}")
            continue
        from . import stages
        project.script_json.write_text(stages._json(en.to_dict()))
        (project.root / "italian_script.json").write_text(stages._json(it.to_dict()))
        stages.emit_script_md(en, project.script_md)
        _, _, editor_model = _endpoint(cfg, True)
        report = {"version": 3, "autonomous": True, "attempt": attempt, "topic": topic, "title": en.title,
                  "editor_model": editor_model, "independent_editor": independent_editor(cfg), **evidence,
                  "brief": brief}
        (project.root / "editorial_report.json").write_text(_j(report))
        project.manifest.data["topic"] = topic
        project.manifest.data["title"] = en.title
        project.manifest.data.setdefault("editorial", {}).update(
            {"version": 3, "attempt": attempt, "editor_model": editor_model, "independent_editor": independent_editor(cfg),
             "angle": (evidence.get("winner") or {}).get("angle", "")})
        project.manifest.mark("script", "done", title=en.title, segments=len(en.segments))
        log.info("Editorial script approved on attempt %d: %r — angle: %s", attempt, en.title,
                 (evidence.get("winner") or {}).get("angle", ""))
        return en
    raise last or EditorialError("editorial generation failed")
