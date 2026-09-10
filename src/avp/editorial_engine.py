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
from .subtitles import _backend, dropped_names, italian_lint, names_for_italian

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
_UNSPEAKABLE = re.compile(r"[×^]|\b10\^|\d[eE][+-]?\d|\d\*\d")


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
    """(key, url, model). The WRITER side (director's candidates, brief, arcs, writers, cuts, repairs) may be a
    LOCAL model on an OpenAI-compatible endpoint (Ollama: http://localhost:11434/v1/chat/completions) — env
    AVP_WRITER_* or script.writer_*; the EDITOR side (selection, arc choice, reviews) may be another provider
    (AVP_EDITOR_* or script.editor_*). Each falls back to DeepSeek so the engine runs everywhere, and the
    report says who did what."""
    sc = cfg.script
    deepseek_model = (str(getattr(sc, "brief_model", "") or "").strip()
                      or str(getattr(sc, "factcheck_model", "deepseek-chat") or "deepseek-chat"))
    if not editor:
        url = os.getenv("AVP_WRITER_URL", "").strip() or str(getattr(sc, "writer_url", "") or "").strip()
        model = os.getenv("AVP_WRITER_MODEL", "").strip() or str(getattr(sc, "writer_model", "") or "").strip()
        if url and model:
            key = (os.getenv("AVP_WRITER_API_KEY", "").strip() or str(getattr(sc, "writer_api_key", "") or "").strip()
                   or "local")
            return key, url, model
        return factcheck._api_key(cfg), factcheck.DEEPSEEK_URL, deepseek_model
    writer_model = deepseek_model
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
    local = "localhost" in url or "127.0.0.1" in url
    r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
                            "response_format": {"type": "json_object"},
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                      timeout=(10, 900 if local else 240))          # a local 20B writes 3000 tokens in a minute or two
    if r.status_code >= 400:
        raise EditorialError(f"editorial model HTTP {r.status_code}: {(r.text or '')[:300]}")
    msg = r.json()["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content.strip() and msg.get("reasoning"):        # a reasoning model that put everything in its thoughts
        content = msg["reasoning"]
    return factcheck._extract_json(content)


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
viewer SEES while it is said, and its role (opening | build | turn | peak | close). THE VISUAL IS A PHOTOGRAPH A
SPACECRAFT OR A TELESCOPE COULD TAKE, physically faithful to the fact base (the object's true colours, its
atmosphere, rings only if it has rings, its real surface and sky): the object itself, its surface, its sky, its
instruments. Never people, hands, clocks, treadmills, toys, tables, arrows, diagrams, or a metaphor staged as an
object; never a realistic historical scene; never a detailed nebula. No prose. EVERY BEAT CARRIES A DIFFERENT FACT: a
number, a comparison or a claim never appears in two beats — an arc that "intensifies the same comparison" is a
loop, not an arc (measured: it produced a script that repeated one figure four times). The arc must escalate and
end by resolving or honestly opening the central question. Return STRICT JSON only."""

NARRATIVE_USER = """SUBJECT: {topic}
BRIEF:
{brief}
FACT BASE:
{facts}

Return {{"arcs": [{{"id": 1, "thesis": "...", "beats": [{{"beat": 1, "fact": "...", "visual": "...", "role": "opening"}}],
"why": "..."}}, {{"id": 2, ...}}, {{"id": 3, ...}}]}}"""

ARC_SELECT_SYSTEM = """You are a senior narrative editor at a serious astronomy magazine. Choose the strongest arc for
a {seconds}-second video with generated images. Judge, in this order: DISTINCT BEATS — an arc whose beats repeat a
fact, a number or a comparison is rejected outright, however elegant; then coherence and escalation, scientific
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
attribution; never repeat a fact; every segment is a sentence with a subject and a verb — never a "Label:
definition" fragment ("Ring rain: charged molecules pulled…" is a glossary entry, not narration); numbers as
digits but always SPEAKABLE and holdable — a comparison the viewer can picture beats a figure they cannot ("an
Olympic pool every 30 minutes", not "15 quintillion kilograms"; "40 percent of Mimas" only if Mimas has been
introduced); no symbols a voice cannot read; no citation-speak ("according to a 2018 study in Icarus" → "a 2018
study" at most); no metaphor unless the next sentence cashes it in; at most one in the whole text. Never name
another app or product. Return STRICT JSON only."""

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

{limits_note}
Return {{"title": "...", "segments": [{{"narration": "...", "visual": "the beat's visual as a PHOTOGRAPH a spacecraft or a
telescope could take — the object, its surface, its sky, physically faithful; never an animation, diagram, split screen,
timeline, stick figure, arrow, clock, treadmill, person or staged comparison", "keywords": ["..."]}}],
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

Note on the cta_bridge: the channel ends every video on its own app's card — a format requirement, not a choice of
this script. Judge the bridge only for honesty and for being earned by the story (it may be empty with bridge_kind
"none"); never penalise its presence.

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
    return words, int(round(words * 0.85)), int(round(words * 1.05))   # +5%: gaps and reading pauses eat the rest


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


_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def arc_redundancy(arc: dict) -> list[str]:
    """Facts an arc states twice: a number or a near-identical fact in two beats. Such an arc makes the writer
    repeat itself and the reviewer blame the prose (seventh Venus trial: 6.5 km/h in three beats of six)."""
    import difflib
    beats = [str(b.get("fact", "")) for b in (arc.get("beats") or []) if isinstance(b, dict)]
    problems: list[str] = []
    seen: dict[str, int] = {}
    for i, f in enumerate(beats, 1):
        for n in set(_NUM.findall(f)):
            if len(n) < 2 or n in ("10", "100", "1000"):
                continue
            if n in seen and seen[n] != i:
                problems.append(f"the number {n} appears in beats {seen[n]} and {i}")
            seen.setdefault(n, i)
    low = [re.sub(r"[^a-z0-9 ]", " ", f.lower()) for f in beats]
    for i in range(len(low)):
        for j in range(i + 1, len(low)):
            if low[i] and low[j] and difflib.SequenceMatcher(None, low[i], low[j]).ratio() >= 0.6:
                problems.append(f"beats {i + 1} and {j + 1} say nearly the same thing")
    return problems


def _angles_from(director: dict) -> list[dict]:
    """The director's angles wherever the model put them: "angles", "candidates", or any list of dicts
    that carry an "angle" field; ids are filled in when missing."""
    if not isinstance(director, dict):
        return []
    cands = None
    for key in ("angles", "candidates", "editorial_angles", "stories", "ideas"):
        if isinstance(director.get(key), list):
            cands = director[key]
            break
    if cands is None:
        for v in director.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) and "angle" in x for x in v):
                cands = v
                break
    out = []
    for i, a in enumerate(cands or [], 1):
        if isinstance(a, dict) and str(a.get("angle", "")).strip():
            a.setdefault("id", i)
            out.append(a)
    return out


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


_STAGED_VISUAL = re.compile(r"\b(animat\w*|diagram|infographic|schematic|split[- ]screen|side[- ]by[- ]side|timeline|"
                            r"stick figure|silhouette|person|people|human|walker|walking|treadmill|clock|hourglass|arrow|"
                            r"compar(?:ison|ing|ed)|montage|collage|overlay|label\w*|caption\w*|graph|chart|marker|"
                            r"cutaway|cross[- ]section|3d model|render|cgi|cartoon)\b", re.I)
VISUAL_SYSTEM = "You write one image brief for a photoreal generator: a photograph a spacecraft or a telescope could take. Return STRICT JSON only."
VISUAL_USER = """Beat {n} of a video on "{topic}". What the voice says: {narration}
Fact of the beat: {fact}
Write the PHOTOGRAPH that should be on screen: the real object itself, its surface, its sky or its instrument, in
its true colours as the fact base describes it (no rings unless the object has rings). One sentence, concrete,
starting with the shot scale (wide shot | medium shot | close-up). Never an animation, diagram, split screen,
timeline, stick figure, arrow, clock, treadmill, person, text or staged comparison. {extra}
Return {{"visual": "...", "keywords": ["3-5 words naming what is in the frame"]}}"""


def staged_visuals(script: Script) -> list[int]:
    """Segments whose visual cue asks the generator for something a camera cannot shoot."""
    return [x.index for x in script.segments if x.kind != "cta" and _STAGED_VISUAL.search(x.visual or "")]


def photographic_visuals(cfg, script: Script, arc: dict | None, topic: str) -> int:
    """Replace every staged visual with a photograph cue: the arc beat's own visual when it is clean,
    otherwise one call per beat. Measured 10/09: the writer 'refined' six clean beats into split screens,
    stick figures, animated diagrams and timelines, and the generator drew exactly that."""
    beats = {int(b.get("beat", 0) or 0): b for b in ((arc or {}).get("beats") or []) if isinstance(b, dict)}
    fixed = 0
    for x in script.segments:
        if x.kind == "cta" or not _STAGED_VISUAL.search(x.visual or ""):
            continue
        beat = beats.get(x.index, {})
        cand = " ".join(str(beat.get("visual", "")).split())
        if cand and not _STAGED_VISUAL.search(cand):
            x.visual = cand
            fixed += 1
            continue
        try:
            data = _call(cfg, VISUAL_SYSTEM, VISUAL_USER.format(n=x.index, topic=topic, narration=x.narration,
                                                                fact=str(beat.get("fact", "")) or x.narration,
                                                                extra="The subject is a real body: its archive photograph is the reference."),
                         editor=False, temperature=0.3, max_tokens=300)
            new = " ".join(str(data.get("visual") or "").split())
            if new and not _STAGED_VISUAL.search(new):
                x.visual = new
                kws = [str(k).strip() for k in (data.get("keywords") or []) if str(k).strip()]
                if kws:
                    x.keywords = kws
                fixed += 1
        except Exception as e:  # noqa: BLE001
            log.warning("visual repair skipped for beat %d (%s)", x.index, e)
    return fixed


def hygiene_en(script: Script, cfg, budget: tuple[int, int, int]) -> list[str]:
    """The mechanical nets on the English — errors, not quality. Each string is a reason for the writer."""
    reasons: list[str] = []
    content = [s for s in script.segments if s.kind != "cta"]
    total = sum(_words(s.narration) for s in content)
    words, lo, hi = budget
    if total < lo or total > hi:
        reasons.append(f"total {total} spoken words, the budget is {lo}-{hi} (about {words}): "
                       + ("cut" if total > hi else "add substance, not padding"))
    # the hook does not BEGIN with a number (a cold listener cannot hold it); "Saturn's rings lose mass at 0.5
    # tonnes" is fine — the six-word rule of the old polish made a cut-to-length pass and a rewrite chase each other
    head = " ".join(content[0].narration.split()[:3]).lower() if content else ""
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
        if _UNSPEAKABLE.search(s.narration):
            reasons.append(f"segment {s.index}: notation a voice cannot read ({_UNSPEAKABLE.search(s.narration).group(0)!r}) — say the number in words a listener follows")
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
        if ratio > 1.5:      # Italian runs 15-30% longer by nature; the voice waits up to 1.2 s for the reader
            reasons.append(f"segmento {i.index}: {ratio:.1f}× i caratteri dell'inglese — il lettore non arriva in fondo; stessa battuta, più asciutta")
        elif ratio < 0.6:
            reasons.append(f"segmento {i.index}: {ratio:.1f}× i caratteri dell'inglese — manca contenuto della battuta")
    for p in italian_lint(it.cta_bridge or ""):
        reasons.append(f"ponte finale: italiano scorretto — {p}")
    if (hit := competitor_mentions(it.cta_bridge or "")):
        reasons.append(f"ponte finale: nomina un'altra app ({hit[0]})")
    return reasons


def _ok(word: str, floor: tuple[str, ...]) -> bool:
    return str(word or "").strip().lower() in floor


def _rubric_clear(d: dict) -> bool:
    return (all(_ok(d.get(k), ("solid", "strong", "exceptional")) for k in ("specificity", "density", "narration", "language"))
            and _ok(d.get("idea"), ("strong", "exceptional"))
            and _ok(d.get("originality"), ("solid", "strong", "exceptional"))
            and _ok(d.get("ai_smell"), ("none", "mild")))


def publishable(review: dict, after_rewrite: bool = False) -> bool:
    """The rubric's own rule: nothing weak, the idea at least strong, the smell of AI at most mild. On a first
    review the editor's word is final ("rewrite" sends the text back even when every dimension passes: the
    weak sentences are worth one pass). After a rewrite the rubric decides — measured 10/09: three reviews in a
    row rated every dimension solid or strong and still said "rewrite" over sentences that "could be more
    vivid", and no video was made."""
    d = review.get("dimensions") or {}
    decision = str(review.get("decision", "")).lower()
    if decision == "reject_story":
        return False
    if decision == "publish":
        return _rubric_clear(d)
    return after_rewrite and _rubric_clear(d)


def story_rejected(review: dict) -> bool:
    d = review.get("dimensions") or {}
    return (str(review.get("decision", "")).lower() == "reject_story"
            or _ok(d.get("idea"), ("weak",)) or _ok(d.get("originality"), ("weak",)))


# ----------------------------------------------------------------------------- the pass
def _limits_note(ref: Script | None) -> str:
    """For the Italian writer: the fitted English beat sets a character cap and the names that must survive.
    The English TEXT is not shown — the Italian is written from the beats, not translated."""
    if ref is None:
        return ""
    rows = []
    for x in ref.segments:
        if x.kind == "cta":
            continue
        names = ", ".join(names_for_italian(x.narration)) or "—"
        rows.append(f"battuta {x.index}: al massimo {int(len(x.narration) * 1.3)} caratteri; nomi da conservare: {names}")
    return ("LIMITI PER BATTUTA (rigidi, contali; la voce inglese dura quanto dura, il lettore deve arrivare in fondo):\n- "
            + "\n- ".join(rows))


def _write(cfg, language: str, topic: str, brief: dict, arc: dict, facts: str, budget: tuple[int, int, int],
           n_beats: int, poetics: str, target: int, diagnosis: dict | None = None, notes: list[str] | None = None,
           temperature: float = 0.55, ref: Script | None = None) -> Script:
    words, lo, hi = budget
    system = WRITE_SYSTEM.format(poetics=poetics)
    user = WRITE_USER.format(language=language, topic=topic, brief=_j(brief), arc=_j(arc), facts=facts, words=words,
                             lo=lo, hi=hi, per=max(MIN_WORDS_PER_BEAT, words // max(1, n_beats)), n_beats=n_beats,
                             min_w=MIN_WORDS_PER_BEAT, max_w=MAX_WORDS_PER_BEAT,
                             run_on=RUN_ON_WORDS, language_note=LANG_NOTES.get(language, ""),
                             limits_note=_limits_note(ref) if language == "Italian" else "")
    if diagnosis:
        user += REVISE_NOTE.format(diagnosis=_j(diagnosis))
    if notes:
        user += "\n\nHYGIENE NOTES from the previous attempt — every one must be resolved:\n- " + "\n- ".join(notes)
    out = _to_script(_call(cfg, system, user, editor=False, temperature=temperature, max_tokens=3200), topic, target)
    if len(out.segments) != n_beats:            # one segment per beat is the contract: ask once more, plainly
        out = _to_script(_call(cfg, system, user + f"\n\nYou returned {len(out.segments)} segments. Return EXACTLY {n_beats}, one per beat, in order.",
                               editor=False, temperature=temperature, max_tokens=3200), topic, target)
        if len(out.segments) != n_beats:
            raise EditorialError(f"the {language} writer returned {len(out.segments)} segments for {n_beats} beats, twice")
    return out


_LENGTH_MARKERS = ("spoken words", "run-on", "words (>", "caratteri dell'inglese")
_TOO_SHORT_MARKERS = ("add substance", "manca contenuto", "a flash, not a beat")


def _length_only(reasons: list[str]) -> bool:
    """True when every reason is an OVER-length problem — the only kind a cut can solve. A text that is too
    short (fifth Venus trial: 82 words, three cuts, 82 words) needs the writer, not the scissors."""
    return bool(reasons) and all(any(m in r for m in _LENGTH_MARKERS) and not any(t in r for t in _TOO_SHORT_MARKERS)
                                 for r in reasons)


COMPRESS_SYSTEM = "You compress one spoken sentence of a science video to a hard word cap. You remove words, never facts, numbers or names. Return STRICT JSON only."
COMPRESS_USER = """Compress this {language} segment to AT MOST {cap} words (it has {n}). Keep every fact, number, name and the
meaning; drop asides, doubled adjectives, repeated context and citation-speak ("per a 2018 Icarus study" → "a 2018
study" or nothing). One or two sentences, none longer than {run_on} words. {extra}
SEGMENT: {text}
Return {{"narration": "...", "words": <your count>}}"""


def _compress_segments(cfg, language: str, script: Script, caps: dict[int, int], extra: str = "") -> Script:
    """The reliable cut: one segment, one hard cap, one call — and the count is checked. A segment still over
    its cap is asked again with a lower cap, three tries at most (the model returns 21 for a cap of 19; told 17
    it returns 19). Segments already under their cap are untouched."""
    for x in script.segments:
        if x.kind == "cta" or x.index not in caps or _words(x.narration) <= caps[x.index]:
            continue
        note = extra + (" Do not begin with a digit or a number word." if x.index == 1 else "")
        cap = caps[x.index]
        best = x.narration
        for _try in range(3):
            data = _call(cfg, COMPRESS_SYSTEM, COMPRESS_USER.format(language=language, cap=cap, n=_words(best),
                                                                    run_on=RUN_ON_WORDS, extra=note, text=best),
                         editor=False, temperature=0.2, max_tokens=400)
            new = " ".join(str(data.get("narration") or "").split())
            if new and MIN_WORDS_PER_BEAT <= _words(new) < _words(best):
                best = new
            if _words(best) <= caps[x.index]:
                break
            cap = max(MIN_WORDS_PER_BEAT, cap - 2)      # ask for less than needed: the model undershoots the cut
        x.narration = best
    return script


COMPRESS_IT_SYSTEM = "Riscrivi una battuta di un video scientifico entro un limite rigido di caratteri. Togli parole, mai fatti, numeri o nomi. Restituisci SOLO JSON."
COMPRESS_IT_USER = """Riscrivi questa battuta in AL MASSIMO {cap} caratteri (ne ha {n}), conservando ogni fatto, ogni numero e questi
nomi: {names}. Stessa battuta, stesso senso, italiano naturale da redazione con il soggetto esplicito; niente calchi,
niente "scatto" per una tacca. {extra}
BATTUTA: {text}
Restituisci {{"narration": "...", "chars": <conteggio>}}"""


def _compress_it(cfg, it: Script, en: Script, squeeze: int = 0) -> Script:
    """The Italian cut: one beat, one character cap from the fitted English beat, the names listed."""
    ec = {x.index: x for x in en.segments if x.kind != "cta"}
    for x in it.segments:
        if x.kind == "cta" or x.index not in ec:
            continue
        target_cap = int(len(ec[x.index].narration) * 1.35) - squeeze
        if len(x.narration) <= target_cap:
            continue
        names = ", ".join(names_for_italian(ec[x.index].narration)) or "nessuno"
        best, cap = x.narration, target_cap
        for _try in range(3):                       # the model undershoots the cut: ask for less each time
            data = _call(cfg, COMPRESS_IT_SYSTEM,
                         COMPRESS_IT_USER.format(cap=cap, n=len(best), names=names, text=best,
                                                 extra="Non iniziare con un numero." if x.index == 1 else ""),
                         editor=False, temperature=0.2, max_tokens=400)
            new = " ".join(str(data.get("narration") or "").split())
            if new and len(new) < len(best) and not italian_lint(new) and not italian.bad_sense(new):
                best = new
            if len(best) <= target_cap:
                break
            cap = max(40, int(cap * 0.88))
        x.narration = best
    return it


REPAIR_IT_SYSTEM = "Ripari una sola battuta del copione italiano di un video scientifico. Stessa battuta, stessi fatti, italiano da redazione. Restituisci SOLO JSON."
REPAIR_IT_USER = """Ripara questa battuta (la numero {n}) risolvendo TUTTI i problemi elencati, senza toccare le altre.
Fatto della battuta (dall'arco narrativo): {fact}
Nomi che la battuta deve contenere: {names}
Lunghezza: tra {lo} e {hi} caratteri (ora {now}).
Problemi da risolvere:
{problems}
Regole: soggetto esplicito, passato prossimo, niente calchi, terminologia esatta (tacca non scatto, obiettivo non lente,
chilometri), numeri in cifre in formato italiano; non tradurre: scrivi la battuta in italiano da redazione. {extra}
TESTO ATTUALE: {text}
Restituisci {{"narration": "..."}}"""


def _repair_it(cfg, it: Script, en: Script, arc: dict, reasons: list[str]) -> Script:
    """The Italian's corrective path: one beat, its own problems, one call. Whole-script rewrites lost names
    and length in the very beats they were asked to fix (sixth Venus trial); a beat repaired alone with its
    fact, its names and its length band converges."""
    by_seg: dict[int, list[str]] = {}
    for r in reasons:
        m = re.search(r"segmento (\d+)", r)
        if m:
            by_seg.setdefault(int(m.group(1)), []).append(re.sub(r"^segmento \d+:\s*", "", r))
    ec = {x.index: x for x in en.segments if x.kind != "cta"}
    beats = arc.get("beats") or []
    for x in it.segments:
        if x.kind == "cta" or x.index not in by_seg or x.index not in ec:
            continue
        e = ec[x.index]
        fact = next((str(b.get("fact", "")) for b in beats if int(b.get("beat", 0) or 0) == x.index), "") or "(vedi la battuta inglese: stessi fatti)"
        lo, hi = int(len(e.narration) * 0.75), int(len(e.narration) * 1.35)
        data = _call(cfg, REPAIR_IT_SYSTEM,
                     REPAIR_IT_USER.format(n=x.index, fact=fact, names=", ".join(names_for_italian(e.narration)) or "nessuno",
                                           lo=lo, hi=hi, now=len(x.narration), problems="- " + "\n- ".join(by_seg[x.index]),
                                           extra="Non iniziare con un numero." if x.index == 1 else "", text=x.narration),
                     editor=False, temperature=0.3, max_tokens=500)
        new = " ".join(str(data.get("narration") or "").split())
        if new and not italian_lint(new) and not italian.bad_sense(new) and not competitor_mentions(new):
            x.narration = new
    return it


def _caps_en(script: Script, hi: int, squeeze: int = 0) -> dict[int, int]:
    """Per-segment word caps that sum to a little UNDER the budget's ceiling: proportional to the current
    lengths, floored, with a margin of 3 words so rounding never lands on the line."""
    content = [x for x in script.segments if x.kind != "cta"]
    total = sum(_words(x.narration) for x in content)
    scale = min(1.0, (hi - 3 - squeeze) / max(1, total))
    return {x.index: max(MIN_WORDS_PER_BEAT, int(_words(x.narration) * scale)) for x in content}


def _fit(cfg, language: str, topic: str, brief: dict, arc: dict, facts: str, budget: tuple[int, int, int], n_beats: int,
         poetics: str, target: int, script: Script, check, drafts: list, ref: Script | None = None,
         diagnosis: dict | None = None) -> Script:
    """Three corrective passes at most: a rewrite with the notes when the problems are of substance, a cut to
    length when they are only of length (the writer overshoots by 20-30% whatever it is told, the cut lands);
    then the nets are final. Measured 10/09: rewrite → cut → cut converges, rewrite → rewrite does not."""
    for attempt in (1, 2, 3):
        reasons = check(script)
        drafts.append({"language": language, "attempt": attempt, "reasons": reasons, "words": sum(_words(x.narration) for x in script.segments),
                       "segments": [x.narration for x in script.segments]})
        if not reasons:
            return script
        log.info("Hygiene %s (%d): %s", language, attempt, "; ".join(reasons))
        if language == "Italian" and ref is not None and all(re.search(r"segmento \d+", r) for r in reasons):
            over = [r for r in reasons if "il lettore non arriva in fondo" in r]
            other = [r for r in reasons if r not in over]
            if other:
                script = _repair_it(cfg, script, ref, arc, other)         # beat by beat: names, content, lint
            if over or hygiene_it(ref, script):
                script = _compress_it(cfg, script, ref, squeeze=(attempt - 1) * 8)   # the beats that run long
        elif _length_only(reasons):
            words, lo, hi = budget
            if language == "English":
                script = _compress_segments(cfg, language, script, _caps_en(script, hi, squeeze=(attempt - 1) * 4))
            else:
                script = _compress_it(cfg, script, ref, squeeze=(attempt - 1) * 12)
        else:
            script = _write(cfg, language, topic, brief, arc, facts, budget, n_beats, poetics, target, notes=reasons,
                            temperature=0.4, ref=ref, diagnosis=diagnosis)
    reasons = check(script)
    drafts.append({"language": language, "attempt": 4, "reasons": reasons, "words": sum(_words(x.narration) for x in script.segments),
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


def _pass(cfg, topic: str, project, facts: str, history: str, attempt: int, avoid: list[str],
          scratch: dict | None = None) -> tuple[Script, Script, dict, dict]:
    scratch = scratch if scratch is not None else {}
    theory = standard("theory_of_value.md") or "Prefer tension, reversal, tangible scale, mechanism, honest open questions, human traces, a change of view; kill the obvious, the unverifiable, the unvisualisable, the recently told."
    poetics = standard("poetics.md") or "Precision, information density, depth, rhythm, naturalness, voice; no declared enthusiasm, no riddles, no unearned adjectives, no textbook explanations, no promotional closers."
    benchmark = benchmark_text()
    target = int(getattr(cfg.script, "target_seconds", 48) or 48)
    root: Path = project.root

    # 1 · director (the answer is read leniently and asked again once: a truncated or oddly keyed JSON is
    #     not "no story worth telling")
    avoid_note = ("\nANGLES ALREADY REJECTED BY THE EDITOR THIS RUN (do not propose them again):\n- " + "\n- ".join(avoid) + "\n") if avoid else ""
    director_user = DIRECTOR_USER.format(topic=topic, facts=facts, history=history, avoid=avoid_note)
    director, angles = {}, []
    for extra in ("", "\n\nYour previous answer had no readable \"angles\" list. Return exactly {\"angles\": [...10 items...], \"ranking\": [...], \"comparison\": \"...\"}."):
        director = _call(cfg, DIRECTOR_SYSTEM.format(theory=theory), director_user + extra, editor=False, temperature=0.55, max_tokens=5000)
        angles = _angles_from(director)
        if len(angles) >= 3:
            break
    (root / "editorial_director_raw.json").write_text(_j(director))
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
    scratch["brief"], scratch["winner"], scratch["why"] = brief, winner, selection.get("why_this_story", "")
    # 4 · narrative design + independent arc selection
    n_beats = 6
    arcs_data = _call(cfg, NARRATIVE_SYSTEM.format(n_beats=n_beats), NARRATIVE_USER.format(topic=topic, brief=_j(brief), facts=facts),
                      editor=False, temperature=0.5, max_tokens=3600)
    arcs = [a for a in (arcs_data.get("arcs") or []) if isinstance(a, dict) and a.get("beats")]
    redundant = {int(a.get("id", i)): arc_redundancy(a) for i, a in enumerate(arcs, 1)}
    clean = [a for a in arcs if not redundant.get(int(a.get("id", 0)))]
    if len(clean) < 2:            # ask once more, naming the repeats
        note = "\n\nYour previous arcs repeated facts across beats: " + "; ".join(
            f"arc {k}: {', '.join(v)}" for k, v in redundant.items() if v) + ". Every beat a different fact."
        arcs_data = _call(cfg, NARRATIVE_SYSTEM.format(n_beats=n_beats), NARRATIVE_USER.format(topic=topic, brief=_j(brief), facts=facts) + note,
                          editor=False, temperature=0.5, max_tokens=3600)
        arcs = [a for a in (arcs_data.get("arcs") or []) if isinstance(a, dict) and a.get("beats")]
        redundant = {int(a.get("id", i)): arc_redundancy(a) for i, a in enumerate(arcs, 1)}
        clean = [a for a in arcs if not redundant.get(int(a.get("id", 0)))] or arcs
    arcs = clean
    if len(arcs) < 1:
        raise EditorialError("the narrative designer returned no usable arc")
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
        if staged_visuals(en):
            n_fixed = photographic_visuals(cfg, en, arc, topic)
            log.info("Visuals: %d staged cue(s) replaced with photograph cues", n_fixed)
        it = _write(cfg, "Italian", topic, brief, arc, facts, budget, n_beats, poetics, target, ref=en)
        it = _fit(cfg, "Italian", topic, brief, arc, facts, budget, n_beats, poetics, target, it,
                  lambda s: hygiene_it(en, s), drafts, ref=en)
    finally:
        (root / "editorial_drafts.json").write_text(_j(drafts))
    scratch["en"], scratch["it"], scratch["arc"] = en, it, arc
    # 6 · facts and proof (corrections), back-translation compare (one Italian pass if beats disagree)
    notes = _fix_facts_and_proof(cfg, en, it, facts, root)
    diffs = [n for n in notes if "senso diverso" in n]
    if diffs:
        log.info("Compare IT/EN: %s", "; ".join(diffs))
        it = _write(cfg, "Italian", topic, brief, arc, facts, budget, n_beats, poetics, target, notes=diffs + hygiene_it(en, it),
                    temperature=0.4, ref=en)
        it = _fit(cfg, "Italian", topic, brief, arc, facts, budget, n_beats, poetics, target, it,
                  lambda x: hygiene_it(en, x), drafts, ref=en)
    # 7 · editorial review, rewrite, second review
    reviews: dict = {}
    for lang, s in (("English", en), ("Italian", it)):
        scratch["en"], scratch["it"] = en, it
        r1 = _review(cfg, lang, topic, brief, s, poetics, benchmark)
        scratch.setdefault("reviews", {})[f"{lang}_v1"] = r1
        suffix = "en" if lang == "English" else "it"
        (root / f"editorial_review_{suffix}_v1.json").write_text(_j(r1))
        reviews[f"{suffix}_v1"] = r1
        if story_rejected(r1):
            raise StoryRejected(wid, f"{lang}: {r1.get('summary', '')}")
        if publishable(r1):
            continue
        prev, diag, approved = s, r1, None
        for round_no in (2, 3):                      # at most two rewrites, each judged against the version before
            v_next = _write(cfg, lang, topic, brief, arc, facts, budget, n_beats, poetics, target, diagnosis=diag,
                            temperature=0.35, ref=en if lang == "Italian" else None)
            # the rewrite is fitted like a first draft: a long rewrite is cut, a wrong one is written again with the diagnosis
            v_next = _fit(cfg, lang, topic, brief, arc, facts, budget, n_beats, poetics, target, v_next,
                          (lambda x: hygiene_en(x, cfg, budget)) if lang == "English" else (lambda x: hygiene_it(en, x)),
                          drafts, ref=en if lang == "Italian" else None, diagnosis=diag)
            if lang == "English" and staged_visuals(v_next):
                photographic_visuals(cfg, v_next, arc, topic)
            r_next = _review2(cfg, lang, topic, brief, prev, diag, v_next, poetics, benchmark)
            scratch.setdefault("reviews", {})[f"{lang}_v{round_no}"] = r_next
            scratch["en" if lang == "English" else "it"] = v_next
            (root / f"editorial_review_{suffix}_v{round_no}.json").write_text(_j(r_next))
            reviews[f"{suffix}_v{round_no}"] = r_next
            if r_next.get("improved") is False or story_rejected(r_next):
                raise StoryRejected(wid, f"{lang}: the rewrite merely complied — {r_next.get('improvement_note') or r_next.get('summary', '')}")
            if publishable(r_next, after_rewrite=True):
                approved = v_next
                break
            prev, diag = v_next, r_next             # improving but not there yet: one more round with the new diagnosis
        if approved is None:
            raise EditorialError(f"{lang} script below the editorial standard after two rewrites: {diag.get('summary', '')}")
        v2 = approved
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
        en.segments.append(_cta_segment(en, it, cfg))
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
    scratch: dict = {}
    for attempt in (1, 2):
        try:
            en, it, brief, evidence = _pass(cfg, topic, project, facts, history, attempt, avoid, scratch)
        except StoryRejected as e:
            last = e
            log.warning("Attempt %d: %s — the director looks for another angle", attempt, e)
            avoid.append(f"angle id {e.angle_id}: {e.why[:160]}")
            continue
        except EditorialError as e:
            last = e
            break
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
    if os.getenv("AVP_EDITORIAL_TRIAL", "").strip() == "1" and scratch.get("en") and scratch.get("it"):
        # TRIAL MODE: the story was rejected, but the owner wants to SEE what the machine wrote. The last
        # drafts are saved flagged "rejected"; the build runs, QA refuses to publish (qa.check reads the flag).
        _save_rejected(project, cfg, topic, scratch, str(last))
        raise last or EditorialError("editorial generation failed")
    raise last or EditorialError("editorial generation failed")


def _cta_segment(en: Script, it: Script, cfg) -> Segment:
    """The spoken CTA and its Italian card. A bridge the policy does not speak is dropped; the generic funnel
    question gets the configured Italian card (the Italian writer wrote no bridge for it)."""
    from . import stages
    cta_text = stages._cta_narration(en, cfg)
    if en.cta_bridge and en.cta_bridge not in cta_text:
        en.cta_bridge = it.cta_bridge = ""
    generic = cfg.funnel.cta_line.format(app=cfg.funnel.app_name)
    if cta_text.strip() == generic.strip():
        card = str(getattr(cfg.funnel, "cta_line_it", "") or "")
        en.cta_bridge = generic.split("Get ")[0].strip()          # what parse_script_md will read back as the bridge
    else:
        card = it.cta_bridge if en.cta_bridge and it.cta_bridge else ""
    return Segment(index=len(en.segments) + 1, narration=cta_text, visual="App endcard", keywords=[], kind="cta", italian=card)


def _save_rejected(project, cfg, topic: str, scratch: dict, why: str) -> None:
    from . import stages
    en, it = scratch["en"], scratch["it"]
    for a, b in zip([x for x in en.segments if x.kind != "cta"], [x for x in it.segments if x.kind != "cta"]):
        a.italian = b.narration
    if cfg.funnel.enabled and not any(x.kind == "cta" for x in en.segments):
        en.segments.append(_cta_segment(en, it, cfg))
    project.script_json.write_text(stages._json(en.to_dict()))
    (project.root / "italian_script.json").write_text(stages._json(it.to_dict()))
    stages.emit_script_md(en, project.script_md)
    (project.root / "editorial_report.json").write_text(_j({"version": 3, "status": "rejected", "topic": topic, "title": en.title,
                                                            "why": why, "winner": scratch.get("winner"), "why_this_story": scratch.get("why"),
                                                            "reviews": scratch.get("reviews", {}), "brief": scratch.get("brief")}))
    project.manifest.data["topic"] = topic
    project.manifest.data["title"] = en.title
    project.manifest.data.setdefault("editorial", {}).update({"version": 3, "status": "rejected", "why": why[:300]})
    project.manifest.mark("script", "done", title=en.title, segments=len(en.segments))
    log.warning("TRIAL MODE: rejected story saved for viewing only — QA will refuse to publish it (%s)", why[:120])
