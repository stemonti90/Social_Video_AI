"""Script generation via a local Ollama model (default: qwen3:14b, Apache-2.0).

Talks to Ollama's HTTP API and asks for strict JSON, which we parse into a Script.
The prompt encodes the channel's editorial rules: accurate, original, hook-first,
never template-stamped (which is what keeps it monetizable under YouTube's
'inauthentic content' policy)."""
from __future__ import annotations

import json
import re

import requests

from .config import FunnelConfig, LLMConfig
from .log import get_logger
from .models import Script, Segment, dedupe_segments

log = get_logger("avp.llm")

SYSTEM = """You are an elite scriptwriter for a faceless short-form video channel about \
astronomy and space (TikTok / Reels / YouTube Shorts). You write to make a stranger stop scrolling and \
stay to the end. Craft first, then the truth rules — both are non-negotiable.

WHAT WORKS ON THIS CHANNEL — the voice to match (never copy these lines):
  "Saturn's rings are younger than the dinosaurs."
  "There is a moon where it rains gasoline."
  "Jupiter has a storm three Earths wide that has been raging for centuries."
  "The Moon is quietly leaving us — four centimetres a year."
  "A robot the size of a car is driving on Mars right now, alone."
  "Mercury hides ice in craters the Sun has never once lit."
Every one of those is a CLAIM THAT STOPS YOU: something surprising, hidden, impossible-sounding, or \
too close to home. Each has one concrete image and each makes you need the next line. And every one is \
A REAL FACT IN DISGUISE: under "younger than the dinosaurs" sits Cassini's estimate of the rings' age; \
under "rains gasoline" sits Titan's methane cycle; under "quietly leaving us" sits the 3.8 cm/year \
recession measured by laser. Before you write an image, name to yourself the checkable fact beneath \
it. An image with no fact under it is not bold, it is false, and it is banned. These example lines are \
OFF-LIMITS verbatim: reuse the energy, never the words.

THE TONE IS WONDER, NOT THE MORGUE. Earlier scripts opened with corpses, hemorrhages, tombs, graveyards, \
autopsies, torture chambers and screaming — the audience found it grim, not gripping. BANNED anywhere \
in the script: corpse, cadaver, dead, dying, died, death, kill, killed, murder, tomb, grave, graveyard, \
autopsy, hemorrhage, bleed, bleeding, wound, scar, torture, suicide, hell, hellscape, nightmare, \
scream, screaming, butcher, prison, prisoner, gutted, rot, decay, doom, apocalypse, and any "planetary \
X" where X is a body part or an injury. Stakes come from AWE and SURPRISE — scale, strangeness, \
things that should not exist, things happening right now over your head — never from gore or dread. \
Curious, confident, a little playful; the tone of someone who cannot wait to tell you this.

WHAT FAILED ON THIS CHANNEL (the lines viewers called "written by a child"):
  "These deep craters trap vapor in shadows where the sun never reaches, ensuring no heat can enter."
  "This extreme cold turns surrounding gases into a solid, crystalline frost."
Those are EXPLAINER sentences: calm mechanism, "This/These" openers, restating the line before. Nobody \
stops scrolling for a mechanism. BANNED: any sentence that opens with "This", "These" or "It" followed \
by a calm verb; any line that paraphrases an earlier one; "ensuring", "allowing", "creating", \
"resulting in"; and padding of any kind.

- HOOK: the first 6-8 words RENAME the subject as something surprising — a fact that sounds impossible, a secret, a scale that breaks intuition. Never a measurement, never a question, never "Imagine", "Have you ever", "Picture this", "In the vast expanse", and never anything from the banned list above. A number is evidence: it belongs in segment 2, where it lands on a viewer you have already stopped.
- EVERY SEGMENT IS A REVEAL. One new, specific, verifiable fact, delivered as a claim that surprises — then its consequence, in one concrete image. If a line could sit in a textbook unchanged, rewrite it until it could not.
- ESCALATE. Each segment raises the stakes of the one before; the penultimate segment is the single biggest turn in the script; the last one lands it with a line that would work as the title.
- LENGTH: about the per-segment word count given below, in one or two spoken sentences. A shorter script in which every line earns its place beats a longer one with a single soft line. NEVER pad to reach a count.
- DRAMATURGY, not exposition: a fact that shouldn't be possible → why it shouldn't be → how it is anyway → what that means. Explanation arrives as the answer to a tension you opened, never as a lecture.
TRUTH RULES (every one of these has cost a science channel its credibility in the comments):
- Segment 1 must NOT begin with the subject's name, and must not announce what the subject is. It states the single strangest concrete thing in the whole script — the fact you would lead with if you had one sentence to stop someone scrolling. The viewer should work out what the video is about from that fact, not be told. ("Voyager 1 carries a golden record into the void" is still an announcement; "A machine 25 billion kilometres away is still talking to us" is a hook.)
- Every segment must leave the next one NECESSARY: end on a consequence, a number that begs a question, or an unresolved tension. If a segment could be the last one, it is written wrong.
- Every segment must make a DISTINCT point. NEVER repeat, restate or paraphrase an earlier line to reach the segment count — if you genuinely run out of distinct facts, broaden the angle (history, mechanism, scale, discovery, what's next) rather than repeating.
- Open a curiosity loop in the first 1-2 segments and PAY IT OFF before the end.
- Escalate to a single peak "wow" moment in the penultimate segment.
- Concrete nouns over adjectives; a confident, flowing spoken cadence (not terse, not padded). BANNED words/phrases: mind-blowing, incredible, literally, breathtaking, journey, unlock, delve, "did you know". No markdown, no emojis, no stage directions.
- Every fact must be real and checkable; if unsure, stay qualitative — never invent numbers.
- NEVER INVENT A MECHANISM. Explaining *how* something worked is where fabrication creeps in: a solar-powered rover described as "scavenging power from the soil" sounds plausible and is simply false. If you do not know the mechanism, say what the thing DID, not how it did it. A vaguer true sentence always beats a vivid invented one.
- GET THE TENSE RIGHT. Missions end, probes fall silent, spacecraft are destroyed on purpose. Before writing "is still...", check whether it still is. Saying a rover that fell silent years ago "is still transmitting" is the kind of error that costs a science channel its credibility in the comments.
- A NUMBER MUST MEASURE WHAT THE SENTENCE SAYS IT MEASURES. Distance, diameter, age, mass, temperature and speed are different quantities, and the most common error in astronomy writing is quoting an object's DISTANCE as its SIZE. The Orion Nebula is ~1,340 light-years AWAY and ~24 light-years ACROSS; writing "a 1,300 light-year wide nebula" is wrong by a factor of fifty. Before every figure, name to yourself which quantity it belongs to.
- SUPERLATIVES are where these scripts get things wrong most often. Before writing "the only", "the first", "the largest", "the farthest", check whether a second case exists — there usually is one (Voyager 2 reached interstellar space too; several probes have left the planets behind). If you cannot be certain, drop the superlative: the fact is almost always just as strong without it.
- Use METRIC units (kilometres). The audience reads subtitles in a metric country, and a distance in miles carries no sense of scale for them.
- Vary structure between videos; never sound template-stamped.
- "keywords": 2-4 ENGLISH, archive-catalogable PROPER NOUNS (e.g. "Cassini Saturn", "Carina Nebula JWST", "Curiosity rover Mars") — concrete subjects a NASA/Hubble search will find, matching that segment's visual.
Return STRICT JSON only, no commentary."""

CRITIQUE_SYSTEM = (
    "You are a ruthless short-form video editor. First, grade this astronomy/space Shorts script 1-5 "
    "on WONDER: does the hook rename the subject as something surprising, does every segment make a "
    "claim that stops you (something hidden, impossible-sounding, happening right now) rather than "
    "explain a mechanism, is the tone awe and curiosity rather than gore or dread (quote every "
    "morbid word — corpse, tomb, hemorrhage, scream, hell — with a wonder-based rewrite), "
    "does it escalate to one peak turn, would a stranger stay to the end? Quote every EXPLAINER "
    "sentence — one that starts with This/These/It and calmly describes how something works, or "
    "restates an earlier line — and give an exact rewrite with stakes for each. Then grade fact "
    "density, the curiosity loop, concrete nouns over adjectives, ZERO cliche, and NO repeated or "
    "near-duplicate segments. Then CHECK THE FACTS: for EVERY number, name the quantity it claims to "
    "measure and say whether it belongs to it — an object's distance quoted as its width is the most "
    "common failure. Flag every superlative (only/first/largest/farthest) that is false or "
    "unverifiable, every number that looks invented, and any distance not in kilometres. "
    "Be specific and brutal. Plain text, no JSON."
)
CRITIQUE_USER = "Script JSON:\n{script}"
REFINE_SUFFIX = (
    "\n- You are now REVISING an existing draft to satisfy an editor's critique: land the hook in "
    "the first 6-8 words, turn every explainer sentence into a claim with stakes, raise fact "
    "density, kill every cliche, keep the curiosity loop paid off, and make every segment DISTINCT "
    "— merge or cut any repeated/near-duplicate lines. Do not soften a line to make it safer; make "
    "it truer. Keep the SAME JSON shape and a similar segment count."
)
REFINE_USER = ("Current draft JSON:\n{script}\n\nEditor critique to apply:\n{critique}\n\n"
               "Return the improved STRICT JSON only.")

USER_TMPL = """Topic: {topic}
The topic may be written in ANOTHER LANGUAGE: translate it faithfully first, then write about EXACTLY that subject. Never drift to a different, more famous subject, and never copy the examples in these instructions.
Target: ~{seconds}s of spoken narration — about {words} words TOTAL (stay within ±10%; do NOT run long), in EXACTLY {nseg}-{nseg2} segments. Going over the length is as wrong as coming in short.
If the topic is a specific MISSION, PROBE, OBJECT, PERSON, or EVENT, EXPLAIN it with a clear through-line — what it is, what it did / what happened, and why it matters — not a list of disconnected trivia.
For each segment provide:
- "narration": about {wseg} words — one or two short spoken sentences,
- "visual": a short cue for the ideal footage or image OF THIS TOPIC. Name the SHOT SCALE explicitly — "extreme close-up of ...", "wide shot, the object tiny against ...", "the object filling the frame". Consecutive segments must NOT use the same scale: the video is cut from these, and two identical scales in a row read as one long static shot.
  It must describe something a CAMERA COULD PHOTOGRAPH. Never ask for a diagram, chart, cutaway, infographic, labelled figure, arrow, split-screen or side-by-side comparison: these are drawn, not shot, and the image generator renders them as garbled pseudo-text. When the narration explains a MECHANISM, film its consequence instead — for "Mars has no plate tectonics" show the unbroken volcanic plain, not a tectonic diagram.
- "keywords": 2-4 ENGLISH search keywords for space archives (they are English-indexed).

Also provide a punchy "title", plus the BRIDGE to astrophotography — this channel exists to make people
want to point their own camera at the sky, so every video has to land somewhere real. Two fields:

- "bridge_kind": choose HONESTLY from exactly one of these:
    "shoot"     — this subject, or the very sky it sits in, can actually be photographed from a
                  balcony with a phone. This is the DEFAULT whenever the topic touches anything
                  visible from Earth, and it is broader than it first looks: ANY naked-eye planet
                  (Mars, Venus, Jupiter, Saturn, Mercury) counts, and so does the sky an object
                  sits in — a video about a Mars rover still bridges to Mars, a bright orange dot
                  anyone can find. Also the Moon, bright nebulae and clusters, the Milky Way, the
                  ISS, eclipses, aurorae, meteor showers, comets. These are examples, NOT a closed
                  list: if a viewer could point a phone at anything in this story, choose "shoot".
    "principle" — the subject itself is out of reach, but it RESTS ON a principle astrophotography also
                  obeys: faint signal buried in noise, gathering light over time, how far light travels,
                  what limits resolution, how little light a sensor really receives.
    "none"      — neither is true without inventing a connection.
- "cta_bridge": ONE short spoken sentence (max 16 words) that earns that bridge.

HARD RULES for the bridge, because getting this wrong makes the channel look ridiculous:
- It may ONLY refer to something this script actually talked about. Naming a different object that
  merely happens to be photographable ("...capture Jupiter's moons") after a video about an
  interstellar probe is exactly the failure to avoid.
- Only choose "principle" when NOTHING in the story is visible from Earth. Reaching for a principle
  while the subject sits in plain sight — bridging a Mars video to light-travel time when Mars itself
  is a bright dot in tonight's sky — throws away the strongest link you had.
- For "principle", bridge through the SHARED IDEA, not the object. A probe whose signal arrives buried
  in noise, and which is read by combining several antennas, bridges to combining many photographs —
  because it is the same physics, not because both involve space.
- The bridge is a claim, and a wrong claim is worse than a weak one. "Starlight that travelled millions
  of kilometres" is false (starlight travels light-YEARS); an editor would catch it and so will your
  audience. Say nothing you cannot stand behind.
- Never write a generic line ("capture the cosmos yourself", "explore the universe"). If it could close
  ANY video on this channel, it is wrong.
- If "none", still write a cta_bridge, but make it a closing thought about the sky itself. Do not
  mention cameras or photography.
- Do NOT name any app.

Return JSON exactly like:
{{"title": "...", "bridge_kind": "shoot|principle|none", "cta_bridge": "...", "segments": [{{"narration": "...", "visual": "...", "keywords": ["...", "..."]}}]}}"""


# Measured speaking rate of the Kokoro voices, words per second, from real builds:
# EN (af_heart) 2.37-2.46 · IT (if_sara) 2.60-2.65. A single 2.5 made English scripts overshoot
# (a 50s target came out 60s of speech), so the budget is language-aware — slightly conservative
# so the video lands UNDER the target rather than over it.
# The writer's terseness, measured on THIS prompt: it returns ~70% of the word budget it is asked for
# (77, 90, 92, 97 words against an ask of 128). An earlier reading of 88% came from builds before the
# hook and photographable-visual rules went in — those rules made it terser, so the compensation had
# to be re-measured rather than carried over.
#
# This is the only length lever that works. Editing does not: asked to LENGTHEN, this model ignores
# the target and simply doubles what it already has (77->152, 90->172, 92->177, 100->211, all ~2.0x),
# and asked to shorten by a quarter it shaves 1-3%. So there is no route from a 77-word draft to 112
# — the guards below can only reject the overshoot and keep the short draft. Getting the FIRST draft
# near target is therefore the whole game, and generation is where the model actually complies.
ASK_INFLATION = 1.15   # was 1.45: the extra 30% came back as padding, and padding read as childish

# The SLOWEST delivery measured, not the average (2.33-2.61 en across builds). Budgeting at the
# average makes half of all videos longer than planned, and long is the failure that breaks
# the 60s rule.
_WPS = {"en": 2.33, "it": 2.60}

# How long one image holds the screen. This single number is the cut rhythm, and both ends of the
# range were found by shipping the mistake: ~7s (four segments in a 50s video) reads as a television
# documentary, ~2.4s reads as the video played at 1.5x — viewers reported not being able to read.
# Neither was a speech problem; the voice measures 2.38 words/second in every build.
SECONDS_PER_IMAGE = 4.3


def _words_for(seconds: int, language: str = "en") -> int:
    return max(20, round(seconds * _WPS.get((language or "en").lower()[:2], 2.4)))


def _read_timeout(model: str) -> int:
    """Read-timeout (seconds) for an Ollama call. The 16GB gemma4 MLX build can need several minutes
    to COLD-LOAD from disk and then generate — notably the metadata stage, which runs after the
    assemble stage evicted every model under memory pressure, so it reloads from scratch. A 5-min
    bound (fine for ~10GB qwen3) was too tight and timed metadata out. Give gemma 10 min; others 5."""
    return 600 if model.lower().lstrip().startswith("gemma") else 300


def _supports_constrained_json(model: str) -> bool:
    """Whether Ollama's constrained-decoding (`format="json"` or a JSON schema) is safe for this
    model. It is NOT for Gemma: under the grammar constraint gemma4 either collapses to an empty
    `{}` (format="json") or hangs past the read timeout (schema) — verified on gemma4:26b. For those
    models we drop the constraint and let `_extract_json` pull the JSON out of a free-text reply
    (the prompts already demand "STRICT JSON only", and gemma honours that in free-text mode).
    qwen3 and the other defaults keep constrained JSON, which guarantees parseable output."""
    return not model.lower().lstrip().startswith("gemma")


class OllamaClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def chat(self, system: str, user: str, fmt: str | None = "json",
             temperature: float | None = None, num_predict: int | None = None) -> str:
        options = {
            "temperature": self.cfg.temperature if temperature is None else temperature,
            # num_ctx caps the KV cache: script/metadata prompts are a few K tokens, so 8K is ample
            # and avoids a multi-GB cache (the model's default 40K context bloated memory → swap).
            "num_ctx": 8192,
        }
        if num_predict and not self.cfg.model.lower().lstrip().startswith("gemma"):
            # Bound generation so a model that "runs away" past the JSON doesn't burn minutes; callers
            # size this generously so a real script never truncates. SKIP for gemma: its Ollama MLX
            # runner returns an EMPTY reply when num_predict is set (verified 2026-06-24) — the prompt
            # + natural stop bound it anyway. qwen3/others honour num_predict fine.
            options["num_predict"] = num_predict
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options,
        }
        if self.cfg.model.lower().lstrip().startswith("gemma"):
            # Gemma 4 (e.g. gemma4:12b-mlx) is a "thinking" model: left on, it spends most of its time
            # emitting reasoning we throw away (a draft call took 329s vs 51s with it off — 6.4× slower).
            # We only want the JSON answer, so disable it. Non-thinking models ignore this flag.
            payload["think"] = False
        if fmt and _supports_constrained_json(self.cfg.model):
            payload["format"] = fmt
        try:
            # (connect, read): fail fast (10s) if the daemon is down; bound the read per-model so a
            # restarted/stuck Ollama can't silently hang a build, while a heavy 16GB model still gets
            # enough time to cold-load + generate (see _read_timeout).
            r = requests.post(f"{self.cfg.host}/api/chat", json=payload,
                              timeout=(10, _read_timeout(self.cfg.model)))
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Ollama request failed ({e}). Is the daemon running and '{self.cfg.model}' pulled?"
            ) from e
        return r.json()["message"]["content"]


def unload(cfg: LLMConfig) -> None:
    """Ask Ollama to evict the model from RAM (keep_alive=0) and WAIT until it's actually gone.
    Frees several GB before the ffmpeg-heavy assemble — ffmpeg SIGSEGVs under memory pressure.
    The eviction is async, so we must block until /api/ps no longer lists it, otherwise the
    caller starts ffmpeg while the RAM is still held. Best-effort; reloads on the next request."""
    import time
    try:
        requests.post(f"{cfg.host}/api/generate",
                      json={"model": cfg.model, "keep_alive": 0}, timeout=(5, 30))
    except requests.RequestException:
        return
    base = cfg.model.split(":")[0]
    for _ in range(24):                     # up to ~12 s for the model to leave RAM
        try:
            models = requests.get(f"{cfg.host}/api/ps", timeout=3).json().get("models", [])
        except requests.RequestException:
            return
        if not any(str(m.get("name", "")).startswith(base) for m in models):
            log.info("Ollama unloaded %s — RAM freed for assemble.", cfg.model)
            return
        time.sleep(0.5)
    log.warning("Ollama still reports %s loaded after wait — proceeding anyway.", cfg.model)


def unload_all(cfg: LLMConfig) -> None:
    """Evict EVERY currently-loaded Ollama model from RAM and wait until none remain. Frees the
    most RAM before the ffmpeg-heavy assemble (incl. unrelated models the user has loaded, e.g. a
    big custom one) so nothing competes for memory. Reversible: each reloads on its next request."""
    import time
    try:
        loaded = requests.get(f"{cfg.host}/api/ps", timeout=5).json().get("models", [])
    except requests.RequestException:
        return
    names = [str(m.get("name") or m.get("model") or "") for m in loaded]
    names = [n for n in names if n]
    if not names:
        return
    for n in names:
        try:
            requests.post(f"{cfg.host}/api/generate", json={"model": n, "keep_alive": 0}, timeout=(5, 30))
        except requests.RequestException:
            pass
    for _ in range(24):                     # up to ~12 s for RAM to clear
        try:
            still = requests.get(f"{cfg.host}/api/ps", timeout=3).json().get("models", [])
        except requests.RequestException:
            return
        if not still:
            log.info("Ollama unloaded all models (%s) — RAM freed for assemble.", ", ".join(names))
            return
        time.sleep(0.5)
    log.warning("Ollama still reports models loaded after wait — proceeding anyway.")


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # qwen3 reasoning traces
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _norm_keywords(raw) -> list[str]:
    """Coerce a model's `keywords` field into a clean list. The model occasionally returns a
    bare string ("Mars, Saturn") instead of a list — splitting on chars would be wrong — or
    sneaks a None into the list, which would otherwise leak a literal 'None' keyword."""
    if isinstance(raw, str):
        raw = re.split(r"[,;]", raw)
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(k).strip() for k in raw if k is not None and str(k).strip()]


def _segment_dicts(data) -> list[dict]:
    """Pull the segment list out of model JSON defensively: tolerate a non-dict top level,
    `segments` given as a dict instead of a list, and stray non-dict entries."""
    if not isinstance(data, dict):
        return []
    segs = data.get("segments")
    if isinstance(segs, dict):
        segs = list(segs.values())
    if not isinstance(segs, list):
        return []
    return [s for s in segs if isinstance(s, dict)]


LANG_NAME = {"en": "English", "it": "Italian"}


def _draft_script_json(client: "OllamaClient", system: str, user: str,
                       temperature: float = 0.85, attempts: int = 4,
                       num_predict: int | None = None) -> dict:
    """Get a script-shaped JSON draft, retrying transient duds. Local models — notably gemma4 in
    free-text mode (it can't use Ollama's constrained JSON; see _supports_constrained_json) —
    intermittently return an EMPTY or unparseable reply. Each call is independent, so a retry
    self-heals; we accept the first reply that parses to a dict with at least one segment."""
    last = "no attempts"
    for i in range(max(1, attempts)):
        try:
            raw = client.chat(system, user, temperature=temperature, num_predict=num_predict)
            if raw and raw.strip():
                data = _extract_json(raw)
                if isinstance(data, dict) and _segment_dicts(data):
                    if i:
                        log.info("Script draft succeeded on attempt %d/%d.", i + 1, attempts)
                    return data
                last = "parsed JSON had no usable segments"
            else:
                last = "empty reply"
        except Exception as e:  # noqa: BLE001 — incl. a request timeout (cold model reload): retry warm
            last = f"call/parse failed ({e})"
        log.warning("Script draft attempt %d/%d unusable (%s) — retrying.", i + 1, attempts, last)
    raise RuntimeError(
        f"Model returned no usable script after {attempts} attempts ({last}). "
        "Try re-running or a different model.")


JUDGE_SYSTEM = (
    "You are the editor-in-chief of a top astronomy/space Shorts channel choosing which of several "
    "drafts of the SAME video to publish. Decide by ONE question: which draft would make a stranger "
    "stop scrolling in the first second and stay to the last line? Rank by wonder, curiosity, surprise "
    "and escalation — the draft that reads like a great campfire story beats the draft that reads "
    "like a textbook, every time; a draft that reaches for gore or dread (corpses, tombs, screaming) "
    "instead of awe loses. The rules (hook not a measurement, one fact per segment, no repetition, "
    "no invented numbers, correct tense, no false superlatives, no morbid lexicon) DISQUALIFY a "
    "draft that breaks them; "
    "they earn no points for being obeyed. Never pick a draft because it 'follows the constraints' — "
    "if two drafts both obey, pick the one with more nerve. Be decisive. Return STRICT JSON only: "
    '{"best": <1-based draft number>, "why": "one short sentence about what grips, not what complies"}.'
)


EXEMPLAR_PHRASES = ("younger than the dinosaurs", "rains gasoline", "three earths wide",
                    "quietly leaving us", "driving on mars right now", "the sun has never once lit",
                    "graveyard of shattered moons", "planetary autopsy", "suicide mission",
                    "scream into the void", "screaming into the void")   # the old voice, still off-limits   # "into the void" alone matched honest lines


MORBID_WORDS = ("corpse", "cadaver", "dead", "dying", "died", "death", "kill", "killed", "murder",
                "tomb", "grave", "graveyard", "autopsy", "hemorrhage", "haemorrhage", "bleed", "bleeding",
                "wound", "torture", "suicide", "hell", "hellscape", "nightmare", "scream", "screaming",
                "butcher", "prison", "prisoner", "gutted", "rot", "rotting", "decay", "doom", "apocalypse")
_MORBID_RE = None


def morbid_word(text: str) -> str | None:
    """The first banned morbid word in `text`, whole-word, or None. The audience found the earlier
    voice grim rather than gripping — corpses, hemorrhages, tombs — and asked for wonder instead."""
    global _MORBID_RE
    if _MORBID_RE is None:
        _MORBID_RE = re.compile(r"\b(" + "|".join(map(re.escape, MORBID_WORDS)) + r")\b", re.IGNORECASE)
    m = _MORBID_RE.search(text or "")
    return m.group(1).lower() if m else None


def morbid_in_script(data: dict) -> str | None:
    for seg in _segment_dicts(data):
        w = morbid_word(str(seg.get("narration", "")))
        if w:
            return w
    return morbid_word(str(data.get("cta_bridge", "")))


def copied_exemplar(data: dict) -> str | None:
    """The first exemplar phrase found verbatim in a draft, or None. The prompt shows the channel's
    best lines as the voice to match and says never to copy them; within an hour the model had
    written "waiting for a catalyst to scream into the void" about Titan. A copied line is not
    voice, it is plagiarism of ourselves, and it is usually false for the new subject."""
    text = " ".join(str(seg.get("narration", "")) for seg in _segment_dicts(data)).lower()
    return next((ph for ph in EXEMPLAR_PHRASES if ph in text), None)


def _judge_best(client: "OllamaClient", drafts: list[dict], topic: str, language: str) -> dict:
    """Pick the single best of several drafts by the channel rubric. Falls back to the first draft on
    any failure — judging must never lose a usable script. A draft that copied one of the exemplar
    lines is disqualified before the judge sees it (unless every draft did)."""
    clean = [d for d in drafts if not copied_exemplar(d)]
    if clean and len(clean) < len(drafts):
        log.info("Best-of: %d draft(s) copied an exemplar line — disqualified.", len(drafts) - len(clean))
        drafts = clean
    if len(drafts) <= 1:
        return drafts[0]
    listing = "\n\n".join(f"DRAFT {i + 1}:\n{json.dumps(d, ensure_ascii=False)}"
                          for i, d in enumerate(drafts))
    user = (f"Topic: {topic}\nLanguage: {LANG_NAME.get(language, 'English')}\n"
            f"Choose the single best draft to publish.\n\n{listing}")
    try:
        verdict = _extract_json(client.chat(JUDGE_SYSTEM, user, temperature=0.2, num_predict=256))
        idx = int(verdict.get("best", 1)) - 1
        if 0 <= idx < len(drafts):
            log.info("Best-of-%d: judge picked draft %d — %s",
                     len(drafts), idx + 1, str(verdict.get("why", ""))[:90])
            return drafts[idx]
    except Exception as e:  # noqa: BLE001 — a bad judge must never lose a usable draft
        log.warning("Draft judge failed (%s) — keeping draft 1.", e)
    return drafts[0]


LENGTH_FLOOR = 0.80          # below this the video is thin — but padding to reach it is worse
LENGTH_HEADROOM_S = 2.0      # seconds of slack above the target before a draft counts as too long


def length_verdict(words_now: int, target: int, wps: float = 2.33) -> str:
    """"ok" | "long" | "short" — whether a draft needs a length pass at all.

    The band is DELIBERATELY not symmetric, and the reason is measured. A short script is a weaker
    video; a long one is a video that breaks the 60s rule the channel publishes under, so the two
    failures do not cost the same and must not get the same tolerance. A symmetric ±12% band put the
    ceiling at 132 words, which renders 64s.

    Above the floor it is expressed in SECONDS, not percent, because that is the thing the ceiling is
    really about: `LENGTH_HEADROOM_S` past the target, converted at the SLOWEST rate observed, since
    underestimating duration is the direction that breaks the ceiling. Measured across three builds
    the delivery runs 2.33-2.61 words/s — a ~6% spread the word count cannot see, which is why the
    target aims below the limit instead of at it."""
    if target <= 0:
        return "ok"
    if words_now < target * LENGTH_FLOOR:
        return "short"
    return "long" if words_now > target + LENGTH_HEADROOM_S * wps else "ok"


def moves_closer(new: int, cur: int, target: int) -> bool:
    """Whether a rewrite is worth keeping. "Longer than before" is not progress: asked to lengthen a
    92-word draft toward 118, the model returned 222 — further from the target than it started, and
    accepted anyway, because the old test was only `new > cur`."""
    return abs(new - target) < abs(cur - target)


# --------------------------------------------------------------------------- per-segment fit
SEGMENT_FIT_SYSTEM = (
    "You are the line editor of a faceless astronomy short-form channel. You rewrite ONE segment of "
    "a script to an exact spoken length WITHOUT losing its nerve. Keep its single fact, its tense, "
    "its numbers, its role in the story and its voice — a claim that surprises, one vivid image, never "
    "a morbid word (corpse, tomb, hemorrhage, scream, hell, dying). To "
    "SHORTEN, cut the weakest clause, never the stakes. To LENGTHEN, add one concrete consequence or "
    "image the line was missing — never restate, never open with This/These/It, never use ensuring/"
    "allowing/creating. If it is segment 1 it is the HOOK: it renames the subject as something "
    "unsettling in its first words and never opens with a number or the subject's name. Spoken, "
    "concrete, no filler, no banned words (mind-blowing, incredible, literally, breathtaking, "
    "journey, unlock, delve, 'did you know'). Return STRICT JSON only: {\"narration\": \"...\"}."
)
SEGMENT_FIT_TOLERANCE = 0.30     # a segment within ±30% of its budget is left alone: fewer rewrites, more voice
SEGMENT_FIT_ATTEMPTS = 2


def _seg_words(seg: dict) -> int:
    return len(str(seg.get("narration", "")).split())


def segment_budgets(words: int, nseg: int) -> list[int]:
    """Per-segment word budgets summing to `words`. The hook (segment 1) is kept a touch shorter —
    it has to land in a breath — and the difference goes to the penultimate 'wow' segment."""
    if nseg <= 0:
        return []
    base = words // nseg
    out = [base] * nseg
    for i in range(words - base * nseg):          # spread the remainder from the end
        out[-1 - (i % nseg)] += 1
    if nseg >= 3:
        shave = max(1, base // 6)
        out[0] -= shave
        out[-2] += shave
    return out


def _fit_one(client: "OllamaClient", seg: dict, budget: int, index: int, nseg: int,
             prev_line: str, next_line: str, language: str) -> dict:
    """Rewrite one segment to `budget` words. Local, exact, verified: the model is asked for a
    NUMBER of words, given the neighbours so the hand-off survives, and the reply is counted. Up to
    SEGMENT_FIT_ATTEMPTS tries; the closest reply wins, and the original stays if nothing beats it."""
    best, best_d = seg, abs(_seg_words(seg) - budget)
    name = LANG_NAME.get(language, "English")
    for _ in range(SEGMENT_FIT_ATTEMPTS):
        cur = _seg_words(best)
        user = (f"Segment {index} of {nseg}. It has {cur} words; rewrite it to EXACTLY {budget} words "
                f"({'longer' if budget > cur else 'shorter'} by {abs(budget - cur)}). Count them.\n"
                f"Write in {name}.\n"
                f"Previous segment: {prev_line or '(none — this is the opening hook)'}\n"
                f"Next segment: {next_line or '(none — this is the last content segment)'}\n"
                f"Segment to rewrite: {best.get('narration', '')}")
        try:
            data = _extract_json(client.chat(SEGMENT_FIT_SYSTEM, user, temperature=0.5,
                                             num_predict=256))
            line = str(data.get("narration", "")).strip() if isinstance(data, dict) else ""
        except Exception as e:  # noqa: BLE001 — a failed fit keeps what we have
            log.debug("segment %d fit failed (%s)", index, e)
            line = ""
        if not line:
            continue
        cand = {**best, "narration": line}
        d = abs(_seg_words(cand) - budget)
        if d < best_d:
            best, best_d = cand, d
        if d <= max(1, round(budget * SEGMENT_FIT_TOLERANCE)):
            break
    return best


def fit_segments(client: "OllamaClient", data: dict, words: int, language: str) -> dict:
    """Bring a script to length ONE SEGMENT AT A TIME.

    Whole-script editing does not work on this model — measured: asked to lengthen it ignores the
    target and doubles what it has (77->152, 90->172, 100->211), asked to cut a quarter it shaves
    1-3%. So the total drifted anywhere from 47s to 64s of video. Small local edits are where a
    language model actually complies, and a total that is the sum of verified parts cannot drift:
    each segment gets a budget, the ones outside ±20% are rewritten to an exact count with their
    neighbours in view, and the reply is counted before it is accepted."""
    segs = _segment_dicts(data)
    if not segs:
        return data
    budgets = segment_budgets(words, len(segs))
    out, before = [], sum(_seg_words(x) for x in segs)
    # Only move in the direction the WHOLE script needs. Fitting every segment to its own budget
    # once turned a 115-word draft (in band) into 121 (over the ceiling): the under-budget segments
    # were lengthened while the over-budget ones sat inside their tolerance. A script already in
    # band is left alone; a short one only ever grows; a long one only ever shrinks.
    verdict = length_verdict(before, words)
    if verdict == "ok":
        return data
    for i, seg in enumerate(segs):
        budget = budgets[i]
        prev_line = str(segs[i - 1].get("narration", "")) if i else ""
        next_line = str(segs[i + 1].get("narration", "")) if i + 1 < len(segs) else ""
        n = _seg_words(seg)
        needs = (verdict == "short" and n < budget) or (verdict == "long" and n > budget)
        if not needs or abs(n - budget) <= max(1, round(budget * SEGMENT_FIT_TOLERANCE)):
            out.append(seg)
            continue
        out.append(_fit_one(client, seg, budget, i + 1, len(segs), prev_line, next_line, language))
    data = {**data, "segments": out}
    log.info("Per-segment fit: %d → %d words (target ~%d, %d segments).",
             before, sum(_seg_words(x) for x in out), words, len(out))
    return data


def _reword_copied(client: "OllamaClient", data: dict, phrase: str, language: str) -> dict:
    """Rewrite only the segment(s) containing a copied exemplar phrase, keeping their fact and length."""
    segs = _segment_dicts(data)
    out = []
    bridge = str(data.get("cta_bridge", "") or "")
    if phrase in bridge.lower():
        try:
            cand = _extract_json(client.chat(SEGMENT_FIT_SYSTEM,
                f"This closing line must lose the word \"{phrase}\" (too morbid for a channel whose tone is "
                f"wonder). Rewrite it in about {len(bridge.split())} words, same meaning, no morbid word. "
                f"Write in {LANG_NAME.get(language, 'English')}.\nLine: {bridge}", temperature=0.7, num_predict=128))
            nb = str(cand.get("narration", "")).strip() if isinstance(cand, dict) else ""
            if nb and phrase not in nb.lower():
                data = {**data, "cta_bridge": nb}
        except Exception as e:  # noqa: BLE001
            log.debug("reword of the bridge failed (%s)", e)
    for i, seg in enumerate(segs):
        line = str(seg.get("narration", ""))
        if phrase not in line.lower():
            out.append(seg); continue
        user = (f"This line must lose the word or phrase \"{phrase}\" — it is either copied from another "
                f"video on the channel or too morbid for the channel's tone, which is wonder and "
                f"curiosity, never gore or dread. Rewrite the line in about {len(line.split())} words, "
                f"keeping its fact, its surprise and its role, with an image of your own; the word must "
                f"not appear and no other morbid word (corpse, tomb, hemorrhage, scream, hell, dying) may "
                f"replace it. Write in {LANG_NAME.get(language, 'English')}.\nLine: {line}")
        try:
            cand = _extract_json(client.chat(SEGMENT_FIT_SYSTEM, user, temperature=0.7, num_predict=256))
            new = str(cand.get("narration", "")).strip() if isinstance(cand, dict) else ""
        except Exception as e:  # noqa: BLE001 — a failed rewrite keeps the line; the caller warns
            log.debug("reword of segment %d failed (%s)", i + 1, e); new = ""
        out.append({**seg, "narration": new} if new and phrase not in new.lower() else seg)
    return {**data, "segments": out}


def generate_script(cfg: LLMConfig, topic: str, seconds: int = 60, language: str = "en",
                    refine_passes: int = 1, best_of: int = 1, images_per_segment: int = 2,
                    fit: str = "whole") -> Script:
    words = _words_for(seconds, language)
    # One segment ≈ one shot, and this number is the whole cut rhythm. A tight upper bound (nseg+1)
    # keeps the model from padding to twice the length.
    #
    # Measured across four real builds. At ~10s per segment a 50s video was FOUR shots and each image
    # sat on screen for nearly seven seconds: documentary pacing, dead on a feed. Overcorrecting to
    # ~5s produced nine shots and 2.4s per image, which a viewer reads as the video being played at
    # 1.5x — the eye never settles, and the writing is squeezed into one-line beats that give an idea
    # no room. The speech itself was never fast: 2.38 words/second in every build, matching the
    # budget below. It was the cutting.
    #
    # Expressed as the thing we actually care about — how long one PICTURE stays on screen — rather
    # than a magic segment count, so it stays correct if images_per_segment ever changes.
    # Floor of 3, not more: a story needs a beginning, a turn and a landing. A higher floor looks
    # protective but fights the rhythm on short videos — at 4 a 30s cut is forced to 2.75s per image,
    # back into the territory viewers read as sped up.
    nseg = max(3, round(seconds / (SECONDS_PER_IMAGE * max(1, images_per_segment))))
    nseg2 = nseg + 1
    # ASK for more than the target, because the writer reliably delivers less. Measured across eleven
    # real builds it returns a MEDIAN 88% of the per-segment budget it is given (range 67-108%), so
    # asking for exactly the target ships a video that is systematically short: Olympus Mons came
    # back at 82% and rendered 42.7s against a 58s target. 1/0.88 is the compensation, and it is
    # applied ONLY to what the writer is told — `words` stays the true target, so the length guards
    # below still measure the draft against the length we actually want.
    wseg = max(8, round(words * ASK_INFLATION / nseg))
    user = USER_TMPL.format(topic=topic, seconds=seconds, words=round(words * ASK_INFLATION),
                            nseg=nseg, nseg2=nseg2, wseg=wseg)
    name = LANG_NAME.get(language, "English")
    system = SYSTEM + f"\n- Write ALL narration in {name}."
    client = OllamaClient(cfg)
    # Generation caps sized with wide margin so a real script never truncates, while a runaway model
    # (gemma free-text can append prose past the JSON) can't burn minutes. The JSON output (narration
    # + visual + keywords + syntax) is a few× the narration words → words*8 with a 1536 floor.
    script_cap = max(1536, words * 8)
    n_draft = max(1, best_of)
    log.info("Generating %s script for %r with %s (best-of-%d) ...", name, topic, cfg.model, n_draft)
    # Generate N diverse drafts (temperature spread for variety), then keep the LLM-judged best — the
    # single biggest quality lever for the one model we run. Each draft is independently retried.
    drafts: list[dict] = []
    for i in range(n_draft):
        temp = 0.85 if n_draft == 1 else round(0.8 + 0.12 * i, 2)
        try:
            drafts.append(_draft_script_json(client, system, user, temperature=temp, num_predict=script_cap))
        except RuntimeError as e:
            log.warning("Draft %d/%d unusable (%s).", i + 1, n_draft, e)
    if not drafts:
        raise RuntimeError("Model returned no usable script. Try re-running or a different model.")
    data = _judge_best(client, drafts, topic, language)

    for n in range(max(0, refine_passes)):     # critique → refine; never regress a usable draft
        try:
            draft = json.dumps(data, ensure_ascii=False)
            critique = client.chat(CRITIQUE_SYSTEM, CRITIQUE_USER.format(script=draft),
                                   fmt=None, temperature=0.6, num_predict=1024)
            refined = _extract_json(client.chat(
                system + REFINE_SUFFIX, REFINE_USER.format(script=draft, critique=critique),
                temperature=0.4, num_predict=script_cap))
            if [s for s in _segment_dicts(refined) if str(s.get("narration", "")).strip()]:
                data = refined
                log.info("Script refine pass %d/%d applied.", n + 1, refine_passes)
        except Exception as e:  # noqa: BLE001 — refinement must never break a usable draft
            log.warning("Script refine pass %d failed (%s) — keeping current draft.", n + 1, e)
            break

    def _nwords(d: dict) -> int:
        return sum(len(str(s.get("narration", "")).split()) for s in _segment_dicts(d))
    # Bring the draft into the band [0.88, 1.12] x target. This used to be two one-shot guards, trim
    # then expand, on the theory that a script is never both. It becomes both the moment an expansion
    # overshoots: asked to lengthen a 92-word draft toward 118, the model returned 222, and with trim
    # already behind it nothing pulled it back — a 93s narration against a 58s target.
    #
    # So: loop, and accept a rewrite ONLY when it moves TOWARD the target. "Longer than before" is not
    # progress if it overshoots by more than the original undershot, and that test is what the old
    # `_nwords(expanded) > cur` was missing. Two passes is enough to go long-then-short or the reverse;
    # Three attempts, not two: a rejected rewrite now re-rolls instead of giving up, so the budget has
    # to cover "overshoot, re-roll, land" without letting a hopeless draft burn the whole night.
    if (fit or "whole").lower() == "segmentwise":
        data = fit_segments(client, data, words, language)
    best = data
    for _ in range(3 if (fit or "whole").lower() != "segmentwise" else 0):
        cur = _nwords(data)
        verdict = length_verdict(cur, words)
        if verdict == "ok":
            best = data
            break
        long_draft = verdict == "long"
        if long_draft:
            fix_user = (
                f"This script is TOO LONG: ~{cur} spoken words, but the target is ~{words} words across "
                f"{nseg}-{nseg2} segments. Tighten it to the target WITHOUT losing facts: cut filler, "
                f"redundant qualifiers and any restated idea, and shorten sentences. Keep the hook, keep "
                f"every distinct fact, keep the same JSON shape.\n\nCurrent JSON:\n"
                + json.dumps(data, ensure_ascii=False))
        else:
            fix_user = (
                f"This script is TOO SHORT: ~{cur} spoken words, but the target is ~{words} words across "
                f"{nseg}-{nseg2} segments. Land BETWEEN {int(words * LENGTH_FLOOR)} and "
                f"{int(words + LENGTH_HEADROOM_S * 2.33)} words — a draft above that ceiling is rejected exactly "
                f"like one below the floor, and overshooting is the more common failure. Aim for "
                f"~{words}. Extend each narration to two full "
                f"sentences (the fact + its consequence/scale/image) and/or add segments with NEW distinct "
                f"real facts (history, mechanism, scale, discovery, what's next). Keep the same hook, zero "
                f"repetition, zero filler. Return the STRICT JSON only.\n\nCurrent draft:\n"
                + json.dumps(data, ensure_ascii=False))
        try:
            cand = _extract_json(client.chat(system, fix_user,
                                             temperature=0.4 if long_draft else 0.6,
                                             num_predict=script_cap))
        except Exception as e:  # noqa: BLE001 — a failed pass just keeps the draft we have
            log.warning("Length guard (%s) failed (%s) — keeping the closest draft.",
                        "trim" if long_draft else "expand", e)
            break
        if not (isinstance(cand, dict) and _segment_dicts(cand)):
            log.warning("Length guard returned nothing usable — keeping the closest draft.")
            break
        # ALWAYS feed the candidate forward, even when it overshot, and keep the closest draft seen
        # separately. Rejecting an overshoot outright was the mistake: measured over three re-rolls
        # this model expands 90 words to 172, 175, 181 — a consistent ~+90% bias, not noise, so
        # re-rolling never lands in band. But it TRIMS accurately (142→134, 140→136), so the route
        # that works is expand-then-trim, and refusing the overshoot is exactly what blocked it.
        # `best` is the safety net: a pass that makes things worse can never be what we ship.
        new = _nwords(cand)
        log.info("Length guard: %s %d → %d words (target ~%d).",
                 "trimmed" if long_draft else "expanded", cur, new, words)
        data = cand
        if moves_closer(new, _nwords(best), words):
            best = cand

    data = best
    # Tone guard: any morbid word is reworded the way a copied exemplar is — same fact, wonder instead.
    for _ in range(3):
        mw = morbid_in_script(data)
        if not mw:
            break
        data = _reword_copied(client, data, mw, language)
        if morbid_in_script(data) == mw:            # the rewrite kept it: stop trying, warn below
            break
    if (mw := morbid_in_script(data)):
        log.warning("Script still contains the morbid word %r after rewrites — fix it in script.md.", mw)
    if (ph := copied_exemplar(data)):
        # A warning in an unattended run is a warning nobody reads: the Moon script went to render
        # with "it's a planetary autopsy" in it. Repair it — one targeted rewrite of the offending
        # segment(s), same fact, own words — and only warn if that also fails.
        data = _reword_copied(client, data, ph, language)
        if (ph2 := copied_exemplar(data)):
            log.warning("Script still reuses the exemplar line %r after a rewrite — reword it in "
                        "script.md before building.", ph2)
        else:
            log.info("Script had copied the exemplar line %r — reworded.", ph)
    if length_verdict(_nwords(data), words) != "ok":
        log.warning("Length guard: settled at %d words against a target of ~%d — the video will be "
                    "off-length.", _nwords(data), words)

    segments = [
        Segment(
            index=i + 1,
            narration=str(s.get("narration", "")).strip(),
            visual=str(s.get("visual", "")).strip(),
            keywords=_norm_keywords(s.get("keywords")),
        )
        for i, s in enumerate(_segment_dicts(data))
        if str(s.get("narration", "")).strip()
    ]
    before = len(segments)
    segments = dedupe_segments(segments)          # drop the model's repeated 'payoff' lines
    if len(segments) < before:
        log.info("Removed %d duplicate segment(s) from the generated script.", before - len(segments))
    if not segments:
        raise RuntimeError("Model returned no usable segments. Try re-running or a different model.")
    title = data.get("title") if isinstance(data.get("title"), (str, int, float)) else topic
    bridge = data.get("cta_bridge") if isinstance(data.get("cta_bridge"), str) else ""
    kind = data.get("bridge_kind") if isinstance(data.get("bridge_kind"), str) else ""
    kind = kind.strip().lower()
    if kind not in ("shoot", "principle", "none"):
        kind = ""                            # unknown/absent → callers fall back to the generic line
    return Script(title=str(title or topic).strip(), segments=segments,
                  target_seconds=seconds, topic=topic, cta_bridge=bridge.strip(),
                  bridge_kind=kind)


def translate_segments(cfg: LLMConfig, texts: list[str], target_lang: str) -> list[str]:
    """Translate each narration line into target_lang for on-screen subtitles (local Ollama).
    Returns a same-length list; on any failure falls back to the source text."""
    name = LANG_NAME.get(target_lang, target_lang)
    items = [{"id": i, "text": t} for i, t in enumerate(texts)]
    system = f"You are a professional {name} subtitle translator. Return STRICT JSON only."
    user = (f"Translate each item's text into natural, fluent {name} for on-screen video subtitles "
            f"(concise spoken register; keep proper nouns and numbers). Return STRICT JSON exactly: "
            f'{{"items":[{{"id":0,"text":"..."}}]}}\n\nItems:\n{json.dumps(items, ensure_ascii=False)}')
    try:
        data = _extract_json(OllamaClient(cfg).chat(system, user, temperature=0.3))
        out = {int(it["id"]): str(it.get("text", "")).strip() for it in data.get("items", [])}
        return [out.get(i) or texts[i] for i in range(len(texts))]
    except Exception as e:  # noqa: BLE001
        log.warning("Subtitle translation failed (%s) — using source text.", e)
        return list(texts)


META_SYSTEM = (
    "You write platform-optimized metadata for a faceless astronomy/space short-form video. "
    "Be accurate and non-clickbait. Use fluent, correctly spelled prose with standard apostrophes "
    "(Earth's, don't) — never invent a word or split one with a stray apostrophe. "
    "Return STRICT JSON only."
)

META_USER = """Title: {title}
Narration: {narration}

Promote this app where natural (in descriptions/captions): {app} — {tagline} ({url}).
Return JSON exactly:
{{
  "youtube": {{"title": "punchy title <=80 chars", "description": "2-3 sentence summary, then a new line: 'Get {app}: {url}'", "tags": ["10-15 short lowercase tags"]}},
  "tiktok": {{"caption": "one-line hook + 4-6 hashtags including #astronomy #space"}},
  "instagram": {{"caption": "one-line hook, then a blank line, then the hashtags"}},
  "instagram_hashtags": ["12-15 tags, LAYERED, see below"]
}}

The Instagram hashtags decide whether anyone outside the followers ever sees the reel, and a flat
list of huge generic tags is the one arrangement that guarantees they will not: a new account cannot
rank in #space, so those tags are decoration. Build FOUR tiers instead, in this order:
  - 2-3 BROAD (#astronomy #space) — context for the algorithm, not reach,
  - 4-5 MID, specific to this video's subject and sized where an account can actually place
    (#planetaryscience #marsexploration #solarsystem),
  - 4-5 NARROW, the exact object and mission by name (#vallesmarineris #marsreconnaissanceorbiter) —
    small audiences, but this is the only tier where the reel can reach the top of a feed,
  - 1-2 COMMUNITY where the people who own telescopes actually gather (#astrophotography
    #backyardastronomy), because this channel exists to reach them.
All lowercase, no spaces, no duplicates, no banned or engagement-bait tags (#followforfollow, #f4f,
#viral, #fyp). Every NARROW tag must name something the narration actually discussed."""


# A stray apostrophe INSIDE a word that isn't a real English contraction (gemma once wrote "Earth'ally"
# for "Earthly") is a glitched generation. We can't guess the intended word, so we DETECT it and re-roll
# the (stochastic) generation rather than ship broken text. Valid contraction suffixes are whitelisted.
_CONTRACTIONS = {"s", "t", "d", "m", "re", "ve", "ll"}
_APOSTROPHE_WORD = re.compile(r"[A-Za-z]+['’]([A-Za-z]+)")


def _clean_text(s):
    """Safe, deterministic tidy-ups only (collapse runs of spaces, trim) — never alter wording."""
    return re.sub(r"[ \t]{2,}", " ", s).strip() if isinstance(s, str) else s


def _meta_texts(data: dict) -> list[str]:
    yt = data.get("youtube") or {}
    return [yt.get("title", ""), yt.get("description", ""),
            (data.get("tiktok") or {}).get("caption", ""),
            (data.get("instagram") or {}).get("caption", "")]


def _meta_looks_clean(data: dict) -> bool:
    """False if any text field has a stray-apostrophe word (e.g. "Earth'ally") → caller re-rolls."""
    return all(suffix.lower() in _CONTRACTIONS
               for txt in _meta_texts(data)
               for suffix in _APOSTROPHE_WORD.findall(txt or ""))


def _clean_metadata(data: dict) -> dict:
    yt = data.get("youtube")
    if isinstance(yt, dict):
        for k in ("title", "description"):
            if k in yt:
                yt[k] = _clean_text(yt[k])
    for plat in ("tiktok", "instagram"):
        d = data.get(plat)
        if isinstance(d, dict) and "caption" in d:
            d["caption"] = _clean_text(d["caption"])
    _merge_instagram_hashtags(data)
    _ensure_brand_tag(data)
    return data


_TAG = re.compile(r"#\w+")


BRAND_TAG = "#astrostackerpro"


def _ensure_brand_tag(data: dict) -> None:
    """Every platform's tag list carries the brand, whatever the model produced."""
    tt = data.get("tiktok")
    if isinstance(tt, dict):
        tags = [str(t).strip().lower() for t in (tt.get("hashtags") or []) if str(t).strip()]
        cap = tt.get("caption", "") or ""
        if BRAND_TAG not in tags and BRAND_TAG not in cap.lower():
            tags.append(BRAND_TAG)
            tt["caption"] = f"{cap.rstrip()} {BRAND_TAG}".strip()
        tt["hashtags"] = tags
    yt = data.get("youtube")
    if isinstance(yt, dict):
        tags = [str(t).strip().lower() for t in (yt.get("tags") or []) if str(t).strip()]
        if BRAND_TAG.lstrip("#") not in tags:
            tags.append(BRAND_TAG.lstrip("#"))          # YouTube tags carry no '#'
        yt["tags"] = tags


def _merge_instagram_hashtags(data: dict) -> None:
    """Fold the layered tag list into the Instagram caption, deterministically.

    The model returns the tiers as their own JSON array, which is the shape it gets right; asking it
    to also lay them out inside the caption is the part it does not. So the layout is done here —
    caption, blank line, tags — and publish.py keeps reading a single `caption` field.

    Any tags the model put inline anyway are stripped first, so a tag cannot appear twice, and the
    list is deduplicated case-insensitively while keeping tier order: the narrow tags are the ones
    that can actually rank, and they must not be the ones dropped at the 30-tag ceiling.
    """
    ig = data.get("instagram")
    tags = data.pop("instagram_hashtags", None)
    if not isinstance(ig, dict):
        return
    if not isinstance(tags, list):
        tags = _TAG.findall(ig.get("caption", "") or "")   # salvage whatever it put inline
    seen, ordered = set(), []
    for t in tags:
        t = str(t).strip().lower()
        if not t:
            continue
        t = "#" + _TAG.sub("", t).lstrip("#").strip() if not t.startswith("#") else t
        t = "#" + "".join(ch for ch in t[1:] if ch.isalnum() or ch == "_")
        if len(t) > 1 and t not in seen:
            seen.add(t)
            ordered.append(t)
    # The brand tag is a hard requirement on every post and is added HERE, not asked of the model:
    # a prompt is advice and this must not depend on it. Appended last so it never displaces a
    # narrow tag at the 30-tag ceiling, and skipped if the model already put it in.
    if BRAND_TAG not in seen:
        ordered.append(BRAND_TAG)
    body = _TAG.sub("", ig.get("caption", "")).strip()          # drop any inline tags
    body = re.sub(r"\s{2,}", " ", body).strip(" \n")
    ig["caption"] = f"{body}\n\n{' '.join(ordered[:30])}"      # Instagram's own ceiling is 30
    ig["hashtags"] = ordered[:30]


BRAINSTORM_SYSTEM = (
    "You are a content strategist for a faceless {theme} short-form video channel. Propose fresh, "
    "SPECIFIC topics — a concrete object, mission, event, person, or phenomenon, each standing on its "
    "own as a 45-60s explainer, NOT vague themes. Never repeat anything in the AVOID list. "
    "Return STRICT JSON only: a flat array of short topic strings."
)


def _norm_topic(s: str) -> str:
    """Normalize a topic for dedup: lowercase, alphanumerics only, collapsed spaces."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _extract_list(text: str) -> list:
    """Array-aware JSON extraction (brainstorm returns a top-level array, which _extract_json's
    object-only fallback would miss when the model wraps it in prose)."""
    try:
        data = _extract_json(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
    except Exception:  # noqa: BLE001 — fall through to the array regex
        pass
    m = re.search(r"\[.*\]", text, flags=re.S)
    return json.loads(m.group(0)) if m else []


def brainstorm_topics(cfg: LLMConfig, avoid, n: int, theme: str = "space and astronomy",
                      language: str = "en") -> list[str]:
    """Ask the LLM for `n` fresh, specific topics not in `avoid` (topics already queued or produced).
    Deduped against `avoid` AND within the batch. Returns up to `n` (fewer if the model repeats)."""
    name = LANG_NAME.get(language, "English")
    system = BRAINSTORM_SYSTEM.format(theme=theme) + f" Write the topics in {name}."
    avoid_list = "\n".join(f"- {a}" for a in list(avoid)[:200]) or "(none yet)"
    user = (f"Propose {n} NEW topics for the channel.\nAVOID (already covered or queued):\n{avoid_list}\n"
            f'Return JSON only: ["topic one", "topic two", …] — up to {n} items, each ≤ 8 words.')
    client = OllamaClient(cfg)
    seen = {_norm_topic(a) for a in avoid}
    out: list[str] = []
    for i in range(3):
        try:
            raw = client.chat(system, user, num_predict=512)
            for t in _extract_list(raw or ""):
                t = str(t).strip().strip('"').strip()
                key = _norm_topic(t)
                if t and key and key not in seen:
                    seen.add(key)
                    out.append(t)
                    if len(out) >= n:
                        return out
        except Exception as e:  # noqa: BLE001 — a bad batch shouldn't crash the daily run
            log.warning("brainstorm attempt %d/3 failed (%s) — retrying.", i + 1, e)
    return out


def generate_metadata(cfg: LLMConfig, script: Script, funnel: FunnelConfig, language: str = "en") -> dict:
    name = LANG_NAME.get(language, "English")
    user = META_USER.format(title=script.title, narration=script.narration,
                            app=funnel.app_name, tagline=funnel.tagline, url=funnel.url)
    system = META_SYSTEM + f" Write titles, descriptions and captions in {name} (hashtags may stay English)."
    client = OllamaClient(cfg)
    log.info("Generating %s metadata with %s ...", name, cfg.model)
    last = "no attempts"                       # same gemma free-text flakiness as the script draft;
    fallback = None                            # last structurally-valid dict, used if all re-rolls glitch
    for i in range(4):                          # the metadata stage also COLD-RELOADS a 16GB model
        try:                                    # (assemble evicted it) → first call may time out, so
            raw = client.chat(system, user, num_predict=1024)   # catch it & retry warm (see _read_timeout)
            if raw and raw.strip():
                data = _extract_json(raw)
                if not (isinstance(data, dict) and data):
                    last = "empty/non-object JSON"
                elif not _meta_looks_clean(data):
                    fallback, last = data, "malformed text (stray apostrophe)"   # re-roll for clean prose
                else:
                    return _clean_metadata(data)
            else:
                last = "empty reply"
        except Exception as e:  # noqa: BLE001 — incl. a request timeout on the cold reload
            last = f"call/parse failed ({e})"
        log.warning("Metadata attempt %d/4 unusable (%s) — retrying.", i + 1, last)
    if fallback is not None:                    # don't fail the build over a typo — ship the best we got
        log.warning("Using last metadata despite %s.", last)
        return _clean_metadata(fallback)
    raise RuntimeError(f"Model returned no usable metadata after 4 attempts ({last}).")
