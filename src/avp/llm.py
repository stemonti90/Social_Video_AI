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
astronomy and space (TikTok / Reels / YouTube Shorts). These craft rules separate gripping from mediocre:
- HOOK: the first 6-8 words must be a concrete, counterintuitive or NUMBER-led statement that stops the scroll. NEVER open with a question, "Imagine", "Have you ever", "Picture this", or "In the vast expanse".
- Exactly ONE new, specific, verifiable fact per content segment (a named object, a number, a comparison, a scale) — then LAND it: a second sentence that gives the consequence, the scale, or a vivid concrete image. State-and-move-on is too thin.
- LENGTH: each content segment is ONE or TWO short spoken sentences, around the per-segment word count given below. Short beats are deliberate: each segment becomes its own shot, and a segment that runs long forces a single image to sit on screen too many seconds. HIT the requested total word count and segment count — coming in SHORT or running LONG are both failures.
- DRAMATURGY, not exposition. A short-form script is not "what it is → what it did → why it matters" — that is a documentary and it loses the scroll. Build instead: a fact that shouldn't be possible → why it shouldn't be possible → how it is possible anyway → what that means. Explain, never list disconnected trivia, but let the explanation ARRIVE as the answer to a tension you opened, not as a lecture delivered up front.
- Segment 1 must NOT begin with the subject's name, and must not announce what the subject is. It states the single strangest concrete thing in the whole script — the fact you would lead with if you had one sentence to stop someone scrolling. The viewer should work out what the video is about from that fact, not be told. ("Voyager 1 carries a golden record into the void" is still an announcement; "A machine 25 billion kilometres away is still talking to us" is a hook.)
- Every segment must leave the next one NECESSARY: end on a consequence, a number that begs a question, or an unresolved tension. If a segment could be the last one, it is written wrong.
- Every segment must make a DISTINCT point. NEVER repeat, restate or paraphrase an earlier line to reach the segment count — if you genuinely run out of distinct facts, broaden the angle (history, mechanism, scale, discovery, what's next) rather than repeating.
- Open a curiosity loop in the first 1-2 segments and PAY IT OFF before the end.
- Escalate to a single peak "wow" moment in the penultimate segment.
- Concrete nouns over adjectives; a confident, flowing spoken cadence (not terse, not padded). BANNED words/phrases: mind-blowing, incredible, literally, breathtaking, journey, unlock, delve, "did you know". No markdown, no emojis, no stage directions.
- Every fact must be real and checkable; if unsure, stay qualitative — never invent numbers.
- A NUMBER MUST MEASURE WHAT THE SENTENCE SAYS IT MEASURES. Distance, diameter, age, mass, temperature and speed are different quantities, and the most common error in astronomy writing is quoting an object's DISTANCE as its SIZE. The Orion Nebula is ~1,340 light-years AWAY and ~24 light-years ACROSS; writing "a 1,300 light-year wide nebula" is wrong by a factor of fifty. Before every figure, name to yourself which quantity it belongs to.
- SUPERLATIVES are where these scripts get things wrong most often. Before writing "the only", "the first", "the largest", "the farthest", check whether a second case exists — there usually is one (Voyager 2 reached interstellar space too; several probes have left the planets behind). If you cannot be certain, drop the superlative: the fact is almost always just as strong without it.
- Use METRIC units (kilometres). The audience reads subtitles in a metric country, and a distance in miles carries no sense of scale for them.
- Vary structure between videos; never sound template-stamped.
- "keywords": 2-4 ENGLISH, archive-catalogable PROPER NOUNS (e.g. "Cassini Saturn", "Carina Nebula JWST", "Curiosity rover Mars") — concrete subjects a NASA/Hubble search will find, matching that segment's visual.
Return STRICT JSON only, no commentary."""

CRITIQUE_SYSTEM = (
    "You are a ruthless short-form video editor. Grade this astronomy/space Shorts script 1-5 on: "
    "hook (stops the scroll in under 2s), fact density (one concrete verifiable fact per segment), "
    "a curiosity loop opened early and paid off, escalation to a single peak, concrete nouns over "
    "adjectives, ZERO cliche, and NO repeated or near-duplicate segments. Then list the 3 weakest "
    "lines with an exact rewrite for each, and explicitly flag any segment that repeats another. "
    "Then CHECK THE FACTS: for EVERY number, name the quantity it claims to measure and say "
    "whether it actually belongs to that quantity — an object's distance quoted as its width is "
    "the most common failure. Flag every superlative (only/first/largest/farthest) that is false "
    "or unverifiable, every number that looks invented, and any distance not in kilometres. "
    "Be specific and brutal. Plain text, no JSON."
)
CRITIQUE_USER = "Script JSON:\n{script}"
REFINE_SUFFIX = (
    "\n- You are now REVISING an existing draft to satisfy an editor's critique: land the hook in "
    "the first 6-8 words, raise fact density, kill every cliche, keep the curiosity loop paid off, "
    "and make every segment DISTINCT — merge or cut any repeated/near-duplicate lines. "
    "Keep the SAME JSON shape and a similar segment count."
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
- "keywords": 2-4 ENGLISH search keywords for space archives (they are English-indexed).

Also provide a punchy "title", plus the BRIDGE to astrophotography — this channel exists to make people
want to point their own camera at the sky, so every video has to land somewhere real. Two fields:

- "bridge_kind": choose HONESTLY from exactly one of these:
    "shoot"     — this subject, or the very sky it sits in, can actually be photographed from a balcony
                  tonight with a phone (the Moon, Saturn, Jupiter, Orion, Andromeda, the ISS, an
                  eclipse, aurorae, the Milky Way, a meteor shower, a comet).
    "principle" — the subject itself is out of reach, but it RESTS ON a principle astrophotography also
                  obeys: faint signal buried in noise, gathering light over time, how far light travels,
                  what limits resolution, how little light a sensor really receives.
    "none"      — neither is true without inventing a connection.
- "cta_bridge": ONE short spoken sentence (max 16 words) that earns that bridge.

HARD RULES for the bridge, because getting this wrong makes the channel look ridiculous:
- It may ONLY refer to something this script actually talked about. Naming a different object that
  merely happens to be photographable ("...capture Jupiter's moons") after a video about an
  interstellar probe is exactly the failure to avoid.
- For "principle", bridge through the SHARED IDEA, not the object. A probe whose signal arrives buried
  in noise, and which is read by combining several antennas, bridges to combining many photographs —
  because it is the same physics, not because both involve space.
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
_WPS = {"en": 2.35, "it": 2.60}


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
    "You are the editor-in-chief of a top astronomy/space Shorts channel. You receive several candidate "
    "scripts for the SAME video and must pick the SINGLE best one to publish, judging in this order: "
    "(1) HOOK — the first line stops the scroll with a concrete, number-led or counterintuitive fact "
    "(NOT a question, NOT 'imagine'); (2) exactly one DISTINCT verifiable fact per segment, zero "
    "repetition or filler; (3) a curiosity loop opened early and paid off; (4) escalation to a single "
    "peak 'wow'; (5) concrete nouns over adjectives and ZERO cliche; (6) factual plausibility — penalise "
    "invented numbers. Be decisive. Return STRICT JSON only: "
    '{"best": <1-based draft number>, "why": "one short sentence"}.'
)


def _judge_best(client: "OllamaClient", drafts: list[dict], topic: str, language: str) -> dict:
    """Pick the single best of several drafts by the channel rubric. Falls back to the first draft on
    any failure — judging must never lose a usable script."""
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


def generate_script(cfg: LLMConfig, topic: str, seconds: int = 60, language: str = "en",
                    refine_passes: int = 1, best_of: int = 1) -> Script:
    words = _words_for(seconds, language)
    # ~10s of speech per 2-sentence segment, so segment count tracks the target length. A tight upper
    # bound (nseg+1) keeps the model from padding to twice the length.
    # One segment ≈ one shot. At ~10s per segment (the old rule) a 50s video was four shots, and a
    # single image sat on screen for six seconds — the pacing of a documentary, not of a feed where
    # the eye expects a new picture every couple of seconds. ~5s per segment doubles the cut rate
    # without changing the video's length or the word budget: the beats simply get shorter.
    nseg = max(6, round(seconds / 5))
    nseg2 = nseg + 1
    wseg = max(8, round(words / nseg))       # per-segment budget, so the total still lands on target
    user = USER_TMPL.format(topic=topic, seconds=seconds, words=words,
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

    # Length guard: the model tends to write terse, so a script can land well under the target length
    # (a 75-90s target coming in at ~40s). If so, run ONE expand pass that lengthens it with richer
    # narration + more DISTINCT facts. Never regresses (kept only if it parses and is genuinely longer).
    def _nwords(d: dict) -> int:
        return sum(len(str(s.get("narration", "")).split()) for s in _segment_dicts(d))
    # Symmetric guard. The old code only EXPANDED a short draft, so an over-budget draft sailed
    # through and the video ran past the 60s ceiling (a 50s target produced 60.3s of speech). Trim
    # first, then expand — a script is never both.
    if _nwords(data) > words * 1.12:
        cur = _nwords(data)
        try:
            trim_user = (
                f"This script is TOO LONG: ~{cur} spoken words, but the target is ~{words} words across "
                f"{nseg}-{nseg2} segments. Tighten it to the target WITHOUT losing facts: cut filler, "
                f"redundant qualifiers and any restated idea, and shorten sentences. Keep the hook, keep "
                f"every distinct fact, keep the same JSON shape.\n\nCurrent JSON:\n"
                + json.dumps(data, ensure_ascii=False)
            )
            raw = client.chat(system, trim_user, temperature=0.4, num_predict=script_cap)
            cand = _extract_json(raw) if raw else None
            if isinstance(cand, dict) and _segment_dicts(cand) and _nwords(cand) < cur:
                log.info("Length guard: trimmed %d → %d words (target ~%d).",
                         cur, _nwords(cand), words)
                data = cand
        except Exception as e:  # noqa: BLE001 — a failed trim just keeps the longer draft
            log.warning("Length guard (trim) failed (%s) — keeping the draft.", e)
    if _nwords(data) < words * 0.8:
        cur = _nwords(data)
        try:
            expand_user = (
                f"This script is TOO SHORT: ~{cur} spoken words, but the target is ~{words} words across "
                f"{nseg}-{nseg2} segments. Lengthen it to hit the target: extend each narration to two full "
                f"sentences (the fact + its consequence/scale/image) and/or add segments with NEW distinct "
                f"real facts (history, mechanism, scale, discovery, what's next). Keep the same hook, zero "
                f"repetition, zero filler. Return the longer STRICT JSON only.\n\nCurrent draft:\n"
                f"{json.dumps(data, ensure_ascii=False)}")
            expanded = _extract_json(client.chat(system, expand_user, temperature=0.6, num_predict=script_cap))
            if _segment_dicts(expanded) and _nwords(expanded) > cur:
                log.info("Length guard: expanded script %d → %d words (target ~%d).",
                         cur, _nwords(expanded), words)
                data = expanded
        except Exception as e:  # noqa: BLE001 — must never break a usable script
            log.warning("Length expand failed (%s) — keeping current.", e)

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
  "instagram": {{"caption": "one-line hook + 5-8 hashtags"}}
}}"""


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
    return data


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
