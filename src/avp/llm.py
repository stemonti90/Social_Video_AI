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
- LENGTH: every content segment is TWO full spoken sentences (~18-24 spoken words). HIT the requested total word count and segment count — coming in SHORT or running LONG are both failures. Do not pad with filler or repetition, and do not exceed the target; reach the exact length with real, distinct facts richly told.
- EXPLAIN, don't list: if the topic is a specific mission/probe/object/person/event, build a clear through-line (what it is → what it did → why it matters), not disconnected trivia.
- Every segment must make a DISTINCT point. NEVER repeat, restate or paraphrase an earlier line to reach the segment count — if you genuinely run out of distinct facts, broaden the angle (history, mechanism, scale, discovery, what's next) rather than repeating.
- Open a curiosity loop in the first 1-2 segments and PAY IT OFF before the end.
- Escalate to a single peak "wow" moment in the penultimate segment.
- Concrete nouns over adjectives; a confident, flowing spoken cadence (not terse, not padded). BANNED words/phrases: mind-blowing, incredible, literally, breathtaking, journey, unlock, delve, "did you know". No markdown, no emojis, no stage directions.
- Every fact must be real and checkable; if unsure, stay qualitative — never invent numbers.
- Vary structure between videos; never sound template-stamped.
- "keywords": 2-4 ENGLISH, archive-catalogable PROPER NOUNS (e.g. "Cassini Saturn", "Carina Nebula JWST", "Curiosity rover Mars") — concrete subjects a NASA/Hubble search will find, matching that segment's visual.
Return STRICT JSON only, no commentary."""

CRITIQUE_SYSTEM = (
    "You are a ruthless short-form video editor. Grade this astronomy/space Shorts script 1-5 on: "
    "hook (stops the scroll in under 2s), fact density (one concrete verifiable fact per segment), "
    "a curiosity loop opened early and paid off, escalation to a single peak, concrete nouns over "
    "adjectives, ZERO cliche, and NO repeated or near-duplicate segments. Then list the 3 weakest "
    "lines with an exact rewrite for each, and explicitly flag any segment that repeats another. "
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
Target: ~{seconds}s of spoken narration — about {words} words TOTAL (stay within ±10%; do NOT run long), in EXACTLY {nseg}-{nseg2} segments. Going over the length is as wrong as coming in short.
If the topic is a specific MISSION, PROBE, OBJECT, PERSON, or EVENT, EXPLAIN it with a clear through-line — what it is, what it did / what happened, and why it matters — not a list of disconnected trivia.
For each segment provide:
- "narration": TWO full spoken sentences (~18-24 words) — the point, then its consequence/scale/image,
- "visual": a short cue for the ideal NASA/Hubble footage or image (e.g. "Jupiter's Great Red Spot, close-up"),
- "keywords": 2-4 ENGLISH search keywords for space archives (they are English-indexed).
Also provide a punchy "title".
Return JSON exactly like:
{{"title": "...", "segments": [{{"narration": "...", "visual": "...", "keywords": ["...", "..."]}}]}}"""


def _words_for(seconds: int) -> int:
    return max(20, round(seconds * 2.5))  # ~150 words per minute


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
    words = _words_for(seconds)
    # ~10s of speech per 2-sentence segment, so segment count tracks the target length. A tight upper
    # bound (nseg+1) keeps the model from padding to twice the length.
    nseg = max(4, round(seconds / 10))
    nseg2 = nseg + 1
    user = USER_TMPL.format(topic=topic, seconds=seconds, words=words, nseg=nseg, nseg2=nseg2)
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
    return Script(title=str(title or topic).strip(), segments=segments,
                  target_seconds=seconds, topic=topic)


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
