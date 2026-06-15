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
from .models import Script, Segment

log = get_logger("avp.llm")

SYSTEM = """You are an elite scriptwriter for a faceless short-form video channel about \
astronomy and space (TikTok / Reels / YouTube Shorts). These craft rules separate gripping from mediocre:
- HOOK: the first 6-8 words must be a concrete, counterintuitive or NUMBER-led statement that stops the scroll. NEVER open with a question, "Imagine", "Have you ever", "Picture this", or "In the vast expanse".
- Exactly ONE new, specific, verifiable fact per content segment (a named object, a number, a comparison, a scale). No filler beats, no empty awe.
- Open a curiosity loop in the first 1-2 segments and PAY IT OFF before the end.
- Escalate to a single peak "wow" moment in the penultimate segment.
- Concrete nouns over adjectives; tight spoken cadence. BANNED words/phrases: mind-blowing, incredible, literally, breathtaking, journey, unlock, delve, "did you know". No markdown, no emojis, no stage directions.
- Every fact must be real and checkable; if unsure, stay qualitative — never invent numbers.
- Vary structure between videos; never sound template-stamped.
- "keywords": 2-4 ENGLISH, archive-catalogable PROPER NOUNS (e.g. "Cassini Saturn", "Carina Nebula JWST", "Curiosity rover Mars") — concrete subjects a NASA/Hubble search will find, matching that segment's visual.
Return STRICT JSON only, no commentary."""

CRITIQUE_SYSTEM = (
    "You are a ruthless short-form video editor. Grade this astronomy/space Shorts script 1-5 on: "
    "hook (stops the scroll in under 2s), fact density (one concrete verifiable fact per segment), "
    "a curiosity loop opened early and paid off, escalation to a single peak, concrete nouns over "
    "adjectives, and ZERO cliche. Then list the 3 weakest lines with an exact rewrite for each. "
    "Be specific and brutal. Plain text, no JSON."
)
CRITIQUE_USER = "Script JSON:\n{script}"
REFINE_SUFFIX = (
    "\n- You are now REVISING an existing draft to satisfy an editor's critique: land the hook in "
    "the first 6-8 words, raise fact density, kill every cliche, keep the curiosity loop paid off. "
    "Keep the SAME JSON shape and a similar segment count."
)
REFINE_USER = ("Current draft JSON:\n{script}\n\nEditor critique to apply:\n{critique}\n\n"
               "Return the improved STRICT JSON only.")

USER_TMPL = """Topic: {topic}
Target: ~{seconds}s of spoken narration (about {words} words total), split into {nseg}-{nseg2} segments.
For each segment provide:
- "narration": one or two spoken sentences,
- "visual": a short cue for the ideal NASA/Hubble footage or image (e.g. "Jupiter's Great Red Spot, close-up"),
- "keywords": 2-4 ENGLISH search keywords for space archives (they are English-indexed).
Also provide a punchy "title".
Return JSON exactly like:
{{"title": "...", "segments": [{{"narration": "...", "visual": "...", "keywords": ["...", "..."]}}]}}"""


def _words_for(seconds: int) -> int:
    return max(20, round(seconds * 2.5))  # ~150 words per minute


class OllamaClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def chat(self, system: str, user: str, fmt: str | None = "json",
             temperature: float | None = None) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # num_ctx caps the KV cache: script/metadata prompts are a few K tokens, so 8K is ample
            # and avoids a multi-GB cache (the model's default 40K context bloated memory → swap).
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_ctx": 8192,
            },
        }
        if fmt:
            payload["format"] = fmt
        try:
            # (connect, read): fail fast (10s) if the daemon is down; bound the read at 5 min so a
            # restarted/stuck Ollama can't silently hang a build for the old 10-minute timeout.
            r = requests.post(f"{self.cfg.host}/api/chat", json=payload, timeout=(10, 300))
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Ollama request failed ({e}). Is the daemon running and '{self.cfg.model}' pulled?"
            ) from e
        return r.json()["message"]["content"]


def unload(cfg: LLMConfig) -> None:
    """Ask Ollama to evict the model from RAM (keep_alive=0). Frees several GB before the
    ffmpeg-heavy assemble — ffmpeg SIGSEGVs under memory pressure. Best-effort and reversible:
    the model reloads automatically on the next request (e.g. the metadata stage)."""
    try:
        requests.post(f"{cfg.host}/api/generate",
                      json={"model": cfg.model, "keep_alive": 0}, timeout=(5, 30))
        log.info("Asked Ollama to unload %s (free RAM for assemble).", cfg.model)
    except requests.RequestException:
        pass


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


def generate_script(cfg: LLMConfig, topic: str, seconds: int = 60, language: str = "en",
                    refine_passes: int = 1) -> Script:
    words = _words_for(seconds)
    nseg = max(4, seconds // 9)
    user = USER_TMPL.format(topic=topic, seconds=seconds, words=words, nseg=nseg, nseg2=nseg + 2)
    name = LANG_NAME.get(language, "English")
    system = SYSTEM + f"\n- Write ALL narration in {name}."
    client = OllamaClient(cfg)
    log.info("Generating %s script for %r with %s ...", name, topic, cfg.model)
    data = _extract_json(client.chat(system, user, temperature=0.85))   # creative first draft
    if not isinstance(data, dict):
        raise RuntimeError("Model returned non-object JSON for the script. Try re-running or a different model.")

    for n in range(max(0, refine_passes)):     # critique → refine; never regress a usable draft
        try:
            draft = json.dumps(data, ensure_ascii=False)
            critique = client.chat(CRITIQUE_SYSTEM, CRITIQUE_USER.format(script=draft),
                                   fmt=None, temperature=0.6)
            refined = _extract_json(client.chat(
                system + REFINE_SUFFIX, REFINE_USER.format(script=draft, critique=critique),
                temperature=0.4))
            if [s for s in _segment_dicts(refined) if str(s.get("narration", "")).strip()]:
                data = refined
                log.info("Script refine pass %d/%d applied.", n + 1, refine_passes)
        except Exception as e:  # noqa: BLE001 — refinement must never break a usable draft
            log.warning("Script refine pass %d failed (%s) — keeping current draft.", n + 1, e)
            break

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
    "Be accurate and non-clickbait. Return STRICT JSON only."
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


def generate_metadata(cfg: LLMConfig, script: Script, funnel: FunnelConfig, language: str = "en") -> dict:
    name = LANG_NAME.get(language, "English")
    user = META_USER.format(title=script.title, narration=script.narration,
                            app=funnel.app_name, tagline=funnel.tagline, url=funnel.url)
    system = META_SYSTEM + f" Write titles, descriptions and captions in {name} (hashtags may stay English)."
    log.info("Generating %s metadata with %s ...", name, cfg.model)
    raw = OllamaClient(cfg).chat(system, user)
    return _extract_json(raw)
