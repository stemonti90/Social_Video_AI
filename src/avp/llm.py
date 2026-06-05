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

SYSTEM = """You are a scriptwriter for a faceless short-form video channel \
about astronomy and space. Audience: curious viewers on TikTok / Reels / YouTube Shorts.
Style: vivid, accurate, awe-driven; a strong hook in the first sentence; conversational.
Hard rules:
- Every fact must be real and checkable. If unsure, stay qualitative — never invent numbers.
- No clickbait falsehoods, no filler.
- "narration" is text to be SPOKEN aloud: no stage directions, no emojis, no markdown.
- Vary structure between videos; never sound template-stamped.
Return STRICT JSON only, no commentary."""

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

    def chat(self, system: str, user: str, fmt: str | None = "json") -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
        }
        if fmt:
            payload["format"] = fmt
        try:
            r = requests.post(f"{self.cfg.host}/api/chat", json=payload, timeout=600)
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Ollama request failed ({e}). Is the daemon running and '{self.cfg.model}' pulled?"
            ) from e
        return r.json()["message"]["content"]


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


LANG_NAME = {"en": "English", "it": "Italian"}


def generate_script(cfg: LLMConfig, topic: str, seconds: int = 60, language: str = "en") -> Script:
    words = _words_for(seconds)
    nseg = max(4, seconds // 9)
    user = USER_TMPL.format(topic=topic, seconds=seconds, words=words, nseg=nseg, nseg2=nseg + 2)
    name = LANG_NAME.get(language, "English")
    system = SYSTEM + f"\n- Write ALL narration in {name}."
    log.info("Generating %s script for %r with %s ...", name, topic, cfg.model)
    raw = OllamaClient(cfg).chat(system, user)
    data = _extract_json(raw)

    segments = [
        Segment(
            index=i + 1,
            narration=str(s.get("narration", "")).strip(),
            visual=str(s.get("visual", "")).strip(),
            keywords=[str(k).strip() for k in (s.get("keywords") or []) if str(k).strip()],
        )
        for i, s in enumerate(data.get("segments", []))
        if str(s.get("narration", "")).strip()
    ]
    if not segments:
        raise RuntimeError("Model returned no usable segments. Try re-running or a different model.")
    return Script(title=str(data.get("title", topic)).strip(), segments=segments,
                  target_seconds=seconds, topic=topic)


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
