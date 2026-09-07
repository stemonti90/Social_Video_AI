"""Translated subtitles a viewer can actually read: adapted, condensed, revised — not transcribed.

What was wrong (measured on Philae, 6/9): the Italian subtitles were a faithful translation of an
English narration spoken at 2.6 words a second. Cut into 8-word cards they ran at a median of 19.8
characters per second with peaks at 27 — the accepted ceiling for adult readers is 15-17 — and the
local writer's Italian had calques in it ("ganci" for harpoons, "a galla" for floating in space,
"incunedato"). Subtitling is not translation: a subtitler CONDENSES to what the eye can read in the
time the voice gives it, and writes it in the reader's language, not the speaker's.

So the subtitles are now made by the strongest model available (the fact-check model when its key
is present, else the local one), one video at a time with every segment in view for consistent
terminology, under an explicit character budget per segment — seconds × reading speed — and then
REVISED by the same model for spelling, agreement and calques. A segment that still overruns its
budget gets one targeted shortening pass. Everything fails soft to the previous behaviour and, in
the last resort, to the source text: a subtitle problem must never stop a video.
"""
from __future__ import annotations

import json
import logging

import requests

from . import factcheck

log = logging.getLogger(__name__)

LANG_NAME = {"en": "English", "it": "Italian"}
DEFAULT_CPS = 15.0            # characters per second a viewer reads comfortably (Netflix adult max: 17)
MIN_BUDGET = 24               # even a half-second segment gets room for a few words

EDITOR_SYSTEM = (
    "You are a professional subtitle adapter for a science channel. You write on-screen subtitles in "
    "{name} for a video narrated in English. Subtitles are NOT a translation: they are what a viewer "
    "can read in the seconds the voice allows, in correct, natural, spoken {name}. Return STRICT JSON only."
)

EDITOR_USER = """For each segment you get the English narration, the seconds it is on screen and the MAXIMUM number
of characters (spaces included) the {name} subtitle may have. Write the subtitle for each.

Rules:
- Stay within the character limit, but USE the room: aim for 75-100% of max_chars. When a fact fits,
  keep it — condense only what does not fit (adjectives, asides, repetition first). Keep every
  number, name and the surprise. One or two complete sentences, never a fragment.
- Correct, idiomatic {name} — no calques from English, no anglicisms when a {name} word exists,
  correct scientific terminology (a lander's harpoons are 'arpioni', its legs 'zampe'; something
  floating in space 'resta sospeso'; a crevice is a 'fenditura'; a gas giant's storm is a 'tempesta').
- SPOKEN register, as a documentary voice would say it: in Italian the passato prossimo (è atterrato,
  ha rimbalzato), never the literary passato remoto (atterrò, tacque, seppe). Short, plain words.
- Numbers as digits with the {name} thousands separator (16.000 km, 430 km/h, 60 ore); units abbreviated.
- Keep proper nouns. Use the same term for the same thing in every segment.
- A subtitle never ends on an article or preposition and never splits a name across cards.

Segments:
{items}

Return JSON exactly: {{"items": [{{"id": 1, "text": "..."}}, ...]}} with one item per segment id."""

REVISER_USER = """You are a native {name} copy editor. Below are subtitles for a science video, each with its character
limit. Fix ONLY: spelling, grammar and agreement errors, punctuation, wrong scientific terms, calques
from English, literary verb forms (in Italian: passato remoto → passato prossimo) and unnatural
phrasing. Do not add information. Do not exceed a limit; shorten if you
must. Return the same JSON shape: {{"items": [{{"id": 1, "text": "..."}}, ...]}}.

Subtitles:
{items}"""

FIX_USER = """Some of these {name} subtitles need fixing. Per item: "over" = it exceeds max_chars (spaces
included); "passato_remoto" = the literary verb forms it uses (Italian subtitles speak in the passato
prossimo: è atterrato, hanno fallito, ha trovato). For EACH item return THREE alternative versions,
all in correct spoken {name}, all faithful to the facts and numbers, of decreasing length: the first
close to max_chars, the second about 85% of it, the third about 70%. Drop the least important clause
rather than squeezing words; never a fragment; never the passato remoto.
Return JSON exactly: {{"items": [{{"id": 1, "options": ["...", "...", "..."]}}, ...]}}.

Subtitles:
{items}"""

# 3rd-person passato remoto: regular endings plus the irregulars a science script actually meets.
# Excluded look-alikes: però, ciò, perciò, può (and 1st-person futures like sarò, which a narration
# about a spacecraft does not use).
_REMOTO_IRREGULAR = {"fu", "furono", "ebbe", "ebbero", "fece", "fecero", "disse", "dissero", "seppe",
                     "seppero", "tacque", "tacquero", "divenne", "divennero", "venne", "vennero", "vide",
                     "videro", "prese", "presero", "rimase", "rimasero", "nacque", "nacquero", "morì",
                     "morirono", "scese", "scesero", "cadde", "caddero", "rispose", "risposero", "scoprì",
                     "scoprirono", "raggiunse", "raggiunsero", "perse", "persero", "volle", "vollero",
                     "giunse", "giunsero", "rese", "resero", "scrisse", "scrissero", "mise", "misero",
                     "crebbe", "crebbero", "apparve", "apparvero", "scomparve", "scomparvero"}
_REMOTO_EXCLUDE = {"però", "ciò", "perciò", "può", "sarò", "farò", "andrò", "avrò", "dirò", "vedrò"}


def remoto_forms(text: str) -> list[str]:
    """The passato remoto verbs in an Italian sentence, or [] — the literary tense a subtitle should not use."""
    import re
    out = []
    for w in re.findall(r"[A-Za-zÀ-ÿ']+", text or ""):
        lw = w.lower().strip("'")
        if lw in _REMOTO_EXCLUDE:
            continue
        if lw in _REMOTO_IRREGULAR or re.search(r"(?<![aeiou])(ò|ì|arono|irono|erono|ettero)$", lw) \
                or re.search(r"(arono|irono|erono|ettero)$", lw):
            out.append(w)
    return out


def budget(seconds: float, cps: float = DEFAULT_CPS) -> int:
    """How many characters a subtitle may have for `seconds` of screen time at `cps`."""
    return max(MIN_BUDGET, int(round(max(0.0, seconds) * cps)))


def _items_json(rows: list[dict]) -> str:
    return json.dumps(rows, ensure_ascii=False, indent=1)


def _parse(data: dict, ids: list[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for it in (data.get("items") or []) if isinstance(data, dict) else []:
        try:
            i = int(it.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        text = " ".join(str(it.get("text", "")).split())
        if i in ids and text:
            out[i] = text
    return out


class _Remote:
    """The fact-check model as a chat endpoint (same key, same URL)."""

    def __init__(self, key: str, model: str):
        self.key, self.model = key, model

    def chat(self, system: str, user: str, temperature: float = 0.2) -> dict:
        r = requests.post(
            factcheck.DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            json={"model": self.model,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                  "temperature": temperature,
                  "response_format": {"type": "json_object"}},
            timeout=getattr(factcheck, "TIMEOUT", (30, 180)),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {(r.text or '')[:200]}")
        return factcheck._extract_json(r.json()["choices"][0]["message"]["content"])


class _Local:
    """The local writer, for when there is no remote key. Same prompts; weaker Italian."""

    def __init__(self, llm_cfg):
        from .llm import OllamaClient, _extract_json
        self.client, self._extract = OllamaClient(llm_cfg), _extract_json

    def chat(self, system: str, user: str, temperature: float = 0.3) -> dict:
        return self._extract(self.client.chat(system, user, temperature=temperature))


def _backend(cfg):
    mode = str(getattr(cfg.script, "subtitle_editor", "auto") or "auto").lower()
    key = factcheck._api_key(cfg) if mode in ("auto", "remote") else ""
    if key:
        model = (str(getattr(cfg.script, "brief_model", "") or "").strip()
                 or str(getattr(cfg.script, "factcheck_model", "deepseek-chat") or "deepseek-chat"))
        return _Remote(key, model), model
    if mode == "remote":
        log.warning("subtitle_editor=remote but no API key — using the local model.")
    return _Local(cfg.llm), str(getattr(cfg.llm, "model", "local"))


def adapt(segments: list[tuple[int, str, float]], target_lang: str, cfg,
          cps: float | None = None) -> list[str]:
    """(index, english_text, seconds) per segment → the {target_lang} subtitle text per segment,
    same order. Adapted under budget, revised, shortened once if still over; falls back per segment
    to the source text."""
    if not segments:
        return []
    name = LANG_NAME.get(target_lang, target_lang)
    cps = float(cps or getattr(getattr(cfg, "captions", None), "reading_cps", 0) or DEFAULT_CPS)
    ids = [i for i, _, _ in segments]
    limits = {i: budget(sec, cps) for i, _, sec in segments}
    rows = [{"id": i, "english": txt, "seconds": round(sec, 1), "max_chars": limits[i]}
            for i, txt, sec in segments]
    backend, model = _backend(cfg)
    system = EDITOR_SYSTEM.format(name=name)
    try:
        texts = _parse(backend.chat(system, EDITOR_USER.format(name=name, items=_items_json(rows))), ids)
    except Exception as e:  # noqa: BLE001 — fall back to the source text below
        log.warning("Subtitle adaptation failed (%s) — using the source text.", e)
        texts = {}
    if texts:
        try:      # revision: same model reads its own work cold for spelling, agreement, calques
            rev_rows = [{"id": i, "text": texts[i], "max_chars": limits[i]} for i in ids if i in texts]
            revised = _parse(backend.chat(system, REVISER_USER.format(name=name, items=_items_json(rev_rows))), ids)
            for i, t in revised.items():
                if len(t) <= max(limits[i], len(texts.get(i, ""))):   # a revision must not make it longer
                    texts[i] = t
        except Exception as e:  # noqa: BLE001
            log.debug("subtitle revision skipped (%s)", e)
        italian = name == "Italian"
        for attempt in range(2):      # fix pass: over budget and/or literary tense → three alternatives, we pick
            bad = {}
            for i in ids:
                if i not in texts:
                    continue
                over = len(texts[i]) > limits[i] * 1.10
                rem = remoto_forms(texts[i]) if italian else []
                if over or rem:
                    bad[i] = (over, rem)
            if not bad:
                break
            # the model lands ABOVE the number it is given, so it is asked for 92% of the real limit;
            # choose() judges the options against the real one
            rows_fix = [{"id": i, "text": texts[i], "chars": len(texts[i]), "max_chars": int(limits[i] * 0.92),
                         "over": over, "passato_remoto": rem} for i, (over, rem) in bad.items()]
            try:
                data = backend.chat(system, FIX_USER.format(name=name, items=_items_json(rows_fix)),
                                    temperature=0.4 + 0.3 * attempt)
            except Exception as e:  # noqa: BLE001
                log.debug("subtitle fix pass skipped (%s)", e)
                break
            for i, options in _parse_options(data, ids).items():
                pick = choose(texts[i], options, limits[i], italian)
                if pick:
                    texts[i] = pick
        still = [i for i in ids if i in texts and len(texts[i]) > limits[i] * 1.10]
        if still:
            log.warning("Subtitles over budget after shortening: segments %s (limit %s cps).", still, cps)
    out = [texts.get(i) or txt for i, txt, _ in segments]
    made = sum(1 for i in ids if i in texts)
    log.info("Subtitles: %d/%d segments adapted to %s by %s (budget %.0f cps).", made, len(ids), name, model, cps)
    return out


def _parse_options(data: dict, ids: list[int]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for it in (data.get("items") or []) if isinstance(data, dict) else []:
        try:
            i = int(it.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        opts = it.get("options") if isinstance(it.get("options"), list) else [it.get("text")]
        clean = [" ".join(str(o).split()) for o in opts if str(o or "").strip()]
        if i in ids and clean:
            out[i] = clean
    return out


def choose(current: str, options: list[str], limit: int, italian: bool) -> str | None:
    """The best alternative for a subtitle, or None to keep `current`.

    Preference: no literary tense AND within the limit → the LONGEST such (use the time there is);
    else no literary tense → the shortest; else within the limit → the longest; else the shortest
    option, and only if it is shorter than what we have. An option must be an improvement — fewer
    passato remoto forms or fewer characters — or it is not taken."""
    def rem(t): return len(remoto_forms(t)) if italian else 0
    cands = [o for o in options if o and o != current]
    if not cands:
        return None
    tiers = [
        sorted([o for o in cands if rem(o) == 0 and len(o) <= limit], key=len, reverse=True),
        sorted([o for o in cands if rem(o) == 0], key=len),
        sorted([o for o in cands if len(o) <= limit], key=len, reverse=True),
        sorted(cands, key=len),
    ]
    for tier in tiers:
        if tier:
            pick = tier[0]
            better = rem(pick) < rem(current) or len(pick) < len(current) \
                or (rem(pick) == rem(current) and len(current) > limit * 1.10 and len(pick) <= limit)
            return pick if better else None
    return None


def stale(existing: list[dict] | None, segments: list[tuple[int, str, float]]) -> bool:
    """True when the saved subtitles were made from different source text OR for a different
    duration (the budget is seconds × cps: a re-voiced line that got shorter needs a shorter
    subtitle), or when the record lacks either."""
    if not existing:
        return True
    by_index = {int(d.get("index", -1)): d for d in existing if isinstance(d, dict)}
    for i, txt, sec in segments:
        d = by_index.get(i)
        if not d or "source" not in d or " ".join(str(d["source"]).split()) != " ".join(txt.split()):
            return True
        try:
            was = float(d.get("seconds"))
        except (TypeError, ValueError):
            return True
        if sec and abs(was - sec) > 0.10 * max(sec, 0.1):
            return True
    return False


def reading_speed(cards: list[tuple[str, float, float]]) -> list[float]:
    """Characters per second of each (text, start, end) card."""
    return [len(t) / max(0.05, e - s) for t, s, e in cards]
