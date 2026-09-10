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
import re

import requests

from . import factcheck

log = logging.getLogger(__name__)

LANG_NAME = {"en": "English", "it": "Italian"}
DEFAULT_CPS = 17.0            # characters per second a viewer reads (Netflix adult guideline)
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
- EVERY CARD STANDS ALONE. A viewer reads the cards without the voice: never drop the grammatical
  subject ("Your camera hunts for edges" → "La fotocamera cerca bordi", NOT "Cerca bordi", which
  reads as an order) and never drop a name to save room ("the Milky Way's dust lanes" keeps "della
  Via Lattea"). Cut adjectives and asides instead. A pronoun with nothing to point to is a defect.
- A subtitle never ends on an article or preposition and never splits a name across cards.
- Punctuation marks the PAUSES a reader needs: a comma where one breathes, a full stop before the
  key fact, at most 14 words per sentence, never a semicolon.

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
prossimo: è atterrato, hanno fallito, ha trovato); "errori_di_italiano" = calques or grammar an Italian
reader would notice (e.g. "mai" used as English "ever" in an affirmative sentence — rewrite the idea
in natural Italian, never keep the construction). For EACH item return THREE alternative versions,
all in correct spoken {name}, all faithful to the facts and numbers, of decreasing length: the first
close to max_chars, the second about 85% of it, the third about 70%. Drop the least important clause
rather than squeezing words; never a fragment; never the passato remoto.
Return JSON exactly: {{"items": [{{"id": 1, "options": ["...", "...", "..."]}}, ...]}}.

Subtitles:
{items}"""

PROOF_USER = """Sei un correttore di bozze madrelingua italiano, severo. Questi sottotitoli sono stati scritti da un
modello a partire da un testo inglese: cerca i CALCHI dall'inglese e gli errori che un italiano noterebbe
subito. In particolare: "mai" usato come "ever" in frase affermativa ("ci saluta mai" → "ci mostra un solo
volto"); "attualmente" per "actually"; "eventualmente" per "eventually"; "realizzare" per "realize";
"fare senso"; ordine delle parole inglese; preposizioni sbagliate; accordi di genere e numero; congiuntivi
mancanti; virgola tra soggetto e verbo; anglicismi evitabili. Correggi SOLO cio' che e' sbagliato o
innaturale, senza aggiungere informazioni e senza superare max_chars (accorcia se serve). Per ogni voce
restituisci "ok": true se la frase era gia' corretta, altrimenti "ok": false e il testo corretto.
Restituisci JSON esatto: {{"items": [{{"id": 1, "ok": true, "text": "..."}}, ...]}}.

Sottotitoli:
{items}"""


COMPREHENSION_USER = """Read ONLY these {lang} subtitles, in order, as a viewer who has not heard the voice and
knows nothing about the video. Answer in JSON exactly: {{"subject_en": "the subject of the video in 2-4 ENGLISH
words (e.g. 'black hole radius', 'Jupiter moon Io')", "clear_by_card": <number of the FIRST card after which a
general viewer knows what the video is about, or null if never>, "unclear_cards": [numbers of the cards a viewer
cannot understand EVEN AFTER READING THE CARDS BEFORE IT: no grammatical subject where a statement was meant, a
pronoun with nothing to point to, a thing described but never named, a riddle whose answer never comes], "why": "one sentence"}}.

{items}"""

MAKE_CLEAR_USER = """These Italian subtitles do not work on their own. The subject of the video is "{subject}"; {why}
Rewrite the cards below (max_chars respected) so that each card is understandable by itself — a grammatical
subject in every sentence, no dangling pronoun, the thing named — and by card 2 the subject of the video is
named in plain Italian. Keep the facts and the surprise, correct spoken Italian, passato prossimo.
Return JSON: {{"items": [{{"id": 1, "text": "..."}}, ...]}}.

{items}"""

RESTORE_NAMES_USER = """These Italian subtitles dropped a name that the English narration carries ("restore" lists it per
card). A card must be understandable on its own: rewrite each card so that the name appears in Italian (Milky Way →
Via Lattea, Earth → Terra, Sun → Sole, Moon → Luna, Jupiter → Giove, Saturn → Saturno, Mars → Marte), keeping the
facts, within max_chars — cut adjectives and asides instead. Correct spoken Italian, passato prossimo.
Return JSON: {{"items": [{{"id": 1, "text": "..."}}, ...]}}.

{items}"""

# English proper noun → forms accepted in the Italian card (lower case; matched on the first 5 letters)
_NAME_MAP: dict[str, tuple[str, ...]] = {
    "milky way": ("via lattea",), "earth": ("terra",), "sun": ("sole",), "moon": ("luna",), "mars": ("marte",),
    "jupiter": ("giove",), "saturn": ("saturno",), "venus": ("venere",), "mercury": ("mercurio",),
    "uranus": ("urano",), "neptune": ("nettuno",), "pluto": ("plutone",), "titan": ("titano",),
    "enceladus": ("encelado",), "ceres": ("cerere",), "orion": ("orione",), "pleiades": ("pleiadi",),
    "sirius": ("sirio",), "polaris": ("polare", "polaris"), "andromeda": ("andromeda",),
    "great red spot": ("grande macchia rossa",), "olympus mons": ("olympus mons", "monte olimpo"),
    "world war": ("guerra mondiale",), "russian": ("russ",), "soviet": ("soviet",), "american": ("americ",),
    "european": ("europe",), "italian": ("italian",), "james webb": ("webb",), "north": ("nord",),
    "south": ("sud",), "solar system": ("sistema solare",), "big bang": ("big bang",),
    "titan": ("titano",), "grand finale": ("grand finale", "gran finale"),
}
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)([A-Z])")


def proper_nouns(english: str) -> list[str]:
    """The names an English line carries: runs of capitalised words, minus the word that opens a
    sentence (capitalised for that reason alone: "Squeeze Earth" → Earth, "Karl Schwarzschild" → Schwarzschild)."""
    starts = {m.start(1) for m in _SENTENCE_START.finditer(english or "")}
    out: list[str] = []
    for m in re.finditer(r"(?:[A-Z][\w*'\-]*)(?:\s+[A-Z][\w*'\-]*)*", english or ""):
        words = [w for w in m.group(0).split() if w != "I"]
        words = [re.sub(r"'s$", "", w) for w in words]
        if m.start() in starts:
            words = words[1:]      # capitalised only because it opens the sentence ("Squeeze Earth", "The Milky Way")
        if not words:
            continue
        name = " ".join(words)
        if len(name.replace("*", "")) >= 4:
            out.append(name)
    return out


_ADJECTIVE_ENDINGS = ("ic", "an", "ese", "ish", "ian", "ese")   # Olympic, Jovian, Martian, Chinese: capitalised in English, plain adjectives in Italian


def names_to_keep(english: str) -> list[str]:
    """The names an Italian writer or cutter must keep from an English beat — proper nouns minus the lone
    capitalised adjectives (Olympic, Jovian) that are plain adjectives in Italian. Listing "Olympic" as a name
    made a cutter write "una piscina Olympic" (10/09)."""
    out = []
    for name in proper_nouns(english):
        key = name.lower()
        if " " not in name and key not in _NAME_MAP and key.endswith(_ADJECTIVE_ENDINGS):
            continue
        out.append(name)
    return out


def dropped_names(english: str, italian: str) -> list[str]:
    """Names of the English line that the Italian card lost (accepting the Italian form of the name). A lone
    capitalised adjective (Olympic, Jovian) is not a name unless the map knows it: "an Olympic pool" is "una
    piscina olimpionica" and no name was dropped (measured 10/09, a trial failed on exactly this)."""
    low = (italian or "").lower()
    missing = []
    for name in names_to_keep(english):
        key = name.lower()
        forms = _NAME_MAP.get(key) or tuple(w.lower() for w in name.split() if len(w) >= 4) or (key,)
        if not any((f[:5] if len(f) > 5 else f) in low for f in forms):
            missing.append(name)
    return missing


SAME_SUBJECT_USER = """A cold reader summarised a short science video as: "{subject}". The video's working title is:
"{topic}". Is the reader describing the same video — the same thing being explained, not merely a related
field? Return JSON exactly: {{"same": true or false, "why": "one sentence"}}."""


def same_subject(subject: str, topic: str, backend) -> bool:
    """Semantic fallback for the keyword match: "photographing the Milky Way with a phone" IS the video
    "How to photograph the Milky Way with your phone" even though no keyword lines up."""
    if not subject.strip() or not topic.strip():
        return False
    try:
        data = backend.chat("You compare descriptions. Return STRICT JSON only.",
                            SAME_SUBJECT_USER.format(subject=subject.strip(), topic=topic.strip()), temperature=0.0)
        return bool(isinstance(data, dict) and data.get("same") is True)
    except Exception as e:  # noqa: BLE001
        log.debug("same_subject skipped (%s)", e)
        return False


def comprehension(texts: dict[int, str], topic: str, backend, lang: str = "Italian") -> tuple[bool, str, list[int]]:
    """Can a reader of the subtitles ALONE tell what the video is about? The model names the subject in
    English; it passes if that matches a keyword of the topic or the subject is clear by card 2. It also
    lists the cards that do not work on their own (no subject, dangling pronoun, thing never named)."""
    from .llm import names_subject
    rows = [{"card": i, "text": texts[i]} for i in sorted(texts)]
    data = backend.chat("You are a careful reader. Return STRICT JSON only.",
                        COMPREHENSION_USER.format(lang=lang, items=_items_json(rows)), temperature=0.0)
    subject = str((data or {}).get("subject_en") or "")
    clear = (data or {}).get("clear_by_card")
    try:
        clear = int(clear) if clear is not None else None
    except (TypeError, ValueError):
        clear = None
    unclear: list[int] = []
    for c in (data or {}).get("unclear_cards") or []:
        try:
            unclear.append(int(c))
        except (TypeError, ValueError):
            continue
    ok = names_subject(subject, topic) or (clear is not None and clear <= 2) or same_subject(subject, topic, backend)
    why = f"a reader of the subtitles alone says the video is about {subject!r}, clear by card {clear}"
    if unclear:
        why += f", cards {unclear} do not stand on their own"
    return ok, why, unclear


class SubtitleQualityError(RuntimeError):
    """A subtitle still fails the Italian lint after the proofreader and the fix passes. Raised so the
    build STOPS: the channel's owner asked that a sentence like "Solo un emisfero ci saluta mai" never
    reaches a viewer again — a missing video is recoverable, a published one is not."""


# Deterministic Italian lint — the calques the model itself keeps missing, written as rules it cannot
# talk past. High precision on purpose: a false positive blocks a video.
_NEG_OR_LICIT = re.compile(r"\b(non|nessun\w*|niente|nulla|n[eé]|senza|come|quasi|se|caso|pi[uù] che|meglio che|peggio che)\b", re.I)
_CALQUES = [
    (re.compile(r"\bf(a|anno|are|aceva|acevano|atto)\s+senso\b", re.I), "'fare senso' (make sense → avere senso)"),
    (re.compile(r"\bin ordine (di|a)\b", re.I), "'in ordine di' (in order to → per)"),
    (re.compile(r"\brealizz\w*\s+(che|di)\b", re.I), "'realizzare che' (realize → capire/rendersi conto)"),
    (re.compile(r"\beventualmente\b", re.I), "'eventualmente' (eventually → alla fine)"),
    (re.compile(r"\bprend\w*\s+posto\b", re.I), "'prendere posto' (take place → avvenire)"),
    (re.compile(r"\b(un|una|il|la|lo|i|gli|le)\s+\1\b", re.I), "articolo doppio"),
    (re.compile(r"\b(\w{3,})\s+\1\b", re.I), "parola ripetuta"),
]


def italian_lint(text: str) -> list[str]:
    """Problems an Italian reader would notice at once, or []. Flags 'mai' used as English 'ever' —
    'mai' closing an affirmative clause with no negation, no question, no 'come/quasi/se mai' — plus
    a short list of classic calques. 'quattro mai viste', 'non ... mai', 'hai mai visto?' pass."""
    problems = []
    for clause in re.split(r"[.!?;:]\s*", text or ""):
        clause = clause.strip()
        if not clause:
            continue
        if re.search(r"\bmai\s*,?\s*$", clause, re.I) and not _NEG_OR_LICIT.search(clause) and "?" not in clause:
            problems.append("'mai' affermativo a fine frase (calco di 'ever'): " + clause)
    for pat, why in _CALQUES:
        if pat.search(text or ""):
            problems.append(why)
    # English names of worlds inside an Italian sentence ("l'atmosfera di Saturn", "le lune di Jupiter" — seen
    # 10/09 in a draft the proofreader had to fix twice): Italian has its own names for all of them
    for m in _ENGLISH_WORLDS.finditer(text or ""):
        problems.append(f"nome inglese in una frase italiana: {m.group(0)!r} → {_WORLDS_IT[m.group(0).lower()]}")
    # the passato remoto is literary ("fu individuato", "arrivò"): subtitles speak in the passato prossimo
    rem = remoto_forms(text or "")
    if rem:
        problems.append("passato remoto (" + ", ".join(rem) + "): usa il passato prossimo")
    return problems


_WORLDS_IT = {"saturn": "Saturno", "jupiter": "Giove", "mars": "Marte", "mercury": "Mercurio", "neptune": "Nettuno",
              "uranus": "Urano", "pluto": "Plutone", "earth": "Terra", "moon": "Luna", "sun": "Sole", "venus": "Venere",
              "milky way": "Via Lattea"}
_ENGLISH_WORLDS = re.compile(r"\b(Saturn|Jupiter|Mars|Mercury|Neptune|Uranus|Pluto|Earth|Moon|Sun|Venus|Milky Way)\b")


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
_REMOTO_EXCLUDE = {"però", "ciò", "perciò", "può", "sarò", "farò", "andrò", "avrò", "dirò", "vedrò", "così", "lì", "sì", "dì", "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "potrò", "dovrò", "saprò", "starò", "darò", "verrò", "terrò", "vorrò", "oblò", "metrò"}


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
          cps: float | None = None, topic: str | None = None) -> list[str]:
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
        if italian:        # a native proofreader, cold, temperature 0 — a different task than "translate"
            try:
                proof_rows = [{"id": i, "text": texts[i], "max_chars": limits[i]} for i in ids if i in texts]
                data = backend.chat(system, PROOF_USER.format(items=_items_json(proof_rows)), temperature=0.0)
                fixed = 0
                for it in (data.get("items") or []) if isinstance(data, dict) else []:
                    try:
                        i = int(it.get("id"))
                    except (TypeError, ValueError, AttributeError):
                        continue
                    t = " ".join(str(it.get("text", "")).split())
                    if i in texts and t and t != texts[i] and len(t) <= max(limits[i] * 1.2, len(texts[i])):
                        log.info("Proofreader: seg %d %r → %r", i, texts[i], t)
                        texts[i] = t; fixed += 1
                if fixed:
                    log.info("Proofreader corrected %d subtitle(s).", fixed)
            except Exception as e:  # noqa: BLE001 — the lint below still stands guard
                log.warning("subtitle proofreading skipped (%s)", e)
        for attempt in range(2):      # fix pass: over budget, literary tense or a lint hit → three alternatives, we pick
            bad = {}
            for i in ids:
                if i not in texts:
                    continue
                over = len(texts[i]) > limits[i] * 1.10
                rem = remoto_forms(texts[i]) if italian else []
                lint = list(italian_lint(texts[i])) if italian else []
                longest = max((len(s.split()) for s in re.split(r"(?<=[.!?])\s+", texts[i]) if s.strip()), default=0)
                if longest > 22:            # a breathless sentence is a readability problem, not a lint failure
                    lint.append(f"frase di {longest} parole senza pause: spezzala in due")
                if over or rem or lint:
                    bad[i] = (over, rem, lint)
            if not bad:
                break
            # the model lands ABOVE the number it is given, so it is asked for 92% of the real limit;
            # choose() judges the options against the real one
            rows_fix = [{"id": i, "text": texts[i], "chars": len(texts[i]), "max_chars": int(limits[i] * 0.92),
                         "over": over, "passato_remoto": rem, "errori_di_italiano": lint}
                        for i, (over, rem, lint) in bad.items()]
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
        if italian and texts:   # a name the English carries is not what you cut to fit the budget
            src_by_id = {i: txt for i, txt, _ in segments}
            missing = {i: dropped_names(src_by_id[i], texts[i]) for i in ids if i in texts}
            missing = {i: m for i, m in missing.items() if m}
            if missing:
                log.warning("Subtitles dropped a name: %s — asking for it back", missing)
                try:
                    rows_n = [{"id": i, "english": src_by_id[i], "text": texts[i], "max_chars": limits[i], "restore": m}
                              for i, m in missing.items()]
                    fixed = _parse(backend.chat(system, RESTORE_NAMES_USER.format(items=_items_json(rows_n)),
                                                temperature=0.2), ids)
                    for i, t in fixed.items():
                        if i in missing and len(t) <= limits[i] * 1.2 and not italian_lint(t) \
                                and not dropped_names(src_by_id[i], t):
                            texts[i] = t
                except Exception as e:  # noqa: BLE001
                    log.debug("restore-names pass skipped (%s)", e)
                still = {i: dropped_names(src_by_id[i], texts[i]) for i in missing if i in texts}
                still = {i: m for i, m in still.items() if m}
                if still:
                    log.warning("Subtitles: names still missing after the restore pass: %s", still)
        if italian:        # the hard gate: broken Italian never ships
            broken = {i: italian_lint(texts[i]) for i in ids if i in texts and italian_lint(texts[i])}
            if broken:
                raise SubtitleQualityError("subtitles still fail the Italian lint: " + "; ".join(
                    f"seg {i}: {texts[i]!r} — {', '.join(p)}" for i, p in broken.items()))
        if italian and topic and texts:   # the subject must be understandable from the Italian ALONE
            content_ids = ids[:-1] if len(ids) > 1 else ids      # the CTA bridge is not the subject
            anchor = content_ids[:2]                             # the two cards that give the viewer the context
            ok, why, unclear = comprehension({i: texts[i] for i in content_ids if i in texts}, topic, backend)
            weak = [i for i in unclear if i in content_ids]
            if ok and not weak:
                log.info("Subtitles: %s.", why)                       # on the record: what a cold reader understood
            else:
                log.warning("Subtitles: %s — asking for cards that stand on their own", why)
                try:
                    from .llm import subject_keywords
                    redo = sorted(set(anchor if not ok else []) | set(weak))
                    rows2 = [{"id": i, "text": texts[i], "max_chars": limits[i]} for i in redo if i in texts]
                    fixed = _parse(backend.chat(system, MAKE_CLEAR_USER.format(
                        subject=" ".join(subject_keywords(topic)), why=why, items=_items_json(rows2)), temperature=0.2), ids)
                    for i, t in fixed.items():
                        if i in redo and len(t) <= limits[i] * 1.2 and not italian_lint(t):
                            texts[i] = t
                except Exception as e:  # noqa: BLE001
                    log.debug("make-clear pass skipped (%s)", e)
                ok, why, unclear = comprehension({i: texts[i] for i in content_ids if i in texts}, topic, backend)
                if not ok:
                    raise SubtitleQualityError("i sottotitoli da soli non fanno capire di cosa parla il video — " + why)
                if any(i in unclear for i in anchor):
                    raise SubtitleQualityError("le prime schede non si capiscono da sole — " + why)
                if unclear:
                    log.warning("Subtitles: cards %s still weak on their own (kept: not the anchor)", unclear)
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
    cands = [o for o in options if o and o != current and not (italian and italian_lint(o))]
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
                or (rem(pick) == rem(current) and len(current) > limit * 1.10 and len(pick) <= limit) \
                or (italian and italian_lint(current) and not italian_lint(pick))   # correct beats shorter
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
