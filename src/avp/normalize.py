"""Speech normalization: turn the *written* narration (displayText) into a *spoken* form
(speechText) so the TTS never reads digits letter-by-letter.

The script keeps numbers as digits for the human-edited / on-screen text (more readable: "85%",
"1665", "24 km"). Before synthesis we rewrite them into words in the target language:
    "85%"      -> "ottantacinque per cento"
    "1665"     -> "milleseicentosessantacinque"
    "24 km"    -> "ventiquattro chilometri"
    "3,5%"     -> "tre virgola cinque per cento"
    "€50"      -> "cinquanta euro"
This module is pure + dependency-free (hand-rolled IT/EN cardinals, no new deps) and is the single
source of speechText. Both TTS and the caption/karaoke timing run on its output so the highlighted
word always matches the heard word. Numbers in metadata/JSON/filenames are left untouched.
"""
from __future__ import annotations

import re

# ── Italian cardinals ──────────────────────────────────────────────────────────────────────────
_IT_U = ["", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove",
         "dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici",
         "diciassette", "diciotto", "diciannove"]
_IT_T = ["", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta"]


def _it_two(n: int) -> str:                 # 0..99
    if n < 20:
        return _IT_U[n]
    t, u = divmod(n, 10)
    base = _IT_T[t]
    if u == 0:
        return base
    if u in (1, 8):                         # elision: venti+uno→ventuno, venti+otto→ventotto
        base = base[:-1]
    return base + ("tré" if u == 3 else _IT_U[u])   # ventitré (accented)


def _it_three(n: int) -> str:               # 0..999
    h, rest = divmod(n, 100)
    out = "" if h == 0 else ("cento" if h == 1 else _IT_U[h] + "cento")
    if rest:
        r = _it_two(rest)
        if out and r[0] in "ou":            # cento+otto→centotto, cento+uno→centuno
            out = out[:-1]
        out += r
    return out


def _it_below_million(n: int) -> str:       # 0..999999, written as ONE concatenated word
    th, rest = divmod(n, 1000)
    out = ""
    if th:
        out = "mille" if th == 1 else _it_three(th) + "mila"
    if rest:
        out += _it_three(rest)
    return out or "zero"


def int_to_words_it(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "meno " + int_to_words_it(-n)
    parts: list[str] = []
    miliardi, n = divmod(n, 1_000_000_000)
    milioni, n = divmod(n, 1_000_000)
    if miliardi:
        parts.append("un miliardo" if miliardi == 1 else int_to_words_it(miliardi) + " miliardi")
    if milioni:
        parts.append("un milione" if milioni == 1 else _it_below_million(milioni) + " milioni")
    if n or not parts:
        parts.append(_it_below_million(n))
    return " ".join(p for p in parts if p)


# ── English cardinals (channel can run language=en) ──────────────────────────────────────────────
_EN_U = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_EN_T = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _en_three(n: int) -> str:               # 0..999
    h, rest = divmod(n, 100)
    out = [] if h == 0 else [_EN_U[h], "hundred"]
    if rest:
        if h:
            out.append("and") if False else None      # keep US style "one hundred five"
        if rest < 20:
            out.append(_EN_U[rest])
        else:
            t, u = divmod(rest, 10)
            out.append(_EN_T[t] + ("-" + _EN_U[u] if u else ""))
    return " ".join(out)


def int_to_words_en(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + int_to_words_en(-n)
    groups = [("billion", 1_000_000_000), ("million", 1_000_000), ("thousand", 1000)]
    parts: list[str] = []
    for name, val in groups:
        q, n = divmod(n, val)
        if q:
            parts.append(_en_three(q) + " " + name)
    if n or not parts:
        parts.append(_en_three(n))
    return " ".join(parts)


def int_to_words(n: int, lang: str = "it") -> str:
    return int_to_words_it(n) if lang.startswith("it") else int_to_words_en(n)


# ── unit / currency lexicons (symbol → spoken word) ──────────────────────────────────────────────
_UNITS = {
    "it": {"km": "chilometri", "m": "metri", "cm": "centimetri", "mm": "millimetri",
           "kg": "chilogrammi", "g": "grammi", "km/h": "chilometri orari", "km/s": "chilometri al secondo",
           "°c": "gradi", "°": "gradi", "°k": "kelvin", "k": "kelvin", "ly": "anni luce", "au": "unità astronomiche"},
    "en": {"km": "kilometers", "m": "meters", "cm": "centimeters", "mm": "millimeters",
           "kg": "kilograms", "g": "grams", "km/h": "kilometers per hour", "km/s": "kilometers per second",
           "°c": "degrees", "°": "degrees", "°k": "kelvin", "k": "kelvin", "ly": "light years", "au": "astronomical units"},
}
_CURRENCY = {
    "it": {"€": "euro", "$": "dollari", "£": "sterline", "¥": "yen"},
    "en": {"€": "euros", "$": "dollars", "£": "pounds", "¥": "yen"},
}
_PERCENT = {"it": "per cento", "en": "percent"}
_POINT = {"it": "virgola", "en": "point"}

# A numeric expression: optional currency, digits with . thousands-sep and , decimal, optional %/unit.
_NUM_RE = re.compile(
    r"(?:(?P<cur>[€$£¥])\s?)?"           # currency + its own optional space (don't eat a leading word's space)
    r"(?P<int>\d{1,3}(?:\.\d{3})+|\d+)"
    r"(?:,(?P<dec>\d+))?"
    r"(?P<suf>\s?%|\s?°[CKck]?|\s?(?:km/h|km/s|km|cm|mm|kg|ly|au|m|g|k)\b)?",
    re.IGNORECASE,
)


def _digits_to_words(s: str, lang: str) -> str:
    return " ".join(int_to_words(int(d), lang) for d in s)


def _spoken_number(m: re.Match, lang: str) -> str:
    lang_k = "it" if lang.startswith("it") else "en"
    int_part = int(m.group("int").replace(".", ""))
    dec = m.group("dec")
    words = int_to_words(int_part, lang)
    if dec is not None:
        words += f" {_POINT[lang_k]} " + _digits_to_words(dec, lang)
    cur = m.group("cur")
    suf = (m.group("suf") or "").strip().lower()
    tail = ""
    if suf == "%":
        tail = " " + _PERCENT[lang_k]
    elif suf:
        tail = " " + _UNITS[lang_k].get(suf, suf)
    if cur:
        tail = " " + _CURRENCY[lang_k].get(cur, "") + tail
    return (words + tail).strip()


def to_speech(text: str, lang: str = "it") -> str:
    """Rewrite digit-numbers in `text` into spoken words for `lang` (it|en). Idempotent on
    already-spoken text; leaves word-suffixes like 'milioni'/'mila' and tech tickers alone."""
    if not text:
        return text
    return _NUM_RE.sub(lambda m: _spoken_number(m, lang), text)


def segment_speech(narration: str, lang: str = "it", normalize: bool = True) -> str:
    """speechText for one segment's narration (displayText). The single entry point used by both
    the TTS and the caption/karaoke timing so the heard word == the highlighted word."""
    txt = (narration or "").strip()
    return to_speech(txt, lang) if normalize else txt
