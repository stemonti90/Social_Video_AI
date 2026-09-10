"""The ITALIAN script — the second of the two scripts every video now has.

The owner's decision (9/9): two distinct scripts, both written with full care for syntax and grammar.
The English one is the voice; the Italian one IS the subtitles. Until now the Italian cards were a
compression of the English under a reading budget, and compression is where the damage came from:
"Cerca bordi che una stella non ha" (subject dropped, a statement turned into an order), "appariranno
le bande di polvere" (the Milky Way cut to fit), "qualche scatto indietro" (a slider's click rendered
as a photo shot). Grammar checkers approved all three — they were grammatical. They were not Italian
anyone would write.

So the Italian is written as a text of its own by the strong model, from the fact sheet and the
English line, with a subject in every sentence and every card readable alone — and then checked by
tasks that are NOT "judge your own work":

  * back-translation: the Italian is translated back to English literally and compared with the
    original line — a word in the wrong sense, a missing name or number, a statement turned into an
    order shows up as a difference;
  * a strict native proofreader that also sees the English (sense, not just grammar);
  * the fact-check against the sheet, in Italian;
  * the lint (calques), the name check (Via Lattea, Terra…) and the cold reader (what is the video
    about, which cards do not stand alone).

Failing cards are rewritten with the reasons, twice at most; then the script stage FAILS. A video
without a correct Italian script is not made.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import factcheck
from .models import Script, Segment
from .llm import competitor_mentions
from .subtitles import (_backend, _items_json, _parse, comprehension, dropped_names, italian_lint)

log = logging.getLogger(__name__)


class ItalianScriptError(RuntimeError):
    """The Italian script could not be brought to the standard — the video must not be built."""


TERMS = ("obiettivo (lens: mai 'lente'), messa a fuoco (focus), cursore (slider), tacca (a click or step of a "
         "slider: mai 'scatto', mai 'passo'), scatto = una singola foto, esposizione e posa lunga (exposure), sovrapporre o "
         "sommare le foto (stacking), rumore (noise), sensore, treppiede, orizzonte degli eventi, raggio e "
         "diametro (non confonderli), anno luce, chilometri e metri (mai miglia e piedi), bande di polvere (dust "
         "lanes: mai 'corsie'), ammasso, nebulosa, galassia, sonda, lander, rover. Mai nominare altre app o "
         "marchi: il nome dell'app del canale sta solo sulla scheda finale")

WRITER_SYSTEM = ("Sei uno sceneggiatore scientifico madrelingua italiano per un canale di astronomia. Scrivi il "
                 "copione ITALIANO di un video la cui voce narrante è in inglese: il tuo testo diventa i sottotitoli, "
                 "letti da chi non sente l'inglese. Non traduci parola per parola: scrivi in italiano naturale, "
                 "corretto e completo, con gli stessi fatti. Restituisci SOLO JSON.")

WRITER_USER = """Scheda fatti (gli UNICI fatti ammessi):
{facts}

Per ogni segmento hai la frase inglese che la voce dirà. Scrivi la versione italiana: stesso contenuto,
stessi numeri e nomi, stesso ordine, una o due frasi complete.

Regole, tutte obbligatorie:
- Ogni frase ha un soggetto esplicito e un verbo. Mai una frase che sembri un ordine se l'inglese afferma
  ("Your camera hunts for edges" → "La fotocamera cerca i bordi", non "Cerca bordi"). Mai un pronome, un "lo"
  o un "ne" che non rimandi a qualcosa detto nella stessa scheda o in quella prima.
- Ogni scheda si capisce da sola: chi legge solo quella deve sapere di cosa si parla. Entro la scheda 2 il
  soggetto del video è nominato con la parola comune ("buco nero", "Via Lattea", "messa a fuoco").
- Italiano parlato da documentario: parole semplici, passato prossimo (è atterrato, ha calcolato), mai il
  passato remoto, mai calchi dall'inglese ("realizzare" per capire, "attualmente" per in realtà, "mai" per
  ever in frase affermativa, "fare senso").
- Terminologia corretta: {terms}.
- Numeri in cifre, formato italiano (15 secondi; 8,87 millimetri; 26.000 anni; 12 milioni di chilometri;
  1916), unità metriche; in lettere solo i numeri fino a dieci quando non sono misure.
- Italiano da redazione scientifica, non da traduttore: niente parole letterali ("punti singoli" → "punti"),
  niente colloquialismi ("agganciati a un lampione" → "blocca la messa a fuoco su un lampione lontano"),
  niente ridondanze ("provala per trovarla"). Ogni scheda deve sembrare scritta in italiano da chi conosce
  l'argomento.
- Lunghezza: dal 90% al 125% dei caratteri dell'inglese. Se un'immagine inglese non ha un equivalente
  naturale, scrivi il fatto in chiaro.
- Il segmento con role "cta_bridge" è la frase che collega il tema al fotografare il cielo: onesta, in
  italiano naturale, senza nominare l'app (il nome è sulla scheda finale).

Segmenti:
{items}

Restituisci JSON esatto: {{"items": [{{"id": 1, "text": "..."}}, ...]}}, una voce per ogni id."""

REWRITE_USER = """Queste schede italiane hanno problemi (campo "problems") rispetto alla frase inglese e alle regole.
Riscrivi SOLO le schede elencate risolvendo ogni problema, con le stesse regole: soggetto esplicito in ogni
frase, scheda comprensibile da sola, stessi fatti e numeri dell'inglese, terminologia corretta ({terms}),
dal 90% al 125% dei caratteri dell'inglese, passato prossimo, niente calchi.

Schede:
{items}

Restituisci JSON esatto: {{"items": [{{"id": 1, "text": "..."}}, ...]}}."""

BACKTRANSLATE_USER = """Translate each Italian subtitle card below into English, LITERALLY: keep every word in the sense it
has in Italian, every number and every name; do not improve, do not interpret, do not fix.
Return JSON exactly: {{"items": [{{"id": 1, "text": "..."}}, ...]}}.

Cards:
{items}"""

COMPARE_USER = """For each item you get the ORIGINAL English line of a science video and a literal back-translation of
its Italian subtitle. Decide whether the Italian says the same thing: same facts, same numbers and names, the
same sense of every key word, a stated subject where the original states one. A word used in a different
sense (a photo "shot" where a slider's "click" was meant), a missing number or name, an order where the
original makes a statement, or a claim the original does not make = NOT the same. Differences of style,
word order and length are fine.
Return JSON exactly: {{"items": [{{"id": 1, "same": true, "why": "one sentence, in Italian"}}, ...]}}.

Items:
{items}"""

PROOF_SCRIPT_USER = """Sei un correttore di bozze madrelingua italiano, severo, per un canale di astronomia. Queste frasi sono
i sottotitoli di un video e devono essere italiano perfetto: grammatica, sintassi, accordi, punteggiatura,
congiuntivi, preposizioni, ordine delle parole. Cerca i calchi dall'inglese ("mai" come ever in frase
affermativa, "attualmente" per actually, "eventualmente", "realizzare" per capire, "fare senso"), le parole
usate nel senso sbagliato in un contesto fotografico o astronomico ("scatto" per la tacca di un cursore,
"lente" per obiettivo, "miglia"), le frasi senza soggetto che suonano come un ordine, i pronomi che non
rimandano a nulla. Hai anche l'inglese originale per capire il senso: correggi SOLO ciò che è sbagliato o
innaturale, senza aggiungere informazioni. Per ogni voce restituisci "ok": true se la frase era già
perfetta, altrimenti "ok": false e il testo corretto.
Restituisci JSON esatto: {{"items": [{{"id": 1, "ok": true, "text": "..."}}, ...]}}.

Voci:
{items}"""

EDITORIAL_USER = """Sei il caporedattore di una rivista scientifica italiana. Queste sono le schede dei sottotitoli di un video
di astronomia; sono già corrette. Il tuo compito è alzare il livello: segnala ogni scheda che un redattore
riscriverebbe perché letterale ("punti singoli", "corsie di polvere"), colloquiale ("agganciati a un
lampione"), goffa, ridondante ("provala per trovarla"), con ordine delle parole inglese o con un termine che un
appassionato di astrofotografia non userebbe. Hai l'inglese originale per il senso: la versione migliore deve
dire le stesse cose, con gli stessi numeri e nomi, in italiano naturale da redazione, e restare tra il 90% e il
125% dei caratteri dell'inglese. Non toccare le schede già buone. Per ogni voce restituisci "natural": true se la
scheda è già da pubblicare, altrimenti "natural": false, "why" (in italiano, breve) e "better" con la scheda
riscritta. Restituisci JSON esatto: {{"items": [{{"id": 1, "natural": true, "why": "", "better": ""}}, ...]}}.

Voci:
{items}"""

IT_NOTE = ("\n\nNOTE: this script is written in ITALIAN. Judge the facts exactly as for an English script and write "
           "every fix in Italian, in the same register.")

MAX_ROUNDS = 3          # one writing, two rewrites with reasons — then the stage fails

# Words in a sense no Italian photographer uses — the proofreader itself once turned a correct "tacca"
# into "scatto" (a photo), so a correction that introduces one is refused and a card that has one fails.
_BAD_SENSE = (re.compile(r"\bscatt[oi]\s+(indietro|avanti|prima|dopo|più|meno)\b", re.I),
              re.compile(r"\b(la|una|delle?|alla)\s+lent[ei]\b", re.I),
              re.compile(r"\bmigli[ao]\b", re.I),
              re.compile(r"\bcorsi[ae]\s+di\s+polvere\b", re.I))      # dust lanes are "bande", not traffic lanes


def bad_sense(text: str) -> list[str]:
    return [m.group(0) for rx in _BAD_SENSE if (m := rx.search(text or ""))]


def _parse_any(data: dict, ids: list[int]) -> dict[int, str]:
    """Like subtitles._parse, but a rewrite that echoes the input schema ("italian" instead of "text")
    is still read — observed 9/9: two rewrite rounds were silently dropped for that."""
    out: dict[int, str] = {}
    for it in (data.get("items") or []) if isinstance(data, dict) else []:
        try:
            i = int(it.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        text = " ".join(str(it.get("text") or it.get("italian") or it.get("testo") or "").split())
        if i in ids and text:
            out[i] = text
    return out
LEN_MIN, LEN_MAX = 0.6, 1.6   # Italian chars vs English chars: outside this the card lost or padded content


def cards(script: Script) -> dict[int, str]:
    """index → Italian card, for every segment that has one (the CTA's card is its Italian bridge)."""
    return {s.index: s.italian.strip() for s in script.segments if (s.italian or "").strip()}


def complete(script: Script) -> bool:
    """Does every content segment (and the CTA, when its bridge is spoken) carry its Italian card?"""
    have = cards(script)
    for s in script.segments:
        if s.kind == "cta":
            spoken = (script.cta_bridge or "").strip()
            if spoken and spoken in (s.narration or "") and s.index not in have:   # a bridge that is SPOKEN needs its card
                return False
        elif s.narration.strip() and s.index not in have:
            return False
    return bool(have)


def _rows(script: Script) -> list[dict]:
    rows = [{"id": s.index, "english": s.narration.strip()} for s in script.segments
            if s.kind != "cta" and s.narration.strip()]
    cta = next((s for s in script.segments if s.kind == "cta"), None)
    if cta and (script.cta_bridge or "").strip():
        rows.append({"id": cta.index, "english": script.cta_bridge.strip(), "role": "cta_bridge"})
    return rows


def _proof(texts: dict[int, str], en: dict[int, str], backend) -> int:
    """The strict native proofreader, with the English at hand for the sense. Corrections are applied."""
    rows = [{"id": i, "english": en[i], "text": texts[i]} for i in sorted(texts)]
    data = backend.chat("Sei un correttore di bozze madrelingua italiano. Restituisci SOLO JSON.",
                        PROOF_SCRIPT_USER.format(items=_items_json(rows)), temperature=0.0)
    fixed = 0
    for it in (data.get("items") or []) if isinstance(data, dict) else []:
        try:
            i = int(it.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        t = " ".join(str(it.get("text", "")).split())
        if i in texts and it.get("ok") is False and t and t != texts[i] and not italian_lint(t) and not bad_sense(t):
            log.info("Correttore: scheda %d %r → %r", i, texts[i], t)
            texts[i] = t
            fixed += 1
    return fixed


def _editorial(texts: dict[int, str], en: dict[int, str], backend) -> list[str]:
    """The bar above correctness: a magazine editor's pass. A better card replaces the current one when
    it keeps the names, the terms and the length; the other checks then verify the new text."""
    rows = [{"id": i, "english": en[i], "text": texts[i]} for i in sorted(texts)]
    data = backend.chat("Sei il caporedattore di una rivista scientifica italiana. Restituisci SOLO JSON.",
                        EDITORIAL_USER.format(items=_items_json(rows)), temperature=0.0)
    notes: list[str] = []
    for it in (data.get("items") or []) if isinstance(data, dict) else []:
        try:
            i = int(it.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        better = " ".join(str(it.get("better") or "").split())
        if i not in texts or it.get("natural") is not False or not better or better == texts[i]:
            continue
        ratio = len(better) / max(1, len(en[i]))
        if italian_lint(better) or bad_sense(better) or dropped_names(en[i], better) or not (LEN_MIN <= ratio <= LEN_MAX):
            notes.append(f"redazione: proposta per la scheda {i} scartata dalle guardie ({better[:60]!r})")
            continue
        log.info("Redazione: scheda %d %r → %r (%s)", i, texts[i], better, str(it.get("why", ""))[:80])
        notes.append(f"redazione: scheda {i} rialzata — {str(it.get('why', ''))[:80]}")
        texts[i] = better
    return notes


def _factcheck(texts: dict[int, str], script: Script, facts: str | None, cfg) -> list[str]:
    """The sheet is the ground truth for the Italian too. Confident fixes are applied; 'unsure' is noted."""
    content = [s for s in script.segments if s.kind != "cta" and s.index in texts]
    tmp = Script(title=script.title, topic=script.topic,
                 segments=[Segment(index=s.index, narration=texts[s.index], visual=s.visual, keywords=list(s.keywords))
                           for s in content])
    notes: list[str] = []
    try:
        findings = factcheck._judge(tmp, cfg, facts, note=IT_NOTE)
    except Exception as e:  # noqa: BLE001 — the checker is a net; the other checks still run
        log.warning("Fact-check dell'italiano saltato (%s)", e)
        return notes
    if findings:
        applied = factcheck.apply(tmp, findings)
        for s in tmp.segments:
            texts[s.index] = s.narration
        for f in findings:
            notes.append(f"seg {f.segment} [{f.verdict}]: {f.claim[:60]} — {f.why[:100]}")
        log.info("Fact-check dell'italiano: %d rilievi, %d correzioni applicate", len(findings), applied)
    return notes


def _meaning(texts: dict[int, str], en: dict[int, str], backend) -> dict[int, str]:
    """Back-translate literally, then compare with the original English. id → why, for the cards that
    do not say the same thing."""
    rows = [{"id": i, "text": texts[i]} for i in sorted(texts)]
    back = _parse(backend.chat("You are a literal translator. Return STRICT JSON only.",
                               BACKTRANSLATE_USER.format(items=_items_json(rows)), temperature=0.0), list(texts))
    pairs = [{"id": i, "original_english": en[i], "back_translation": back.get(i, "")} for i in sorted(texts) if i in back]
    if not pairs:
        return {}
    data = backend.chat("You compare meanings. Return STRICT JSON only.",
                        COMPARE_USER.format(items=_items_json(pairs)), temperature=0.0)
    out: dict[int, str] = {}
    for it in (data.get("items") or []) if isinstance(data, dict) else []:
        try:
            i = int(it.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        if i in texts and it.get("same") is False:
            out[i] = f"senso diverso dall'inglese: {str(it.get('why', '')).strip()[:160]} (ritraduzione: {back.get(i, '')[:100]!r})"
    return out


def _check(texts: dict[int, str], en: dict[int, str], script: Script, facts: str | None, cfg, backend,
           content_ids: list[int], editorial: bool = True) -> tuple[dict[int, list[str]], list[str]]:
    """Every check on the current Italian. Corrections (proofreader, fact-check) are applied in place;
    what remains wrong comes back as reasons per card, for the rewrite."""
    notes: list[str] = []
    try:
        n = _proof(texts, en, backend)
        if n:
            notes.append(f"correttore: {n} schede corrette")
    except Exception as e:  # noqa: BLE001
        log.warning("Correttore saltato (%s)", e)
    if editorial:
        try:
            notes += _editorial(texts, en, backend)
        except Exception as e:  # noqa: BLE001
            log.warning("Passata di redazione saltata (%s)", e)
    notes += _factcheck(texts, script, facts, cfg)

    problems: dict[int, list[str]] = {}

    def add(i: int, why: str) -> None:
        problems.setdefault(i, []).append(why)

    for i, t in texts.items():
        for p in italian_lint(t):
            add(i, f"italiano scorretto: {p}")
        for w in bad_sense(t):
            add(i, f"termine nel senso sbagliato: {w!r} (una tacca del cursore non è uno scatto; lens è obiettivo; niente miglia)")
        for app in competitor_mentions(t):
            add(i, f"nomina un'altra app ({app}): mai; scrivi 'un'app con messa a fuoco manuale'")
        lost = dropped_names(en[i], t)
        if lost:
            add(i, f"manca il nome che l'inglese ha: {', '.join(lost)}")
        ratio = len(t) / max(1, len(en[i]))
        if ratio > LEN_MAX:
            add(i, f"troppo lunga rispetto al parlato ({ratio:.1f}× i caratteri dell'inglese): il lettore non arriva in fondo")
        elif ratio < LEN_MIN:
            add(i, f"troppo corta ({ratio:.1f}× i caratteri dell'inglese): manca contenuto")
    try:
        for i, why in _meaning(texts, en, backend).items():
            add(i, why)
    except Exception as e:  # noqa: BLE001
        log.warning("Confronto di senso saltato (%s)", e)
    try:
        ok, why, unclear = comprehension({i: texts[i] for i in content_ids if i in texts}, script.topic, backend)
        if not ok:
            for i in content_ids[:2]:
                add(i, "il soggetto del video non è chiaro entro la scheda 2 — " + why)
        anchor = content_ids[:2]                    # the two cards that must give the viewer the context
        for i in unclear:
            if i in anchor:
                add(i, "la scheda non si capisce da sola (soggetto mancante, riferimento vago o cosa mai nominata)")
            elif i in content_ids:
                notes.append(f"lettore a freddo: la scheda {i} non si regge da sola (tollerato: non è l'ancora)")
        notes.append("lettore a freddo: " + why)
    except Exception as e:  # noqa: BLE001
        log.warning("Lettore a freddo saltato (%s)", e)
    return problems, notes


def run(script: Script, facts: str | None, cfg, out_dir: Path | None = None) -> Script:
    """Write the Italian script for `script` (content lines + the CTA bridge), check it, rewrite what
    fails with the reasons, and store it in each segment's `italian`. Raises ItalianScriptError when
    the standard is not reached — no Italian script, no video."""
    if not factcheck._api_key(cfg):
        raise ItalianScriptError("nessuna chiave API per il copione italiano (script.factcheck_key o DEEPSEEK_API_KEY)")
    backend, model = _backend(cfg)
    rows = _rows(script)
    if not rows:
        raise ItalianScriptError("copione vuoto: niente da scrivere in italiano")
    ids = [r["id"] for r in rows]
    en = {r["id"]: r["english"] for r in rows}
    content_ids = [r["id"] for r in rows if r.get("role") != "cta_bridge"]
    texts = _parse(backend.chat(WRITER_SYSTEM, WRITER_USER.format(
        facts=(facts or "(nessuna scheda fatti)").strip(), terms=TERMS, items=_items_json(rows)), temperature=0.3), ids)
    missing = [i for i in ids if i not in texts]
    if missing:
        raise ItalianScriptError(f"il copione italiano manca dei segmenti {missing}")

    report: dict = {"model": model, "rounds": []}
    for round_no in range(1, MAX_ROUNDS + 1):
        problems, notes = _check(texts, en, script, facts, cfg, backend, content_ids, editorial=round_no < MAX_ROUNDS)
        report["rounds"].append({"round": round_no, "notes": notes,
                                 "problems": {str(i): p for i, p in problems.items()},
                                 "texts": {str(i): texts[i] for i in ids}})
        if not problems:
            break
        log.warning("Copione italiano, giro %d: %d schede da riscrivere — %s", round_no, len(problems),
                    "; ".join(f"{i}: {p[0][:70]}" for i, p in sorted(problems.items())))
        if round_no == MAX_ROUNDS:
            _write(report, out_dir)
            raise ItalianScriptError("il copione italiano non raggiunge lo standard dopo %d giri: %s. "
                                     "Il copione inglese è salvato in script.md: scrivi a mano le righe ITALIAN "
                                     "mancanti o sbagliate e lancia `avp build`." % (
                MAX_ROUNDS, "; ".join(f"scheda {i}: {' / '.join(p)}" for i, p in sorted(problems.items()))))
        redo = [{"id": i, "english": en[i], "italian": texts[i], "problems": problems[i]} for i in sorted(problems)]
        fixed = _parse_any(backend.chat(WRITER_SYSTEM, REWRITE_USER.format(terms=TERMS, items=_items_json(redo)),
                                        temperature=0.3), ids)
        if not any(i in fixed for i in problems):
            log.warning("Riscrittura non leggibile (nessuna scheda restituita per %s)", sorted(problems))
        for i, t in fixed.items():
            if i in problems:
                texts[i] = t

    for s in script.segments:
        s.italian = texts.get(s.index, "")
    report["final"] = {str(i): texts[i] for i in ids}
    _write(report, out_dir)
    log.info("Copione italiano pronto: %d schede, %d giri (%s)", len(texts), len(report["rounds"]), model)
    return script


def verify(script: Script, cfg) -> list[str]:
    """Before the cards are cut: the Italian on disk (possibly hand-edited) must still pass the lint,
    keep the English's names, and let a cold reader name the subject. Returns the hard failures."""
    have = cards(script)
    problems: list[str] = []
    en = {s.index: (script.cta_bridge if s.kind == "cta" else s.narration) for s in script.segments}
    for i, t in have.items():
        for p in italian_lint(t):
            problems.append(f"scheda {i}: {p} — {t!r}")
        lost = dropped_names(en.get(i, ""), t)
        if lost:
            log.warning("Scheda %d: manca il nome %s — %r", i, lost, t)
    if script.topic and factcheck._api_key(cfg):
        try:
            backend, _ = _backend(cfg)
            content = {i: t for i, t in have.items() if any(s.index == i and s.kind != "cta" for s in script.segments)}
            ok, why, unclear = comprehension(content, script.topic, backend)
            log.info("Sottotitoli: %s.", why)
            anchor = sorted(content)[:2]
            if not ok:
                problems.append("i sottotitoli da soli non fanno capire di cosa parla il video — " + why)
            elif any(i in unclear for i in anchor):
                problems.append("le prime schede non si capiscono da sole — " + why)
        except Exception as e:  # noqa: BLE001
            log.warning("Lettore a freddo saltato (%s)", e)
    return problems


def _write(report: dict, out_dir: Path | None) -> None:
    if out_dir is None:
        return
    try:
        (Path(out_dir) / "italian_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as e:  # noqa: BLE001
        log.debug("italian_report.json not written (%s)", e)
