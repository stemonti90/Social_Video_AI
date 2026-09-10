"""The editorial machine (avp/editorial_engine.py): a story is chosen and developed before anyone writes.
A fake API answers every role by recognising its prompt; the mechanical nets and the pipeline glue are real."""
import inspect
import re
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from avp import editorial_engine as E
from avp.config import Config
from avp.models import Script, Segment

TOPIC = "Saturn's rings are disappearing"
EN = ["Saturn's rings look permanent, but they are falling into the planet right now.",
      "Cassini measured the fall: about a thousand kilograms of ring ice rain onto Saturn every second.",
      "At that rate the rings vanish in roughly 100 million years, a blink for the planet.",
      "The ice is pulled by Saturn's magnetic field, which drags charged grains down its lines.",
      "The rings may also be young: perhaps 100 million years old, born after the dinosaurs.",
      "We are watching Saturn in the brief age when it has rings, and that age is ending."]
IT = ["Gli anelli di Saturno sembrano eterni, ma stanno cadendo sul pianeta mentre li guardiamo.",
      "Cassini ha misurato la caduta: circa mille chilogrammi di ghiaccio degli anelli piovono su Saturno ogni secondo.",
      "A questo ritmo gli anelli scompaiono in circa 100 milioni di anni, un istante nella vita del pianeta.",
      "Il ghiaccio è trascinato dal campo magnetico di Saturno, che porta i grani carichi lungo le sue linee.",
      "Gli anelli potrebbero anche essere giovani: forse solo 100 milioni di anni, nati dopo i dinosauri.",
      "Stiamo guardando Saturno nella breve età in cui ha gli anelli, e quell'età sta finendo."]
BRIDGE_EN = "Saturn's rings are visible in a small telescope tonight, for now."
BRIDGE_IT = "Gli anelli di Saturno si vedono stasera con un piccolo telescopio, per ora."


def _cfg(tmp):
    cfg = Config()
    cfg.script.factcheck_key = "k"
    cfg.script.target_seconds = 48
    cfg.script.language = "en"
    cfg.script.subtitle_language = "it"
    cfg.script.engine = "editorial"
    cfg.paths.projects_dir = tmp
    return cfg


def _project(tmp):
    root = Path(tmp) / "saturn"
    root.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(root=root, script_json=root / "script.json", script_md=root / "script.md",
                           manifest=SimpleNamespace(data={}, mark=mock.Mock()))


class FakeAPI:
    """Answers every role of the engine from the prompt it recognises; remembers what it was asked."""

    def __init__(self, *, first_review=None, review2=None, reject_first_story=False, en_lines=None):
        self.prompts: list[tuple[str, str]] = []
        self.first_review = first_review or {"idea": "strong", "specificity": "strong", "density": "strong",
                                             "originality": "solid", "narration": "strong", "language": "strong", "ai_smell": "none"}
        self.review2 = review2
        self.reject_first_story = reject_first_story
        self.reviews_seen = 0
        self.director_calls = 0
        self.en_lines = en_lines or EN

    def __call__(self, url, headers=None, json=None, timeout=None):
        import json as _json
        system, user = json["messages"][0]["content"], json["messages"][1]["content"]
        self.prompts.append((system, user))
        payload = self.route(system, user)

        class R:
            status_code = 200
            text = ""
            def json(self): return {"choices": [{"message": {"content": _json.dumps(payload, ensure_ascii=False)}}]}
        return R()

    def route(self, system, user):
        if "Generate 10 genuinely different editorial angles" in user:
            self.director_calls += 1
            return {"angles": [
                {"id": 1, "angle": "Saturn has rings", "central_question": "why", "value_sources": [], "strongest_fact": "rings",
                 "visual_idea": "rings", "risk": "obvious", "killed": True, "kill_reason": "the fact everyone cites first"},
                {"id": 2, "angle": "The rings are raining onto Saturn and will be gone in 100 million years",
                 "central_question": "How long do the rings have?", "value_sources": ["reversal", "scale"],
                 "strongest_fact": "ring rain ~1000 kg/s", "visual_idea": "ice falling along field lines", "risk": "rate uncertain", "killed": False, "kill_reason": None},
                {"id": 3, "angle": "The rings are young", "central_question": "How old are they?", "value_sources": ["reversal"],
                 "strongest_fact": "age 100 Myr", "visual_idea": "clean ice", "risk": "debated", "killed": False, "kill_reason": None}],
                "ranking": [2, 3], "comparison": "2 has the number and the reversal."}
        if "Choose the strongest angle" in user:
            return {"winner": 2, "why_this_story": "Verifiable rate, reversal of permanence, tangible timescale.",
                    "rejected": [{"id": 3, "reason": "debated age"}]}
        if "CHOSEN ANGLE:" in user:
            return {"editorial_angle": "The rings are disappearing", "central_question": "How long do the rings have?",
                    "central_tension": "the icon of permanence is temporary", "audience_takeaway": "rings are an age, not a feature",
                    "key_facts": ["ring rain ~1000 kg/s", "100 Myr"], "optional_facts": ["young rings"], "excluded_facts": ["Titan"],
                    "misconception": "rings are eternal", "quantitative_comparison": "1000 kg/s", "opening_type": "counterintuitive fact",
                    "closing_type": "change of view", "what_not_to_do": ["no 'majestic'"]}
        if "narrative designer" in system:
            beats = [{"beat": i + 1, "fact": f"fact {i + 1}", "visual": f"visual {i + 1}", "role": "build"} for i in range(6)]
            return {"arcs": [{"id": k, "thesis": f"arc {k}", "beats": beats, "why": "w"} for k in (1, 2, 3)]}
        if "senior narrative editor" in system:
            return {"winner": 3, "why": "arc 3 escalates."}
        if "compress one spoken sentence" in system:
            self.tightened = getattr(self, "tightened", 0) + 1
            import re as _re
            cap = int(_re.search(r"AT MOST (\d+) words", user).group(1))
            text = user.split("SEGMENT:", 1)[1].split("Return {", 1)[0].strip()
            return {"narration": " ".join(text.split()[:cap]).rstrip(",;") + ".", "words": cap}
        if "Ripari una sola battuta" in system:
            self.repaired = getattr(self, "repaired", 0) + 1
            import re as _re
            n = int(_re.search(r"la numero (\d+)", user).group(1))
            return {"narration": IT[n - 1]}
        if "limite rigido di caratteri" in system:
            import re as _re
            cap = int(_re.search(r"AL MASSIMO (\d+) caratteri", user).group(1))
            text = user.split("BATTUTA:", 1)[1].split("Restituisci {", 1)[0].strip()
            cut = text[:cap].rsplit(" ", 1)[0].rstrip(",;:") + "."
            return {"narration": cut, "chars": len(cut)}
        if "cutting a spoken script to length" in system:
            self.tightened = getattr(self, "tightened", 0) + 1
            import json as _json
            body = _json.loads(user.split("SCRIPT:", 1)[1].rsplit("Return the same JSON shape", 1)[0].strip())
            for seg in body["segments"]:
                seg["narration"] = " ".join(seg["narration"].split()[:14]).rstrip(",;") + "."
            return body
        if user.startswith("LANGUAGE: English"):
            lines = self.en_lines
            if "You are REVISING" in user:
                lines = [l.replace("look permanent", "look eternal") + " and this revising clause runs long on purpose" for l in lines]
            return {"title": "The Rings Are Falling", "segments": [{"narration": l, "visual": f"visual {i}", "keywords": ["Saturn"]} for i, l in enumerate(lines, 1)],
                    "bridge_kind": "shoot", "cta_bridge": BRIDGE_EN}
        if user.startswith("LANGUAGE: Italian"):
            return {"title": "Gli anelli stanno cadendo", "segments": [{"narration": l, "visual": f"visual {i}", "keywords": ["Saturno"]} for i, l in enumerate(IT, 1)],
                    "bridge_kind": "shoot", "cta_bridge": BRIDGE_IT}
        if "VERSION 2 (rewritten" in user:
            r2 = self.review2 or {"dimensions": dict(self.first_review, language="strong"), "weak_sentences": [], "fact_risks": [],
                                  "decision": "publish", "improved": True, "improvement_note": "rhythm found", "summary": "ok"}
            return r2
        if "ruthless independent editor" in system:
            self.reviews_seen += 1
            if self.reject_first_story and self.director_calls == 1:
                return {"dimensions": dict(self.first_review, idea="weak"), "weak_sentences": [], "fact_risks": [],
                        "decision": "reject_story", "summary": "the fact everyone knows"}
            dims = dict(self.first_review)
            decision = "publish"
            if dims.get("language") == "weak":
                decision = "rewrite"
            return {"dimensions": dims, "weak_sentences": [{"segment": 1, "quote": "look permanent", "reason": "flat"}] if decision == "rewrite" else [],
                    "fact_risks": [], "decision": decision, "summary": "s"}
        raise AssertionError("unexpected prompt: " + user[:90])


def _quiet():
    """The hygiene fixers are real code with their own tests; here they are silenced."""
    return (mock.patch("avp.editorial_engine.factcheck.run", return_value=SimpleNamespace(findings=[])),
            mock.patch("avp.editorial_engine.polish.proofread", side_effect=lambda s, cfg: s),
            mock.patch("avp.editorial_engine.italian._proof", return_value=0),
            mock.patch("avp.editorial_engine.italian._factcheck", return_value=[]),
            mock.patch("avp.editorial_engine.italian._meaning", return_value={}),
            mock.patch("avp.editorial_engine._backend", return_value=(mock.Mock(), "m")),
            mock.patch("avp.editorial_engine.brief_mod.build", return_value="FACT BASE\n- ring rain ~1000 kg/s (Cassini)\n- rings may be 100 Myr old"),
            mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "", "AVP_EDITOR_API_KEY": "", "AVP_EDITOR_URL": "", "AVP_EDITOR_MODEL": ""}, clear=False))


class TheEditorialMachine(unittest.TestCase):

    def _run(self, fake, tmp):
        cfg, project = _cfg(tmp), _project(tmp)
        patches = _quiet()
        with mock.patch("avp.editorial_engine.requests.post", fake), patches[0], patches[1], patches[2], patches[3], \
             patches[4], patches[5], patches[6], patches[7]:
            script = E.run(project, cfg, TOPIC)
        return script, project

    def test_a_story_is_chosen_designed_written_reviewed_and_stored(self):
        fake = FakeAPI()
        with tempfile.TemporaryDirectory() as tmp:
            script, project = self._run(fake, tmp)
            root = project.root
            for name in ("editorial_director.json", "editorial_brief.json", "narrative_arcs.json", "editorial_review_en_v1.json",
                         "editorial_review_it_v1.json", "editorial_report.json", "italian_script.json", "script.json", "script.md"):
                self.assertTrue((root / name).exists(), name)
            report = json.loads((root / "editorial_report.json").read_text())
            md = (root / "script.md").read_text()
        self.assertEqual(report["winner"]["id"], 2)                                  # the editor's choice, not the first angle
        self.assertFalse(report["independent_editor"])                             # DeepSeek in both roles: recorded
        content = [s for s in script.segments if s.kind != "cta"]
        self.assertEqual(len(content), 6)
        self.assertEqual([s.italian for s in content], IT)                          # every beat carries its Italian card
        cta = script.segments[-1]
        self.assertEqual(cta.kind, "cta")
        self.assertIn(BRIDGE_EN, cta.narration)
        self.assertIn("AstroStackerPro", cta.narration)
        self.assertEqual(cta.italian, BRIDGE_IT)                                   # the Italian bridge is the CTA card
        self.assertIn("ITALIAN: " + IT[0], md)
        self.assertEqual(fake.reviews_seen, 2)
        project.manifest.mark.assert_called_with("script", "done", title="The Rings Are Falling", segments=7)
        # the documents that define "magazine" reach the prompts
        director_sys = next(s for s, u in fake.prompts if "Generate 10 genuinely different" in u)
        writer_sys = next(s for s, u in fake.prompts if u.startswith("LANGUAGE: English"))
        review_sys = next(s for s, u in fake.prompts if "ruthless independent editor" in s)
        self.assertIn("Sette sorgenti di valore", director_sys)
        self.assertIn("Tendenze vietate", writer_sys)
        self.assertIn("Schwarzschild", review_sys)                                 # the annotated benchmark
        select_user = next(u for s, u in fake.prompts if "Choose the strongest angle" in u)
        self.assertNotIn('"id": 1', select_user)                                    # killed angles never reach the editor

    def test_a_rewrite_must_become_better_not_merely_compliant(self):
        fake = FakeAPI(first_review={"idea": "strong", "specificity": "strong", "density": "solid", "originality": "solid",
                                     "narration": "solid", "language": "weak", "ai_smell": "mild"})
        with tempfile.TemporaryDirectory() as tmp:
            script, project = self._run(fake, tmp)
            self.assertTrue((project.root / "editorial_review_en_v2.json").exists())
            self.assertTrue((project.root / "editorial_review_it_v2.json").exists())
        self.assertTrue(any("You are REVISING" in u for s, u in fake.prompts))
        self.assertTrue(any("merely comply" in u for s, u in fake.prompts))          # the second review's question
        self.assertIn("look eternal", script.segments[0].narration)                # v2 is what ships
        self.assertLessEqual(sum(len(s.narration.split()) for s in script.segments if s.kind != "cta"), 94)   # a long rewrite is cut, not refused
        # a rewrite that only complied rejects the story
        fake2 = FakeAPI(first_review={"idea": "strong", "specificity": "strong", "density": "solid", "originality": "solid",
                                      "narration": "solid", "language": "weak", "ai_smell": "mild"},
                        review2={"dimensions": {"idea": "strong", "specificity": "strong", "density": "solid", "originality": "solid",
                                                "narration": "solid", "language": "solid", "ai_smell": "none"},
                                 "weak_sentences": [], "fact_risks": [], "decision": "publish", "improved": False,
                                 "improvement_note": "adjectives removed, no idea gained", "summary": "s"})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(E.EditorialError) as ctx:
                self._run(fake2, tmp)
        self.assertIn("merely complied", str(ctx.exception))
        self.assertEqual(fake2.director_calls, 2)                                   # a second angle was tried before giving up

    def test_a_rejected_story_sends_the_director_back_with_the_rejection(self):
        fake = FakeAPI(reject_first_story=True)
        with tempfile.TemporaryDirectory() as tmp:
            script, project = self._run(fake, tmp)
            report = json.loads((project.root / "editorial_report.json").read_text())
        self.assertEqual(fake.director_calls, 2)
        self.assertEqual(report["attempt"], 2)
        second = [u for s, u in fake.prompts if "Generate 10 genuinely different" in u][1]
        self.assertIn("ANGLES ALREADY REJECTED", second)
        self.assertIn("angle id 2", second)

    def test_the_hygiene_nets_catch_errors_not_taste(self):
        cfg = _cfg(".")
        budget = E.word_budget(cfg, 6)
        self.assertEqual(budget, (90, 76, 94))                                     # (48-9)*0.92 s × 2.5 words/s; +5% only: gaps and pauses eat the rest
        good = Script(title="t", topic=TOPIC, cta_bridge=BRIDGE_EN, bridge_kind="shoot",
                      segments=[Segment(index=i, narration=l) for i, l in enumerate(EN, 1)])
        self.assertEqual(E.hygiene_en(good, cfg, budget), [])
        bad = Script(title="t", topic=TOPIC, cta_bridge="Use Halide tonight.", bridge_kind="shoot", segments=[
            Segment(index=1, narration="One thousand kilograms of ice fall onto Saturn every second, forty miles up."),
            Segment(index=2, narration="Short line."),
            Segment(index=3, narration=" ".join(["word"] * 30) + "."),
            Segment(index=4, narration=EN[3]), Segment(index=5, narration=EN[4]), Segment(index=6, narration=EN[5])])
        bad.segments[4].narration = "The rings hold about 1.5 × 10^19 kilograms of ice, roughly 40% the mass of Mimas."
        reasons = " | ".join(E.hygiene_en(bad, cfg, budget))
        for needle in ("opens on a number", "imperial", "segment 2 has 2 words", "run-on", "names another app (Halide)", "notation a voice cannot read"):
            self.assertIn(needle, reasons)
        it_bad = Script(title="t", topic=TOPIC, cta_bridge=BRIDGE_IT, segments=[
            Segment(index=i, narration=l) for i, l in enumerate(IT, 1)])
        it_bad.segments[1].narration = "La sonda ha misurato la caduta: mille chilogrammi di ghiaccio ogni secondo."   # Cassini dropped
        it_bad.segments[2].narration = "Solo un emisfero ci saluta mai."                                          # calque, and short
        it_bad.segments[3].narration = "Regola la lente di qualche scatto indietro e osserva."                     # wrong senses
        reasons = " | ".join(E.hygiene_it(good, it_bad))
        # (a sentence-initial "Cassini" is not detected as a name by design: only "Saturn" is asked back)
        for needle in ("manca il nome", "Saturn", "italiano scorretto", "senso sbagliato"):
            self.assertIn(needle, reasons)
        self.assertEqual(E.hygiene_it(good, Script(title="t", topic=TOPIC, cta_bridge=BRIDGE_IT,
                                                   segments=[Segment(index=i, narration=l) for i, l in enumerate(IT, 1)])), [])

    def test_a_script_over_budget_is_cut_to_length_not_rewritten(self):
        """The first real trial (10/09) failed here: three full rewrites, each LONGER than the last (133 → 138 →
        143 words against 126). A length problem is now a cut, done on the text itself."""
        long_lines = [l + " and this clause pads the line with words that add nothing at all" for l in EN]
        fake = FakeAPI(en_lines=long_lines)
        with tempfile.TemporaryDirectory() as tmp:
            script, project = self._run(fake, tmp)
            drafts = json.loads((project.root / "editorial_drafts.json").read_text())
        self.assertGreaterEqual(getattr(fake, "tightened", 0), 1)
        self.assertFalse(any("HYGIENE NOTES" in u for s, u in fake.prompts if u.startswith("LANGUAGE: English")))   # no blind rewrite
        content = [s for s in script.segments if s.kind != "cta"]
        self.assertLessEqual(sum(len(s.narration.split()) for s in content), 94)
        compress_users = [u for s, u in fake.prompts if "compress one spoken sentence" in s]
        self.assertTrue(compress_users and all("AT MOST" in u for u in compress_users))         # one segment, one cap
        self.assertTrue(any(d["language"] == "English" and d["reasons"] for d in drafts))                           # the record of the cut
        self.assertEqual([s.italian for s in content], IT)
        it_user = next(u for s, u in fake.prompts if u.startswith("LANGUAGE: Italian"))
        self.assertIn("LIMITI PER BATTUTA", it_user)                                       # the Italian hears the fitted beats' caps
        self.assertIn("Saturno", it_user)                                                   # names in their Italian form
        self.assertIsNone(re.search(r"nomi da conservare: Saturn(?!o)", it_user))            # never the English form
        self.assertNotIn(EN[1], it_user)                                                    # but never the English text

    def test_an_italian_beat_with_a_problem_is_repaired_alone(self):
        """Sixth Venus trial: whole-script rewrites of the Italian lost Sole and Sistema Solare in the very beats
        they were asked to fix, three times. A beat is repaired by itself, with its fact, names and length."""
        bad_it = list(IT); bad_it[1] = "La sonda ha misurato la caduta: mille chilogrammi di ghiaccio ogni secondo."   # Saturn dropped
        fake = FakeAPI()
        orig_route = fake.route
        def route(system, user):
            if user.startswith("LANGUAGE: Italian") and "You are REVISING" not in user:
                return {"title": "Gli anelli", "segments": [{"narration": l, "visual": f"visual {i}", "keywords": ["Saturno"]} for i, l in enumerate(bad_it, 1)],
                        "bridge_kind": "shoot", "cta_bridge": BRIDGE_IT}
            return orig_route(system, user)
        fake.route = route
        with tempfile.TemporaryDirectory() as tmp:
            script, project = self._run(fake, tmp)
        self.assertEqual(getattr(fake, "repaired", 0), 1)                                   # one beat, one call
        self.assertEqual([s.italian for s in script.segments if s.kind != "cta"], IT)
        repair_user = next(u for s, u in fake.prompts if "Ripari una sola battuta" in s)
        self.assertIn("Saturno", repair_user)                                               # the names in Italian form
        self.assertIn("la numero 2", repair_user)

    def test_an_arc_that_repeats_a_fact_is_caught_before_anyone_writes(self):
        """Seventh Venus trial: the chosen arc carried 6.5 km/h in three beats of six; the writer repeated it,
        the reviewer blamed the prose, the story was rejected twice. The loop belongs to the arc."""
        loop = {"id": 1, "beats": [{"beat": 1, "fact": "A point on Venus's equator moves at only about 6.5 km/h."},
                                   {"beat": 2, "fact": "Earth's equator moves at 1,670 km/h."},
                                   {"beat": 3, "fact": "A brisk walker moves at about 6.5 km/h."},
                                   {"beat": 4, "fact": "Venus rotates once every 243 Earth days."}]}
        problems = E.arc_redundancy(loop)
        self.assertTrue(any("6.5" in p for p in problems))
        good = {"id": 2, "beats": [{"beat": 1, "fact": "Venus rotates in the opposite direction to almost all planets."},
                                   {"beat": 2, "fact": "The Sun rises in the west there."},
                                   {"beat": 3, "fact": "Its axial tilt is 2.64 degrees, so it is not upside down."},
                                   {"beat": 4, "fact": "A giant impact is the leading explanation."}]}
        self.assertEqual(E.arc_redundancy(good), [])
        self.assertIn("DISTINCT BEATS", E.ARC_SELECT_SYSTEM)
        self.assertIn("EVERY BEAT CARRIES A DIFFERENT FACT", E.NARRATIVE_SYSTEM)

    def test_too_short_goes_back_to_the_writer_not_to_the_scissors(self):
        self.assertTrue(E._length_only(["total 140 spoken words, the budget is 94-116 (about 110): cut"]))
        self.assertFalse(E._length_only(["total 82 spoken words, the budget is 94-116 (about 110): add substance, not padding"]))
        self.assertFalse(E._length_only(["segmento 1: 0.5× i caratteri dell'inglese — manca contenuto della battuta"]))
        self.assertTrue(E._length_only(["segmento 2: 1.6× i caratteri dell'inglese — il lettore non arriva in fondo; stessa battuta, più asciutta"]))
        self.assertFalse(E._length_only([]))

    def test_the_directors_angles_are_read_wherever_the_model_put_them(self):
        """Eighth trial: the director answered without an "angles" key and the engine declared "no story worth
        telling" — a parsing failure is not an editorial verdict."""
        self.assertEqual(len(E._angles_from({"editorial_angles": [{"angle": "a"}, {"angle": "b"}]})), 2)
        self.assertEqual(E._angles_from({"stories": [{"angle": "a"}]})[0]["id"], 1)
        self.assertEqual(len(E._angles_from({"whatever": [{"angle": "x", "id": 4}], "note": "n"})), 1)
        self.assertEqual(E._angles_from({"angles": "not a list"}), [])
        self.assertEqual(E._angles_from("garbage"), [])

    def test_publishable_and_rejection_rules(self):
        ok = {"decision": "publish", "dimensions": {"idea": "strong", "specificity": "solid", "density": "solid", "originality": "solid",
                                                    "narration": "solid", "language": "solid", "ai_smell": "mild"}}
        self.assertTrue(E.publishable(ok))
        self.assertFalse(E.publishable({**ok, "dimensions": {**ok["dimensions"], "idea": "solid"}}))       # the idea must be strong
        self.assertFalse(E.publishable({**ok, "dimensions": {**ok["dimensions"], "ai_smell": "strong"}}))
        self.assertFalse(E.publishable({**ok, "decision": "rewrite"}))                              # first review: the editor's word
        self.assertTrue(E.publishable({**ok, "decision": "rewrite"}, after_rewrite=True))           # after a rewrite: the rubric decides
        self.assertFalse(E.publishable({**ok, "decision": "reject_story"}, after_rewrite=True))
        self.assertFalse(E.publishable({**ok, "decision": "rewrite", "dimensions": {**ok["dimensions"], "narration": "weak"}}, after_rewrite=True))
        self.assertTrue(E.story_rejected({"decision": "rewrite", "dimensions": {"idea": "weak"}}))
        self.assertTrue(E.story_rejected({"decision": "reject_story", "dimensions": {}}))
        self.assertFalse(E.story_rejected(ok))

    def test_the_stage_dispatches_to_the_engine_and_keeps_the_classic_chain(self):
        from avp import stages
        src = inspect.getsource(stages.stage_script)
        self.assertIn("editorial_engine.run(project, cfg, topic)", src)
        self.assertLess(src.index("editorial_engine.run"), src.index("brief.build"))     # the switch comes first
        self.assertIn("polish.run(script, facts, cfg", src)                               # classic chain still there
        cfg = Config()
        self.assertEqual(cfg.script.engine, "editorial")
        self.assertFalse(E.independent_editor(cfg))
        with mock.patch.dict("os.environ", {"AVP_EDITOR_URL": "https://api.mistral.ai/v1/chat/completions", "AVP_EDITOR_MODEL": "mistral-large-latest"}):
            self.assertTrue(E.independent_editor(cfg))

    def test_no_fact_sheet_no_story(self):
        fake = FakeAPI()
        with tempfile.TemporaryDirectory() as tmp:
            cfg, project = _cfg(tmp), _project(tmp)
            with mock.patch("avp.editorial_engine.requests.post", fake), \
                 mock.patch("avp.editorial_engine.brief_mod.build", return_value=None):
                with self.assertRaises(E.EditorialError) as ctx:
                    E.run(project, cfg, TOPIC)
        self.assertIn("refuses to invent facts", str(ctx.exception))
        self.assertEqual(fake.prompts, [])


if __name__ == "__main__":
    unittest.main()
