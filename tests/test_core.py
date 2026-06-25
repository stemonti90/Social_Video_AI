"""Minimal unit tests for the engine's critical pure logic (stdlib unittest, no deps).

Run: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from avp import captions, cli, ffmpeg, footage, llm, stt, tts
from avp.config import Config
from avp.models import Script, Segment
from avp.stages import emit_script_md, parse_script_md


class FootageLogic(unittest.TestCase):
    def test_is_diagram(self):
        self.assertTrue(footage._is_diagram("Spectrum map of Saturn", ""))
        self.assertFalse(footage._is_diagram("The Greatest Saturn Portrait", "a photo"))

    def test_score_prefers_relevant_photo(self):
        terms = ["saturn", "rings"]
        photo = {"title": "Saturn rings close-up", "description": "photo"}
        diagram = {"title": "Titan data map", "description": "chart"}
        self.assertGreater(footage._score(photo, terms), footage._score(diagram, terms))

    def test_score_penalizes_diagram(self):
        terms = ["mars"]
        photo = {"title": "Mars surface photo", "description": ""}
        diagram = {"title": "Mars orbit diagram", "description": "chart"}
        self.assertGreater(footage._score(photo, terms), footage._score(diagram, terms))
        self.assertLess(footage._score(diagram, terms), 0)   # diagrams get a real negative penalty

    def test_key_terms(self):
        seg = Segment(index=1, narration="x", visual="Jupiter storm", keywords=["Great Red Spot"])
        t = footage._key_terms(seg, Script(title="t", segments=[seg], topic="Jupiter"))
        self.assertIn("great", t)
        self.assertIn("jupiter", t)

    def test_safe_url_encodes_spaces(self):
        # NASA hrefs sometimes carry literal spaces (e.g. nasa_id "What is a Black Hole").
        u = "http://images-assets.nasa.gov/video/What is a Black Hole/collection.json"
        out = footage._safe_url(u)
        self.assertNotIn(" ", out)
        self.assertIn("%20", out)

    def test_safe_url_idempotent(self):
        # Already-encoded URLs must not be double-encoded (%20 -> %2520).
        u = "http://x/a%20b.jpg"
        self.assertEqual(footage._safe_url(u), u)
        self.assertEqual(footage._safe_url(footage._safe_url(u)), u)


class LlmJson(unittest.TestCase):
    def test_extract_json_plain(self):
        self.assertEqual(llm._extract_json('{"a": 1}'), {"a": 1})

    def test_extract_json_fenced_with_think(self):
        raw = '<think>reasoning</think>\n```json\n{"title": "x", "segments": []}\n```'
        self.assertEqual(llm._extract_json(raw)["title"], "x")


class SttEven(unittest.TestCase):
    def test_words_even(self):
        w = stt.words_even("one two three four", 4.0)
        self.assertEqual([x.text for x in w], ["one", "two", "three", "four"])
        self.assertAlmostEqual(w[0].start, 0.0)
        self.assertAlmostEqual(w[-1].end, 4.0)

    def test_words_even_empty(self):
        self.assertEqual(stt.words_even("", 5.0), [])

    def test_distribute_words_for_translated_karaoke(self):
        # translated subtitles get per-word karaoke: each segment's words fill ITS window, contiguous
        w = stt.distribute_words([("La sonda orbita.", 3.0), ("Scoperti oceani.", 2.0)])
        self.assertEqual([x.text for x in w], ["La", "sonda", "orbita.", "Scoperti", "oceani."])
        self.assertAlmostEqual(w[0].start, 0.0)
        self.assertAlmostEqual(w[2].end, 3.0)          # segment 1 ends at its duration
        self.assertAlmostEqual(w[3].start, 3.0)        # segment 2 starts right after
        self.assertAlmostEqual(w[-1].end, 5.0)         # fills the total
        for a, b in zip(w, w[1:]):
            self.assertAlmostEqual(a.end, b.start)     # contiguous

    def test_weighted_by_length_and_punctuation(self):
        # a long word gets more on-screen time than a short one; a comma adds a hold
        w = {x.text: (x.end - x.start) for x in stt.words_even("a milleseicentosessantacinque", 4.0)}
        self.assertLess(w["a"], w["milleseicentosessantacinque"])
        short = stt.words_even("via via via", 3.0)[0]
        held = stt.words_even("via, via via", 3.0)[0]              # trailing comma → longer hold
        self.assertGreater(held.end - held.start, short.end - short.start)
        self.assertAlmostEqual(stt.words_even("a b c", 3.0)[-1].end, 3.0)   # still fills duration

    def test_transcribe_retries_then_falls_back_to_even(self):
        # a flaky aligner (e.g. parakeet OOM under memory pressure) must be retried ONCE, then the
        # build must not block — it falls back to even timing so karaoke still renders.
        from avp.config import STTConfig
        calls = {"n": 0}
        def boom(*a, **k):
            calls["n"] += 1
            raise RuntimeError("parakeet-mlx produced no JSON output")
        orig, stt._parakeet = stt._parakeet, boom
        orig_sleep, stt.time.sleep = stt.time.sleep, lambda *_: None   # don't actually wait in tests
        try:
            cfg = STTConfig(engine="parakeet")
            words, method = stt.transcribe(Path("x.wav"), "una due tre", cfg, "it", duration=3.0)
        finally:
            stt._parakeet, stt.time.sleep = orig, orig_sleep
        self.assertEqual(calls["n"], 2)                       # tried once, retried once
        self.assertEqual(method, "even")                      # then fell back
        self.assertEqual([w.text for w in words], ["una", "due", "tre"])


class CaptionsTime(unittest.TestCase):
    def test_ass_time(self):
        self.assertEqual(captions._ass_time(0), "0:00:00.00")
        self.assertEqual(captions._ass_time(61.5), "0:01:01.50")


class ConfigDefaults(unittest.TestCase):
    def test_defaults(self):
        c = Config.load(None)
        self.assertEqual(c.script.language, "en")
        self.assertEqual(c.stt.engine, "parakeet")
        self.assertIn("youtube", c.publish.platforms)

    def test_to_dict(self):
        self.assertEqual(Config.load(None).to_dict()["video"]["width"], 1080)


class TtsLanguage(unittest.TestCase):
    def _cfg(self, lang, engine):
        c = Config.load(None)
        c.script.language = lang
        c.tts.engine = engine
        return c

    def test_only_kokoro(self):                          # Chatterbox removed → always Kokoro
        self.assertEqual([p.name for p in tts.get_providers(self._cfg("en", "both"))], ["kokoro"])
        self.assertEqual([p.name for p in tts.get_providers(self._cfg("it", "chatterbox"))], ["kokoro"])

    def test_primary_is_kokoro(self):
        self.assertEqual(tts.primary_engine(), "kokoro")   # no arg — Kokoro is the only engine

    def test_en_uses_native_voice(self):
        self.assertEqual(tts.get_providers(self._cfg("en", "kokoro"))[0].voice, "af_heart")
        self.assertEqual(tts.get_providers(self._cfg("it", "kokoro"))[0].voice, "if_sara")


class ScriptMdRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        segs = [Segment(index=1, narration="Hello world", visual="v", keywords=["a", "b"]),
                Segment(index=2, narration="Second line", visual="v2", keywords=["c"])]
        s = Script(title="My Title", segments=segs, topic="t")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "script.md"
            emit_script_md(s, p)
            base = Script(title="?", topic="t",
                          segments=[Segment(index=1, narration=""), Segment(index=2, narration="")])
            out = parse_script_md(p, base)
            self.assertEqual(out.title, "My Title")
            self.assertEqual(out.segments[0].narration, "Hello world")
            self.assertEqual(out.segments[1].keywords, ["c"])


class CliConfigSet(unittest.TestCase):
    def test_deep_merge_preserves_siblings(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.yaml"
            p.write_text("script:\n  language: en\n  target_seconds: 75\n")
            cli._config_set(str(p), {"script": {"language": "it"}})
            import yaml
            d = yaml.safe_load(p.read_text())
            self.assertEqual(d["script"]["language"], "it")
            self.assertEqual(d["script"]["target_seconds"], 75)


class WavDuration(unittest.TestCase):
    """Regression: brew's minimal ffprobe can SIGSEGV on a wav; we read the header instead."""
    def _make_wav(self, path, seconds, rate=8000):
        import wave
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"\x00\x00" * int(seconds * rate))

    def test_wav_duration_reads_header(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.wav"
            self._make_wav(p, 1.5)
            self.assertAlmostEqual(ffmpeg._wav_duration(p), 1.5, places=2)
            # ffprobe_duration must take the wave path for .wav (no subprocess, no SIGSEGV)
            self.assertAlmostEqual(ffmpeg.ffprobe_duration(p), 1.5, places=2)

    def test_wav_duration_bad_file_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.wav"
            p.write_text("not a wav")
            self.assertIsNone(ffmpeg._wav_duration(p))


class FfmpegRunRetry(unittest.TestCase):
    """Regression: brew ffmpeg transiently SIGSEGVs under load → run() retries on signal
    death but fails fast on a genuine error."""
    class _Proc:
        def __init__(self, rc):
            self.returncode = rc

    def test_retries_on_signal_then_succeeds(self):
        seq = [-11, -11, 0]  # SIGSEGV twice, then OK
        calls = {"n": 0}
        def fake_run(cmd, *a, **k):
            rc = seq[calls["n"]]
            calls["n"] += 1
            return FfmpegRunRetry._Proc(rc)
        with mock.patch.object(ffmpeg.subprocess, "run", fake_run), \
             mock.patch.object(ffmpeg.time, "sleep", lambda *_: None):
            ffmpeg.run(["-i", "x", "y"])      # must NOT raise
        self.assertEqual(calls["n"], 3)

    def test_fails_fast_on_real_error(self):
        calls = {"n": 0}
        def fake_run(cmd, *a, **k):
            calls["n"] += 1
            return FfmpegRunRetry._Proc(1)    # genuine ffmpeg error
        with mock.patch.object(ffmpeg.subprocess, "run", fake_run):
            with self.assertRaises(subprocess.CalledProcessError):
                ffmpeg.run(["-i", "x", "y"])
        self.assertEqual(calls["n"], 1)       # positive returncode → no retry


class FootageDedup(unittest.TestCase):
    """Regression: NASA repeats the same photo under different ids → dedup must use title too."""
    def test_title_key_normalizes(self):
        self.assertEqual(footage._title_key("Hubble  Takes, Mars!"),
                         footage._title_key("hubble takes mars"))

    def test_pick_dedups_same_title_different_id(self):
        cands = [
            {"nasa_id": "A1", "title": "Hubble Takes Mars Portrait", "description": "", "collection": "c", "center": "GSFC"},
            {"nasa_id": "A2", "title": "Hubble Takes Mars Portrait", "description": "", "collection": "c", "center": "GSFC"},
            {"nasa_id": "B1", "title": "Andromeda Galaxy", "description": "", "collection": "c", "center": "JPL"},
        ]
        with mock.patch.object(footage, "nasa_candidates", lambda q, media_type="image", limit=30: cands):
            used = set()
            first, rel1 = footage._pick(["mars"], used, ["mars"], "image")   # _pick → (candidate, relevance)
            self.assertEqual(first["nasa_id"], "A1")            # best relevance for "mars"
            self.assertGreater(rel1, 0.0)
            used.add(first["nasa_id"])
            used.add(footage._title_key(first["title"]))         # as _try_nasa does
            second, _ = footage._pick(["mars"], used, ["mars"], "image")
            # A1 (id used) and A2 (same title) both excluded → must move on to a different photo
            self.assertEqual(second["nasa_id"], "B1")
            self.assertNotEqual(footage._title_key(second["title"]),
                                footage._title_key(first["title"]))


class EncodeFlags(unittest.TestCase):
    """Lock the platform-friendly H.264/AAC output spec (TikTok/Reels/Shorts)."""
    def test_h264_out_platform_spec(self):
        flags = ffmpeg._h264_out(crf=20, fps=30)
        s = " ".join(flags)
        for must in ["-profile:v high", "-level 4.0", "-pix_fmt yuv420p", "-color_range tv",
                     "-colorspace bt709", "-b:a 192k", "-ar 48000", "-ac 2",
                     "-movflags +faststart"]:
            self.assertIn(must, s)
        self.assertEqual(flags[flags.index("-g") + 1], "60")     # GOP = 2 x fps
        self.assertEqual(flags[flags.index("-crf") + 1], "20")   # crf threaded through


class QualityCaptions(unittest.TestCase):
    """Regression: text must always be on screen (gap-free caption/subtitle timelines)."""
    def test_caption_timeline_gap_free(self):
        from avp.config import CaptionStyle, VideoConfig
        from avp.stt import Word
        ws = [Word("a", 0.0, 0.3), Word("b", 1.0, 1.3), Word("c", 2.0, 2.3), Word("d", 2.4, 2.7)]
        with tempfile.TemporaryDirectory() as td:
            items = captions.render_caption_pngs(ws, Path(td), CaptionStyle(), VideoConfig(), total_dur=9.0)
        self.assertEqual(items[0][1], 0.0)                  # on screen from the first frame
        self.assertAlmostEqual(items[-1][2], 9.0)           # last runs to the full duration
        for i in range(len(items) - 1):
            self.assertAlmostEqual(items[i][2], items[i + 1][1])   # zero gaps between captions

    def test_phrase_subtitles_gap_free(self):
        from avp.config import CaptionStyle, VideoConfig
        with tempfile.TemporaryDirectory() as td:
            items = captions.render_phrase_pngs(
                [("riga uno", 0.0, 3.0), ("una riga ben piu lunga da mandare a capo", 3.0, 6.0)],
                Path(td), CaptionStyle(), VideoConfig())
        self.assertEqual(items[0][2], items[1][1])          # contiguous


class CaptionFont(unittest.TestCase):
    """Regression: captions must use a bundled commercial-safe font, never macOS system fonts."""
    def test_never_resolves_system_font(self):
        for name in ("Montserrat", "Arial", "Helvetica", ""):
            p = captions._font_path(name)
            self.assertIsNotNone(p, f"no font resolved for {name!r}")
            self.assertNotIn("/System/", p)          # never proprietary macOS fonts
            self.assertTrue(p.endswith(".ttf"))

    def test_resolves_into_bundled_dir(self):
        p = captions._font_path("Montserrat")
        self.assertIn("assets/fonts", p.replace("\\", "/"))


class WikimediaLicense(unittest.TestCase):
    """Regression: never pull Non-Commercial / No-Derivatives media into a monetized video."""
    def test_rejects_nc_and_nd(self):
        for bad in ("CC BY-NC 4.0", "CC BY-NC-SA 3.0", "CC BY-ND 4.0", "Fair use"):
            self.assertFalse(footage._wm_license_ok(bad), bad)

    def test_accepts_free_commercial(self):
        for ok in ("CC BY-SA 4.0", "CC BY 3.0", "CC0", "Public domain", "PD-USGov-NASA", ""):
            self.assertTrue(footage._wm_license_ok(ok), ok)


class ConfigRobustness(unittest.TestCase):
    """Regression (stress test 2026-06-14): a typo'd key or malformed YAML must never brick the
    tool. Config.load now ignores unknown keys and errors cleanly on non-mapping config."""
    def _load(self, text):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.yaml"
            p.write_text(text)
            return Config.load(str(p))

    def test_unknown_leaf_key_ignored(self):
        c = self._load("script:\n  language: it\n  oops_typo: 1\n")
        self.assertEqual(c.script.language, "it")   # known key honored, unknown one dropped

    def test_config_set_typo_then_load_does_not_brick(self):
        # The HIGH finding: config-set with a bad key used to crash EVERY later command at load.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.yaml"
            p.write_text("script:\n  language: it\n")
            cli._config_set(str(p), {"script": {"bogus": 123}})   # writes ok (no validation there)
            c = Config.load(str(p))                                # must NOT raise
            self.assertEqual(c.script.language, "it")

    def test_malformed_non_mapping_raises_clean(self):
        for bad in ("- a\n- b\n", "just a scalar\n"):
            with self.assertRaises(RuntimeError):
                self._load(bad)

    def test_non_mapping_section_uses_defaults(self):
        c = self._load("script: 42\n")              # section is a scalar, not a mapping
        self.assertEqual(c.script.language, "en")   # falls back to defaults, no crash

    def test_empty_env_host_falls_back(self):
        with mock.patch.dict("os.environ", {"OLLAMA_HOST": ""}, clear=False):
            self.assertTrue(Config.load(None).llm.host)   # empty env must not blank the default

    def test_config_set_rejects_non_object_patch(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.yaml"
            self.assertEqual(cli.main(["config-set", "42", "--config", str(p)]), 1)
            self.assertEqual(cli.main(["config-set", "[1,2]", "--config", str(p)]), 1)
            self.assertFalse(p.exists())             # nothing written on a rejected patch


class FootageRobustness(unittest.TestCase):
    """Regression (stress test 2026-06-14): footage helpers must not crash on dirty inputs."""
    def test_safe_url_none(self):
        self.assertEqual(footage._safe_url(None), "")

    def test_key_terms_tolerates_none_and_non_str(self):
        seg = Segment(index=1, narration="n", visual="v", keywords=["Mars", None])
        script = Script(title="t", topic="Saturn rings", segments=[seg])
        terms = footage._key_terms(seg, script)
        self.assertIn("mars", terms)
        self.assertNotIn("none", terms)             # a None keyword must not leak as text
        script.topic = 12345                         # non-str topic must not crash
        self.assertIsInstance(footage._key_terms(seg, script), list)

    def test_score_on_sparse_dict(self):
        self.assertIsInstance(footage._score({}, ["mars"]), int)            # no KeyError
        self.assertIsInstance(footage._score({"title": "Mars"}, ["mars"]), int)

    def test_wm_license_rejects_spaced_no_derivatives(self):
        for bad in ("CC No Derivative Works 3.0", "Attribution-NoDerivs"):
            self.assertFalse(footage._wm_license_ok(bad), bad)


class LlmRobustness(unittest.TestCase):
    """Regression (stress test 2026-06-14): tolerate odd JSON shapes from the model."""
    def test_norm_keywords(self):
        self.assertEqual(llm._norm_keywords("Mars, Saturn"), ["Mars", "Saturn"])
        self.assertEqual(llm._norm_keywords(["Mars", None, "  "]), ["Mars"])
        self.assertEqual(llm._norm_keywords(None), [])

    def test_segment_dicts(self):
        self.assertEqual(llm._segment_dicts([1, 2]), [])                    # non-dict top level
        self.assertEqual(len(llm._segment_dicts({"segments": {"0": {"narration": "x"}}})), 1)  # dict segments
        got = llm._segment_dicts({"segments": [{"narration": "x"}, "junk"]})
        self.assertEqual(len(got), 1)                                       # stray entry dropped


class CaptionMonotonic(unittest.TestCase):
    """Regression (stress test 2026-06-14): out-of-order/bunched STT timings must still yield a
    gap-free, non-overlapping caption timeline."""
    def test_out_of_order_timings(self):
        from avp.config import CaptionStyle, VideoConfig
        from avp.stt import Word
        style = CaptionStyle()
        style.group = 1
        ws = [Word("a", 5.0, 6.0), Word("b", 1.0, 2.0), Word("c", 3.0, 4.0)]
        with tempfile.TemporaryDirectory() as td:
            items = captions.render_caption_pngs(ws, Path(td), style, VideoConfig(), total_dur=8.0)
        starts = [it[1] for it in items]
        self.assertEqual(starts[0], 0.0)
        self.assertEqual(starts, sorted(starts))               # monotonic non-decreasing
        self.assertAlmostEqual(items[-1][2], 8.0)              # last runs to total_dur
        for i in range(len(items) - 1):
            self.assertAlmostEqual(items[i][2], items[i + 1][1])   # contiguous, no overlap/gap
        for s, e in ((it[1], it[2]) for it in items):
            self.assertLessEqual(s, e)                          # every window valid


class OllamaClientPayload(unittest.TestCase):
    """Regression (2026-06-14 hang): cap num_ctx (the model's 40K default bloated memory into
    swap) and use a bounded (connect, read) timeout so a stuck/restarted Ollama can't silently
    hang a build for the old 10-minute timeout."""
    def test_payload_caps_ctx_and_uses_bounded_timeout(self):
        from avp.config import LLMConfig
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"message": {"content": "{}"}}

        def _fake_post(url, json=None, timeout=None):
            captured["json"] = json
            captured["timeout"] = timeout
            return _Resp()

        with mock.patch("avp.llm.requests.post", _fake_post):
            out = llm.OllamaClient(LLMConfig()).chat("sys", "usr")          # default model = gemma4:26b-mlx
        self.assertEqual(out, "{}")
        self.assertEqual(captured["json"]["options"]["num_ctx"], 8192)
        self.assertEqual(captured["timeout"], (10, 600))      # bounded read; gemma gets 600s for cold load
        # a non-gemma model keeps the tighter 5-min bound
        with mock.patch("avp.llm.requests.post", _fake_post):
            llm.OllamaClient(LLMConfig(model="qwen3:14b")).chat("s", "u")
        self.assertEqual(captured["timeout"], (10, 300))


class OllamaConstrainedJsonByModel(unittest.TestCase):
    """Regression (2026-06-24): Gemma collapses to an empty `{}` (format="json") or hangs (schema)
    under Ollama's constrained decoding, so we MUST NOT send `format` for gemma models — the prompt
    demands strict JSON and _extract_json digs it out of free text. qwen3 et al. KEEP constrained
    JSON (guaranteed-parseable output). This guards the per-model branch."""
    def test_helper_excludes_only_gemma(self):
        for m in ("gemma4:26b", "gemma4:31b", "Gemma3:12b", " gemma2:9b"):
            self.assertFalse(llm._supports_constrained_json(m), m)
        for m in ("qwen3:14b", "qwen3:8b", "ministral:8b", "llama3.1:8b", "mistral"):
            self.assertTrue(llm._supports_constrained_json(m), m)

    def test_gemma_omits_format_others_keep_it(self):
        from avp.config import LLMConfig

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"message": {"content": "{}"}}

        def _run(model):
            captured = {}

            def _fake_post(url, json=None, timeout=None):
                captured["json"] = json
                return _Resp()

            with mock.patch("avp.llm.requests.post", _fake_post):
                llm.OllamaClient(LLMConfig(model=model)).chat("sys", "usr", fmt="json")
            return captured["json"]

        self.assertNotIn("format", _run("gemma4:26b"))        # constraint dropped → free-text JSON
        self.assertEqual(_run("qwen3:14b").get("format"), "json")   # constraint kept


class DraftRetry(unittest.TestCase):
    """Regression (2026-06-24): gemma4:26b in free-text mode intermittently returns an EMPTY reply,
    which crashed the script stage at the first draft. _draft_script_json must retry transient duds
    (empty / unparseable / segment-less) and only raise when every attempt fails."""
    _GOOD = '{"title": "T", "segments": [{"narration": "A real fact.", "visual": "v", "keywords": ["k"]}]}'

    def _client(self, replies):
        seq = list(replies)

        class _C:
            def chat(self, system, user, fmt="json", temperature=None, num_predict=None):
                return seq.pop(0)
        return _C()

    def test_retries_past_empty_and_garbage(self):
        client = self._client(["", "   ", "not json at all", self._GOOD])
        data = llm._draft_script_json(client, "sys", "usr", attempts=4)
        self.assertEqual(llm._segment_dicts(data)[0]["narration"], "A real fact.")

    def test_raises_when_all_attempts_fail(self):
        client = self._client(["", "{}", "  ", "{}"])           # empty / segment-less only
        with self.assertRaises(RuntimeError):
            llm._draft_script_json(client, "sys", "usr", attempts=4)

    def test_retries_past_a_call_exception(self):
        """A chat() that RAISES (e.g. a cold-reload timeout) must be caught and retried warm,
        not propagated — regression for the metadata stage timing out on the 16GB MLX reload."""
        calls = {"n": 0}

        class _C:
            def chat(self, system, user, fmt="json", temperature=None, num_predict=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("Ollama request failed (read timeout=600)")
                return DraftRetry._GOOD

        data = llm._draft_script_json(_C(), "sys", "usr", attempts=4)
        self.assertEqual(calls["n"], 2)                         # retried once, then succeeded
        self.assertTrue(llm._segment_dicts(data))


class ReadTimeoutByModel(unittest.TestCase):
    """Heavy gemma (16GB MLX) gets a longer read timeout for its slow cold-load; others stay tight."""
    def test_gemma_gets_more_time(self):
        self.assertEqual(llm._read_timeout("gemma4:26b-mlx"), 600)
        self.assertEqual(llm._read_timeout("gemma4:26b"), 600)
        self.assertEqual(llm._read_timeout("qwen3:14b"), 300)
        self.assertEqual(llm._read_timeout("ministral:8b"), 300)


class BuildStagesOrder(unittest.TestCase):
    """RAM choreography: metadata runs before assemble (keep the LLM warm; assemble's unload_all evicts
    it) AND before captions, so the warm model serves captions' translation with no cold reload, then
    captions evicts it before the STT aligner — which needs the RAM or it falls back to even timing."""
    def test_metadata_before_assemble(self):
        from avp import pipeline
        s = pipeline.BUILD_STAGES
        self.assertIn("metadata", s)
        self.assertIn("assemble", s)
        self.assertLess(s.index("metadata"), s.index("assemble"))

    def test_metadata_before_captions(self):
        from avp import pipeline
        s = pipeline.BUILD_STAGES
        # translation (in captions) reuses the warm model metadata loaded → no extra cold reload
        self.assertLess(s.index("metadata"), s.index("captions"))
        self.assertLess(s.index("captions"), s.index("assemble"))


class ManifestAtomicResilient(unittest.TestCase):
    """save() must be atomic (no truncation on crash) and load_or_create must survive a corrupt
    manifest instead of making the project unrecoverable."""
    def test_save_is_atomic_and_valid(self):
        from avp.manifest import Manifest
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "proj" / "manifest.json"
            m = Manifest(p)
            m.mark("voice", "done")
            self.assertFalse((p.parent / (p.name + ".tmp")).exists())   # tmp renamed away
            self.assertTrue(p.exists())
            self.assertEqual(Manifest.load_or_create(p).state("voice"), "done")   # valid on disk

    def test_load_or_create_survives_corruption(self):
        from avp.manifest import Manifest
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.json"
            p.write_text('{"stages": {"voice": ')                       # truncated JSON
            m = Manifest.load_or_create(p)                              # must NOT raise
            self.assertEqual(m.state("voice"), "pending")               # fell back to fresh


class ChatNumPredict(unittest.TestCase):
    """num_predict bounds generation for qwen3/others, but MUST be skipped for gemma: its Ollama MLX
    runner returns an EMPTY reply when num_predict is set (verified 2026-06-24)."""
    def _opts(self, model):
        from avp.config import LLMConfig
        captured = {}

        class _Resp:
            def raise_for_status(self): pass
            def json(self): return {"message": {"content": "{}"}}

        def _fake_post(url, json=None, timeout=None):
            captured["json"] = json
            return _Resp()

        with mock.patch("avp.llm.requests.post", _fake_post):
            llm.OllamaClient(LLMConfig(model=model)).chat("s", "u", num_predict=1536)
        return captured["json"]["options"]

    def test_applied_for_qwen_skipped_for_gemma(self):
        self.assertEqual(self._opts("qwen3:14b").get("num_predict"), 1536)   # honoured
        self.assertNotIn("num_predict", self._opts("gemma4:26b-mlx"))         # skipped (empty-reply bug)
        self.assertNotIn("num_predict", self._opts("gemma4:26b"))


class FootageVerifyDownload(unittest.TestCase):
    """A 200 returning an HTML error page / truncated body must be rejected so the resolver falls
    through to the next source instead of feeding ffmpeg a broken file."""
    def test_rejects_tiny_file(self):
        from avp import footage
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "x.jpg"
            bad.write_bytes(b"<html>403</html>")                        # < 1KB
            with self.assertRaises(Exception):
                footage._verify_download(bad)


class NumberNormalization(unittest.TestCase):
    """speechText: digits → spoken Italian/English so the TTS never reads numbers letter-by-letter."""
    def test_italian_cardinals_and_units(self):
        from avp import normalize as nz
        cases = {
            "1000": "mille", "2500": "duemilacinquecento", "1665": "milleseicentosessantacinque",
            "2025": "duemilaventicinque", "21": "ventuno", "28": "ventotto", "23": "ventitré",
            "85%": "ottantacinque per cento", "3,5%": "tre virgola cinque per cento",
            "24 km": "ventiquattro chilometri", "1.000.000": "un milione",
            "€50": "cinquanta euro", "$10": "dieci dollari",
        }
        for inp, exp in cases.items():
            self.assertEqual(nz.to_speech(inp, "it"), exp, inp)

    def test_prompt_example_sentences(self):
        from avp import normalize as nz
        self.assertEqual(
            nz.to_speech("Nel 2025 un wallet ha trasformato 1000 euro in 3,5 milioni.", "it"),
            "Nel duemilaventicinque un wallet ha trasformato mille euro in tre virgola cinque milioni.")
        self.assertEqual(
            nz.to_speech("La sonda si trovava a 24 km dalla superficie.", "it"),
            "La sonda si trovava a ventiquattro chilometri dalla superficie.")

    def test_spaces_preserved_and_idempotent(self):
        from avp import normalize as nz
        once = nz.to_speech("Visto a 24 km nel 1665.", "it")
        self.assertNotIn("a24", once)                       # leading space never eaten
        self.assertEqual(nz.to_speech(once, "it"), once)    # already-spoken text is unchanged

    def test_english_and_disable_flag(self):
        from avp import normalize as nz
        self.assertEqual(nz.to_speech("1000 km", "en"), "one thousand kilometers")
        self.assertEqual(nz.segment_speech("85% of it", "it", normalize=False), "85% of it")
        self.assertNotIn("1665", nz.segment_speech("anno 1665", "it", normalize=True))


class MusicMoodClassifier(unittest.TestCase):
    """The bed mood is classified from the script tone (deterministic, auditable) — never a cheerful
    bed on an ominous script."""
    def test_maps_tone_to_mood(self):
        from avp import music
        self.assertEqual(music.classify_mood("Una collisione imminente, impatto in pochi minuti.", "it")["mood"], "tense")
        self.assertEqual(music.classify_mood("Un buco nero oscuro divora ogni cosa nel vuoto.", "it")["mood"], "dark")
        self.assertEqual(music.classify_mood("Una supernova colossale, miliardi di stelle, esplosione epica.", "it")["mood"], "cinematic")
        self.assertEqual(music.classify_mood("", "it")["mood"], "documentary")   # neutral fallback

    def test_decision_is_auditable(self):
        from avp import music
        d = music.classify_mood("Un buco nero misterioso.", "it")
        self.assertIn("rationale", d)
        self.assertIn("params", d)
        self.assertIn("bpm", d["params"])           # musical params present for logging
        self.assertIn(d["mood"], music.PROMPTS)      # every mood has a Stable Audio prompt


class FootageRelevanceFloor(unittest.TestCase):
    """Visual↔segment relevance is normalized 0-1 and gated by a floor; a generic filler scores low."""
    def test_relevance_normalized_and_ranks(self):
        from avp import footage as F
        terms = ["jupiter", "storm"]
        hi = F._relevance({"title": "jupiter great red spot storm", "description": ""}, terms)
        lo = F._relevance({"title": "deep space nebula", "description": ""}, terms)
        self.assertTrue(0.0 <= lo <= hi <= 1.0)
        self.assertGreater(hi, lo)
        self.assertGreaterEqual(hi, 0.9)                 # both terms in the title
        self.assertEqual(F._relevance({}, []), 0.5)      # no terms → neutral, never blocks

    def test_diagram_is_capped_below_photo(self):
        from avp import footage as F
        terms = ["jupiter", "storm"]
        diagram = F._relevance({"title": "jupiter storm diagram chart", "description": ""}, terms)
        photo = F._relevance({"title": "jupiter storm closeup", "description": ""}, terms)
        self.assertLess(diagram, photo)


class JudgeBest(unittest.TestCase):
    """best-of-N: the model picks the strongest draft; judging must never lose a usable script."""
    _DRAFTS = [{"title": "A", "segments": [{"narration": "a"}]},
               {"title": "B", "segments": [{"narration": "b"}]}]

    def test_picks_judged_draft(self):
        class _C:
            def chat(self, system, user, fmt="json", temperature=None, num_predict=None):
                return '{"best": 2, "why": "B has a stronger hook"}'
        self.assertEqual(llm._judge_best(_C(), self._DRAFTS, "topic", "it")["title"], "B")

    def test_single_draft_passthrough(self):
        self.assertEqual(llm._judge_best(None, [{"title": "only"}], "t", "it")["title"], "only")

    def test_falls_back_to_first_on_bad_verdict(self):
        class _C:
            def chat(self, *a, **k):
                return "not json at all"
        self.assertEqual(llm._judge_best(_C(), self._DRAFTS, "t", "it")["title"], "A")


class AbRecommend(unittest.TestCase):
    """The A/B harness chooses fp16 only when it's measurably faster AND lighter; otherwise the safe
    default — readable without looking at the code."""
    def test_fp16_chosen_when_faster_and_lighter(self):
        from avp import abtest
        r = abtest.recommend_fp16([
            {"variant": "float32", "ok": True, "seconds": 120, "peak_mb": 6000},
            {"variant": "float16", "ok": True, "seconds": 90, "peak_mb": 3200},
        ])
        self.assertEqual(r["choice"], "float16")

    def test_fp32_when_fp16_not_better(self):
        from avp import abtest
        r = abtest.recommend_fp16([
            {"variant": "float32", "ok": True, "seconds": 100, "peak_mb": 6000},
            {"variant": "float16", "ok": True, "seconds": 130, "peak_mb": 3000},   # lighter but slower
        ])
        self.assertEqual(r["choice"], "float32")

    def test_all_failed_keeps_current(self):
        from avp import abtest
        self.assertIsNone(abtest.recommend_fp16([{"variant": "float16", "ok": False}])["choice"])
        self.assertIsNone(abtest.recommend_voice([{"variant": "current", "ok": False}])["choice"])

    def test_voice_picks_fastest_clean_render(self):
        from avp import abtest
        r = abtest.recommend_voice([
            {"variant": "current", "ok": True, "seconds": 9, "audio_seconds": 3.1},
            {"variant": "slower", "ok": True, "seconds": 12, "audio_seconds": 3.6},
        ])
        self.assertEqual(r["choice"], "current")


class CliDelete(unittest.TestCase):
    """delete removes a real project folder but refuses unsafe slugs and non-project dirs."""
    def _cfg(self, projects_dir):
        c = Config.load(None)
        c.paths.projects_dir = str(projects_dir)
        return c

    def test_deletes_real_project(self):
        with tempfile.TemporaryDirectory() as td:
            for slug in ("demo", "_smoke", "saturn-rings-2"):   # incl. leading-underscore projects
                proj = Path(td) / slug
                (proj / "footage").mkdir(parents=True)
                (proj / "manifest.json").write_text("{}")
                (proj / "script.md").write_text("x")
                self.assertEqual(cli._cmd_delete(self._cfg(td), slug), 0, slug)
                self.assertFalse(proj.exists(), slug)

    def test_refuses_unsafe_slugs_and_keeps_files_outside(self):
        with tempfile.TemporaryDirectory() as td:
            outside = Path(td) / "secret"
            outside.mkdir()
            (outside / "keep.txt").write_text("important")
            projects = Path(td) / "projects"
            projects.mkdir()
            cfg = self._cfg(projects)
            for bad in ("../secret", "..", "a/b", "", "Demo", "x;rm", "a b"):
                self.assertEqual(cli._cmd_delete(cfg, bad), 1, bad)
            self.assertTrue(outside.exists() and (outside / "keep.txt").exists())

    def test_refuses_dir_without_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "notaproject"
            d.mkdir()
            (d / "random.txt").write_text("x")
            self.assertEqual(cli._cmd_delete(self._cfg(td), "notaproject"), 1)
            self.assertTrue(d.exists())                 # untouched: not a real project

    def test_missing_project_is_clean_error(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(cli._cmd_delete(self._cfg(td), "ghost"), 1)


class MakeClipFallback(unittest.TestCase):
    """Regression (2026-06-15): zoompan (Ken Burns) can SIGSEGV under memory pressure even after
    retries; make_clip must fall back to a static clip so the assemble stage never dies."""
    def test_falls_back_to_static_on_zoompan_failure(self):
        from PIL import Image
        calls = []

        def fake_run(args, retries=6):
            calls.append(args)
            if any("zoompan" in str(a) for a in args):
                raise subprocess.CalledProcessError(-11, ["ffmpeg"])   # simulate SIGSEGV
            return None

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "01.jpg"
            Image.new("RGB", (1200, 800), (20, 30, 60)).save(src)      # real image: PIL prep needs it
            with mock.patch("avp.ffmpeg.run", fake_run):
                ffmpeg.make_clip(src, 6.0, 1080, 1920, 30, True, Path(td) / "out.mp4")
        self.assertEqual(len(calls), 2)                                  # zoompan try + static fallback
        self.assertTrue(any("zoompan" in str(a) for a in calls[0]))      # first attempt = Ken Burns
        self.assertFalse(any("zoompan" in str(a) for a in calls[1]))     # fallback = static (PIL-scaled)


class ExportOutputs(unittest.TestCase):
    """Per-project Desktop share folder (paths.export_dir): copies mp4 + metadata, disableable, safe."""
    def _project(self, td):
        projects = Path(td) / "projects"
        (projects / "demo").mkdir(parents=True)
        (projects / "demo" / "demo.mp4").write_bytes(b"VIDEO")
        (projects / "demo" / "metadata.md").write_text("meta")
        (projects / "demo" / "metadata.json").write_text("{}")
        cfg = Config.load(None)
        cfg.paths.projects_dir = str(projects)
        from avp.manifest import VideoProject as VP
        return VP("demo", cfg), cfg

    def test_copies_video_and_metadata(self):
        from avp import stages
        with tempfile.TemporaryDirectory() as td:
            project, cfg = self._project(td)
            cfg.paths.export_dir = str(Path(td) / "out")
            dest = stages.export_outputs(project, cfg)
            self.assertEqual(dest, Path(td) / "out" / "demo")
            for f in ("demo.mp4", "metadata.md", "metadata.json"):
                self.assertTrue((Path(td) / "out" / "demo" / f).exists(), f)

    def test_disabled_when_empty(self):
        from avp import stages
        with tempfile.TemporaryDirectory() as td:
            project, cfg = self._project(td)
            cfg.paths.export_dir = ""
            self.assertIsNone(stages.export_outputs(project, cfg))

    def test_expands_tilde(self):
        from avp import config as cfgmod
        self.assertTrue(cfgmod.PathsConfig().export_dir.startswith("~/"))   # default lives under HOME


class FfmpegBinResolve(unittest.TestCase):
    """Regression (2026-06-15): GUI-launched apps (Finder/Dock, packaged app) inherit a minimal
    PATH without Homebrew, so a bare 'ffmpeg' raises FileNotFoundError. _bin() must resolve via
    fallback dirs so the engine works however it's launched."""
    def test_resolves_from_fallback_dir_when_not_on_path(self):
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "ffmpeg"
            fake.write_text("#!/bin/sh\n")
            fake.chmod(0o755)
            ffmpeg._bin.cache_clear()
            with mock.patch("avp.ffmpeg.shutil.which", return_value=None), \
                 mock.patch("avp.ffmpeg._FALLBACK_BINDIRS", (td,)):
                self.assertEqual(ffmpeg._bin("ffmpeg"), str(fake))
            ffmpeg._bin.cache_clear()

    def test_prefers_path_when_available(self):
        ffmpeg._bin.cache_clear()
        with mock.patch("avp.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg"):
            self.assertEqual(ffmpeg._bin("ffmpeg"), "/usr/bin/ffmpeg")
        ffmpeg._bin.cache_clear()


class EndcardRender(unittest.TestCase):
    """The redesigned end card must render a valid full-frame image (brand chip + CTA button)."""
    def test_renders_valid_full_frame(self):
        from avp.config import FunnelConfig, VideoConfig
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "end.png"
            captions.render_endcard(p, FunnelConfig(), VideoConfig())
            self.assertTrue(p.exists())
            self.assertEqual(Image.open(p).size, (VideoConfig().width, VideoConfig().height))


class OllamaUnloadAll(unittest.TestCase):
    """Before assemble, free ALL loaded Ollama models (incl. unrelated big ones) so nothing
    competes for RAM with ffmpeg. Reversible — each reloads on demand."""
    def test_unloads_each_loaded_model(self):
        from avp.config import LLMConfig
        posts, ps = [], {"n": 0}

        class _R:
            def __init__(self, p): self._p = p
            def json(self): return self._p

        def fake_get(url, timeout=None):
            ps["n"] += 1
            return _R({"models": [{"name": "qwen3:14b"}, {"name": "knightfall:latest"}]}) if ps["n"] == 1 \
                else _R({"models": []})          # after the unloads, RAM is clear

        def fake_post(url, json=None, timeout=None):
            posts.append(json["model"])
            return _R({})

        with mock.patch("avp.llm.requests.get", fake_get), mock.patch("avp.llm.requests.post", fake_post):
            llm.unload_all(LLMConfig())
        self.assertEqual(sorted(posts), ["knightfall:latest", "qwen3:14b"])   # both evicted


class DedupeSegments(unittest.TestCase):
    """Coherence guard: the local model sometimes pads to the segment count by repeating its
    payoff line (e.g. identical segments 8 & 9) — dedupe_segments drops the repeats + re-indexes."""
    def test_removes_duplicates_and_reindexes(self):
        from avp.models import Segment, dedupe_segments
        segs = [
            Segment(index=1, narration="Il 27% dell'universo è materia oscura."),
            Segment(index=2, narration="Le curve di rotazione rivelano massa mancante."),
            Segment(index=3, narration="Il 27% invisibile struttura l'universo come un telaio di filamenti."),
            Segment(index=4, narration="Il 27% invisibile struttura l'universo come un telaio di filamenti."),  # dup
            Segment(index=5, narration="Want to capture the cosmos yourself? Get AstroStackerPro."),
        ]
        out = dedupe_segments(segs)
        self.assertEqual([s.narration for s in out],
                         [segs[0].narration, segs[1].narration, segs[2].narration, segs[4].narration])
        self.assertEqual([s.index for s in out], [1, 2, 3, 4])      # contiguous re-index

    def test_keeps_distinct_similar_segments(self):
        from avp.models import Segment, dedupe_segments
        segs = [Segment(index=1, narration="La materia oscura forma il telaio delle galassie."),
                Segment(index=2, narration="La lente gravitazionale distorce la luce lontana.")]
        self.assertEqual(len(dedupe_segments(segs)), 2)            # different → both kept


class PublishPostiz(unittest.TestCase):
    """The Postiz client must send each provider's required `settings` (verified vs docs.postiz.com)
    and the documented post body — sending no settings is why the old client could never post."""

    def _meta(self):
        return {"youtube": {"title": "Cassini", "description": "desc", "tags": ["space", "nasa"]},
                "tiktok": {"caption": "tok #space"}, "instagram": {"caption": "ig caption"}}

    def test_canon_aliases(self):
        from avp import publish
        self.assertEqual(publish._canon("YT"), "youtube")
        self.assertEqual(publish._canon("reels"), "instagram")
        self.assertEqual(publish._canon("tt"), "tiktok")

    def test_settings_have_required_fields_per_platform(self):
        from avp import publish
        from avp.config import PublishConfig
        pub = PublishConfig()
        yt = publish._settings_for("youtube", self._meta(), pub, False)
        self.assertEqual(yt["__type"], "youtube")
        self.assertEqual(yt["type"], "public")
        self.assertGreaterEqual(len(yt["title"]), 2)
        self.assertEqual(yt["tags"][0], {"value": "space", "label": "space"})   # YouTube tag shape
        tk = publish._settings_for("tiktok", self._meta(), pub, False)
        for k in ("__type", "privacy_level", "duet", "stitch", "comment", "autoAddMusic",
                  "brand_content_toggle", "brand_organic_toggle", "content_posting_method"):
            self.assertIn(k, tk)
        self.assertEqual(tk["privacy_level"], "PUBLIC_TO_EVERYONE")
        ig = publish._settings_for("instagram", self._meta(), pub, False)
        self.assertEqual(ig, {"__type": "instagram", "post_type": "post"})

    def test_disclose_ai_flows_to_tiktok_flag(self):
        from avp import publish
        from avp.config import PublishConfig
        self.assertFalse(publish._settings_for("tiktok", self._meta(), PublishConfig(), False)["video_made_with_ai"])
        self.assertTrue(publish._settings_for("tiktok", self._meta(), PublishConfig(), True)["video_made_with_ai"])

    def test_privacy_maps_per_platform(self):
        from avp import publish
        from avp.config import PublishConfig
        pub = PublishConfig(privacy="private")
        self.assertEqual(publish._settings_for("tiktok", self._meta(), pub, False)["privacy_level"], "SELF_ONLY")
        self.assertEqual(publish._settings_for("youtube", self._meta(), pub, False)["type"], "private")

    def test_integration_id_config_wins_then_discovery(self):
        from avp import publish
        self.assertEqual(publish._integration_id("youtube", {"youtube": "cfg1"}, {"youtube": "disc"}), "cfg1")
        self.assertEqual(publish._integration_id("youtube", {}, {"youtube": "disc"}), "disc")
        self.assertIsNone(publish._integration_id("tiktok", {}, {"youtube": "disc"}))

    def test_create_post_body_matches_documented_schema(self):
        from avp import publish
        from avp.config import PublishConfig
        captured = {}
        class _Resp:
            def raise_for_status(self): pass
            def json(self): return {"id": "post1"}
        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"], captured["headers"], captured["body"] = url, headers, json
            return _Resp()
        orig, publish.requests.post = publish.requests.post, fake_post
        try:
            c = publish.PostizClient(PublishConfig(postiz_token="k"))
            settings = {"__type": "tiktok", "privacy_level": "PUBLIC_TO_EVERYONE"}
            c.create_post("intg1", "hello", {"id": "m1", "path": "http://x/m.mp4"}, settings, None)
        finally:
            publish.requests.post = orig
        self.assertEqual(captured["headers"]["Authorization"], "k")          # raw key, no Bearer
        self.assertTrue(captured["url"].endswith("/public/v1/posts"))
        b = captured["body"]
        self.assertEqual(b["type"], "now")
        self.assertIn("date", b)                                             # date sent even for "now"
        post = b["posts"][0]
        self.assertEqual(post["integration"]["id"], "intg1")
        self.assertEqual(post["value"][0]["content"], "hello")
        self.assertEqual(post["value"][0]["image"], [{"id": "m1", "path": "http://x/m.mp4"}])
        self.assertEqual(post["settings"], settings)

    def test_create_post_schedule_sets_type_and_date(self):
        from avp import publish
        from avp.config import PublishConfig
        captured = {}
        class _Resp:
            def raise_for_status(self): pass
            def json(self): return {}
        def fake_post(url, headers=None, json=None, timeout=None):
            captured["body"] = json; return _Resp()
        orig, publish.requests.post = publish.requests.post, fake_post
        try:
            c = publish.PostizClient(PublishConfig(postiz_token="k"))
            c.create_post("i", "c", {"id": "m"}, {"__type": "youtube"}, "2026-07-01T18:00:00.000Z")
        finally:
            publish.requests.post = orig
        self.assertEqual(captured["body"]["type"], "schedule")
        self.assertEqual(captured["body"]["date"], "2026-07-01T18:00:00.000Z")

    def test_discover_maps_identifier_to_id(self):
        from avp import publish
        from avp.config import PublishConfig
        class _Resp:
            def raise_for_status(self): pass
            def json(self): return [{"id": "a", "identifier": "youtube"},
                                    {"id": "b", "identifier": "tiktok"}]
        orig, publish.requests.get = publish.requests.get, lambda *a, **k: _Resp()
        try:
            disc = publish._discover(publish.PostizClient(PublishConfig(postiz_token="k")))
        finally:
            publish.requests.get = orig
        self.assertEqual(disc, {"youtube": "a", "tiktok": "b"})


if __name__ == "__main__":
    unittest.main()
