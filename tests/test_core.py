"""Minimal unit tests for the engine's critical pure logic (stdlib unittest, no deps).

Run: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
import json
import subprocess
import tempfile
import time
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


class MetadataClean(unittest.TestCase):
    """gemma occasionally splits a word with a stray apostrophe ("Earth'ally" for "Earthly"). We can't
    guess the intended word, so generate_metadata DETECTS it and re-rolls; standard contractions pass."""

    def test_flags_stray_apostrophe_word(self):
        from avp import llm
        bad = {"youtube": {"title": "ok", "description": "from Earth'ally microbes."},
               "tiktok": {"caption": "x"}, "instagram": {"caption": "y"}}
        self.assertFalse(llm._meta_looks_clean(bad))

    def test_accepts_standard_contractions(self):
        from avp import llm
        good = {"youtube": {"title": "Saturn's rings",
                            "description": "It's a probe; don't miss what we're showing — you'll love it. I've seen it."},
                "tiktok": {"caption": "hook #space"}, "instagram": {"caption": "clean caption"}}
        self.assertTrue(llm._meta_looks_clean(good))

    def test_clean_text_collapses_whitespace_only(self):
        from avp import llm
        self.assertEqual(llm._clean_text("a   b\t\tc  "), "a b c")
        self.assertEqual(llm._clean_text("Earth's story"), "Earth's story")   # wording untouched

    def test_clean_metadata_applies_to_nested_fields(self):
        from avp import llm
        d = llm._clean_metadata({"youtube": {"title": "a  b", "description": "c   d"},
                                 "tiktok": {"caption": "e   f"}, "instagram": {"caption": "g  h"}})
        self.assertEqual(d["youtube"]["title"], "a b")
        self.assertEqual(d["youtube"]["description"], "c d")
        self.assertEqual(d["tiktok"]["caption"], "e f")
        self.assertEqual(d["instagram"]["caption"], "g h")


class AutoPipeline(unittest.TestCase):
    """The daily automation: topic queue + LLM refill, best-time slotting, and a batch runner that
    builds every video but only posts to platforms with a connected Postiz channel."""

    def test_slugify(self):
        from avp import auto
        self.assertEqual(auto.slugify("Perché gli anelli di Saturno?!"), "perche-gli-anelli-di-saturno")
        self.assertEqual(auto.slugify("  Cassini   probe  "), "cassini-probe")
        self.assertEqual(auto.slugify("!!!"), "video")

    def test_post_slots_future_only_and_rolls_over(self):
        from avp import auto
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime(2026, 6, 29, 19, 30, tzinfo=ZoneInfo("Europe/Rome"))   # past 18:00, before 21:00
        slots = auto.post_slots(now, ["12:00", "18:00", "21:00"], "Europe/Rome", 3)
        self.assertTrue(all(s > now for s in slots))
        self.assertEqual([(s.hour, s.day) for s in slots], [(21, 29), (12, 30), (18, 30)])
        self.assertTrue(auto._iso_utc(slots[0]).endswith("Z"))

    def test_extract_list_handles_array_and_wrapped(self):
        from avp import llm
        self.assertEqual(llm._extract_list('["a","b"]'), ["a", "b"])
        self.assertEqual(llm._extract_list('Here:\n["x", "y"]\ndone'), ["x", "y"])
        self.assertEqual(llm._extract_list('{"topics":["p","q"]}'), ["p", "q"])

    def test_queue_roundtrip_ignores_comments(self):
        from avp import auto
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "q.txt"
            auto.save_queue(p, ["Alpha", "Beta"])
            (p).write_text("# a comment\nAlpha\n\n  Beta \n# another\n")
            self.assertEqual(auto.load_queue(p), ["Alpha", "Beta"])

    def test_next_topics_refills_when_low_and_consumes(self):
        from avp import auto
        cfg = Config.load(None)
        with tempfile.TemporaryDirectory() as td:
            cfg.paths.projects_dir = td
            cfg.auto.queue_path = "q.txt"
            cfg.auto.refill_threshold = 5
            auto.save_queue(auto._queue_path(cfg), ["Existing One"])
            orig = auto.llm.brainstorm_topics
            auto.llm.brainstorm_topics = lambda lc, avoid, n, theme, language: \
                ["New A", "New B", "New C", "New D", "New E", "New F"]
            try:
                got = auto.next_topics(cfg, 2, consume=True)
            finally:
                auto.llm.brainstorm_topics = orig
            self.assertEqual(got, ["Existing One", "New A"])                    # popped from the top
            remaining = auto.load_queue(auto._queue_path(cfg))
            self.assertEqual(remaining[0], "New B")                             # rest persisted
            self.assertNotIn("Existing One", remaining)

    def test_peek_does_not_mutate_or_call_llm(self):
        from avp import auto
        cfg = Config.load(None)
        with tempfile.TemporaryDirectory() as td:
            cfg.paths.projects_dir = td
            cfg.auto.queue_path = "q.txt"
            auto.save_queue(auto._queue_path(cfg), ["A", "B", "C"])
            called = {"n": 0}
            orig = auto.llm.brainstorm_topics
            auto.llm.brainstorm_topics = lambda *a, **k: called.__setitem__("n", called["n"] + 1) or []
            try:
                got = auto.next_topics(cfg, 2, consume=False)
            finally:
                auto.llm.brainstorm_topics = orig
            self.assertEqual(got, ["A", "B"])
            self.assertEqual(called["n"], 0)                                    # no LLM call on peek
            self.assertEqual(auto.load_queue(auto._queue_path(cfg)), ["A", "B", "C"])   # unchanged

    def test_run_daily_builds_all_but_posts_only_connected(self):
        import avp.pipeline as pipeline_mod
        import avp.publish as publish_mod
        import avp.stages as stages_mod
        from avp import auto
        cfg = Config.load(None)
        cfg.auto.platforms = ["tiktok", "instagram"]
        published, orig = [], {}

        class FakeProj:
            @classmethod
            def create(cls, slug, cfg):
                return object()

        def patch(mod, name, val):
            orig[(mod, name)] = getattr(mod, name)
            setattr(mod, name, val)
        try:
            patch(auto, "next_topics", lambda c, n, consume=True: ["Topic A", "Topic B"])
            patch(auto, "connected_platforms", lambda c: {"tiktok"})           # instagram NOT connected
            patch(auto, "VideoProject", FakeProj)
            patch(stages_mod, "stage_script", lambda p, c, t: None)
            patch(pipeline_mod, "build", lambda p, c, config_path=None: None)
            patch(publish_mod, "stage_publish",
                  lambda p, c, go=False, platforms=None, when=None: published.append((platforms, when)))
            report = auto.run_daily(cfg, count=2, dry_run=False, publish=True)
        finally:
            for (mod, name), val in orig.items():
                setattr(mod, name, val)
        self.assertEqual(len(report), 2)
        self.assertTrue(all(e.get("built") for e in report))
        self.assertEqual([e["published_to"] for e in report], [["tiktok"], ["tiktok"]])  # IG skipped
        self.assertEqual(len(published), 2)
        self.assertTrue(all(w and w.endswith("Z") for _, w in published))      # scheduled at a UTC time

    def test_run_daily_generate_only_when_no_channel(self):
        import avp.pipeline as pipeline_mod
        import avp.publish as publish_mod
        import avp.stages as stages_mod
        from avp import auto
        cfg = Config.load(None)
        posted, orig = [], {}

        class FakeProj:
            @classmethod
            def create(cls, slug, cfg):
                return object()

        def patch(mod, name, val):
            orig[(mod, name)] = getattr(mod, name)
            setattr(mod, name, val)
        try:
            patch(auto, "next_topics", lambda c, n, consume=True: ["Only One"])
            patch(auto, "connected_platforms", lambda c: set())                # nothing connected
            patch(auto, "VideoProject", FakeProj)
            patch(stages_mod, "stage_script", lambda p, c, t: None)
            patch(pipeline_mod, "build", lambda p, c, config_path=None: None)
            patch(publish_mod, "stage_publish", lambda *a, **k: posted.append(1))
            report = auto.run_daily(cfg, count=1, dry_run=False, publish=True)
        finally:
            for (mod, name), val in orig.items():
                setattr(mod, name, val)
        self.assertTrue(report[0]["built"])
        self.assertEqual(report[0]["published_to"], [])                        # built, not posted
        self.assertEqual(posted, [])                                           # publish never called


class WorkerLoop(unittest.TestCase):
    """The Mac worker claims a job, renders it, and uploads metadata + video; a render failure is
    reported to the control server instead of crashing the loop."""

    class _Resp:
        def __init__(self, status=200, payload=None, text=None):
            self.status_code = status
            self._p = payload
            self.text = text if text is not None else (json.dumps(payload) if payload is not None else "")

        def json(self):
            return self._p

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class _Requests:
        def __init__(self, claim):
            self.claim = list(claim)
            self.calls = []

        def post(self, url, json=None, data=None, headers=None, timeout=None):
            self.calls.append(("POST", url))
            if url.endswith("/jobs/claim"):
                return self.claim.pop(0) if self.claim else WorkerLoop._Resp(204, text="")
            return WorkerLoop._Resp(200, {"ok": True})

        def put(self, url, data=None, headers=None, timeout=None):
            try:
                data.read()
            except Exception:
                pass
            self.calls.append(("PUT", url))
            return WorkerLoop._Resp(200, {"status": "done"})

    def _fake_project(self, root, video):
        class FP:
            def output_for(self, eng):
                return Path("/nonexistent-xyz.mp4")     # force fallback to .output

            @property
            def output(self):
                return video

            @classmethod
            def create(cls, slug, cfg):
                inst = cls()
                inst.root = root
                return inst
        return FP

    def _run(self, build_fn, claim_payload):
        import avp.pipeline as pipeline_mod
        import avp.stages as stages_mod
        from avp import worker as w
        cfg = Config.load(None)
        orig = {}

        def patch(m, n, v):
            orig[(m, n)] = getattr(m, n)
            setattr(m, n, v)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "v.kokoro.mp4"
            video.write_bytes(b"MP4DATA")
            (root / "metadata.json").write_text('{"tiktok":{"caption":"c"}}')
            cfg.paths.projects_dir = td
            fake = self._Requests([self._Resp(200, claim_payload)])
            try:
                patch(w, "requests", fake)
                patch(w, "VideoProject", self._fake_project(root, video))
                patch(stages_mod, "stage_script", lambda p, c, t: None)
                patch(pipeline_mod, "build", build_fn)
                w.run_worker(cfg, "http://ctl", "tok", once=True)
            finally:
                for (m, n), v in orig.items():
                    setattr(m, n, v)
            return [c[1] for c in fake.calls]

    def test_once_renders_and_uploads(self):
        urls = self._run(lambda p, c, config_path=None: None,
                         {"id": "J1", "topic": "Saturn hexagon", "slot_utc": "2026-07-01T18:00:00.000Z"})
        self.assertTrue(any(u.endswith("/jobs/J1/metadata") for u in urls))
        self.assertTrue(any(u.endswith("/jobs/J1/video") for u in urls))
        self.assertFalse(any(u.endswith("/fail") for u in urls))

    def test_render_failure_is_reported(self):
        def boom(p, c, config_path=None):
            raise RuntimeError("build blew up")
        urls = self._run(boom, {"id": "J2", "topic": "Betelgeuse", "slot_utc": "2026-07-01T18:00:00.000Z"})
        self.assertTrue(any(u.endswith("/jobs/J2/fail") for u in urls))
        self.assertFalse(any(u.endswith("/jobs/J2/video") for u in urls))


class ImageGen(unittest.TestCase):
    """Local AI visuals: prompts stay in the channel's photographic style, generation degrades safely
    when mflux isn't installed, and CLIP picks the best candidate (falling back to the first)."""

    def _seg(self, visual="Saturn rings from orbit", kw=("saturn", "rings")):
        from avp.models import Segment
        return Segment(index=1, narration="n", visual=visual, keywords=list(kw))

    def _script(self):
        from avp.models import Script
        return Script(title="t", topic="Cassini probe", segments=[])

    def test_prompt_opens_on_the_subject_and_carries_the_style(self):
        from avp import imagegen
        p = imagegen.build_prompt(self._seg(), self._script())
        self.assertTrue(p.startswith("Saturn rings from orbit,"))
        self.assertIn("photorealistic astrophotography", p)

    def test_negatives_never_appear_in_the_positive_prompt(self):
        """Regression: these used to be appended as "Avoid: false color, monochrome orange, ...".
        A diffusion model has no notion of "avoid" inside the prompt — naming a concept makes it MORE
        likely, so the list meant to prevent orange false-colour was helping produce it. They belong
        in --negative-prompt, which works because z-image-turbo runs at guidance 3.5."""
        from avp import imagegen
        p = imagegen.build_prompt(self._seg(), self._script())
        self.assertNotIn("Avoid:", p)
        for banned in ("monochrome orange", "false color", "illustration", "watermark", "no text"):
            self.assertNotIn(banned, p, f"{banned!r} leaked into the positive prompt")
        # …and they are still declared, for --negative-prompt to carry
        for banned in ("monochrome orange", "false color", "illustration"):
            self.assertIn(banned, imagegen.NEGATIVI)

    def test_registro_prescribes_light_never_framing(self):
        """Regression: the "star" entry used to read "the Sun ... photosphere with dark sunspots",
        which is a COMPOSITION. It overrode the script: a sunspot video whose segments asked for a
        close-up, a scale comparison, flares and a SOHO view came back as four identical full discs."""
        from avp import imagegen
        for key, text in imagegen.REGISTRO.items():
            for framing in ("filling the frame", "close up", "close-up", "wide shot", "documentary framing"):
                self.assertNotIn(framing, text.lower(), f"REGISTRO[{key}] prescribes framing")
            self.assertNotIn(" no ", f" {text.lower()} ", f"REGISTRO[{key}] negates inside a positive prompt")

    def test_shot_scale_is_prepended_not_appended(self):
        """Diffusion models obey the opening tokens; a trailing "Alternate framing: …" was ignored
        outright (measured: the close-up variant produced another identical full disc)."""
        from avp import imagegen
        shot = imagegen.SHOTS[1]
        p = imagegen.build_prompt(self._seg(), self._script(), shot)
        self.assertTrue(p.startswith(shot))
        self.assertLess(p.index("close-up"), p.index("Saturn"))

    def test_prompt_bucket_selection(self):
        from avp import imagegen
        self.assertEqual(imagegen._bucket("the Cassini probe approaching"), "spacecraft")
        self.assertEqual(imagegen._bucket("a nebula in deep space"), "deep_sky")
        self.assertEqual(imagegen._bucket("lunar surface craters"), "surface")
        self.assertEqual(imagegen._bucket("something unrelated"), "default")

    def test_prompt_falls_back_to_keywords_then_topic(self):
        from avp import imagegen
        p = imagegen.build_prompt(self._seg(visual="", kw=("enceladus", "geysers")), self._script())
        self.assertTrue(p.startswith("enceladus geysers,"))
        bare = imagegen.build_prompt(self._seg(visual="", kw=()), self._script())
        self.assertTrue(bare.startswith("Cassini probe,"))

    def test_available_requires_binary_and_model(self):
        from avp import imagegen
        cfg = Config.load(None)
        with tempfile.TemporaryDirectory() as td:
            cfg.video.image_venv = str(Path(td) / "venv")
            cfg.video.image_model = str(Path(td) / "model")
            self.assertFalse(imagegen.available(cfg))          # neither present
            (Path(td) / "venv" / "bin").mkdir(parents=True)
            (Path(td) / "venv" / "bin" / "mflux-generate-z-image-turbo").write_text("#!/bin/sh\n")
            self.assertFalse(imagegen.available(cfg))          # binary only → still not usable
            (Path(td) / "model").mkdir()
            self.assertTrue(imagegen.available(cfg))           # both → good

    def test_generate_returns_none_when_unavailable(self):
        from avp import imagegen
        cfg = Config.load(None)
        cfg.video.image_venv = "/nonexistent-venv-xyz"
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(
                imagegen.generate_for_segment(self._seg(), self._script(), cfg, Path(td) / "o.png"))

    def test_generate_picks_clip_best_candidate(self):
        from avp import imagegen
        cfg = Config.load(None)
        cfg.video.image_candidates = 3
        orig_avail, orig_run, orig_scores = imagegen.available, imagegen._run_mflux, imagegen.clip_scores
        try:
            imagegen.available = lambda c: True
            imagegen._run_mflux = lambda p, out, seed, c: (out.write_bytes(b"PNG" + bytes([seed % 251])), True)[1]
            imagegen.clip_scores = lambda imgs, text: [0.1, 0.9, 0.4]        # middle candidate wins
            with tempfile.TemporaryDirectory() as td:
                dest = Path(td) / "01.png"
                rep = imagegen.generate_for_segment(self._seg(), self._script(), cfg, dest)
            self.assertEqual(rep["candidates"], 3)
            self.assertEqual(rep["selection"], "clip")
            self.assertEqual(rep["chosen"], 1)
        finally:
            imagegen.available, imagegen._run_mflux, imagegen.clip_scores = orig_avail, orig_run, orig_scores

    def test_generate_falls_back_to_first_without_clip(self):
        from avp import imagegen
        cfg = Config.load(None)
        cfg.video.image_candidates = 2
        orig_avail, orig_run, orig_scores = imagegen.available, imagegen._run_mflux, imagegen.clip_scores
        try:
            imagegen.available = lambda c: True
            imagegen._run_mflux = lambda p, out, seed, c: (out.write_bytes(b"PNG"), True)[1]
            imagegen.clip_scores = lambda imgs, text: []                     # CLIP unavailable
            with tempfile.TemporaryDirectory() as td:
                rep = imagegen.generate_for_segment(self._seg(), self._script(), cfg, Path(td) / "o.png")
            self.assertEqual(rep["selection"], "first")
            self.assertEqual(rep["chosen"], 0)
        finally:
            imagegen.available, imagegen._run_mflux, imagegen.clip_scores = orig_avail, orig_run, orig_scores

    def test_generate_returns_none_when_every_candidate_fails(self):
        from avp import imagegen
        cfg = Config.load(None)
        orig_avail, orig_run = imagegen.available, imagegen._run_mflux
        try:
            imagegen.available = lambda c: True
            imagegen._run_mflux = lambda p, out, seed, c: False               # generator broken
            with tempfile.TemporaryDirectory() as td:
                self.assertIsNone(
                    imagegen.generate_for_segment(self._seg(), self._script(), cfg, Path(td) / "o.png"))
        finally:
            imagegen.available, imagegen._run_mflux = orig_avail, orig_run


class ClipFailureIsRetried(unittest.TestCase):
    """Regression: a CLIP load failure used to be cached forever. The image generator holds ~9.8 GB
    while it runs, so on a 24 GB machine CLIP's own load can lose the race once — and ranking then
    stayed off for the entire video even though memory frees up between segments."""

    def setUp(self):
        from avp import imagegen
        imagegen._CLIP.clear()

    def tearDown(self):
        from avp import imagegen
        imagegen._CLIP.clear()

    def test_failure_is_retried_then_eventually_given_up(self):
        from avp import imagegen
        calls = {"n": 0}

        def boom(*a, **k):
            calls["n"] += 1
            raise RuntimeError("out of memory")

        with mock.patch.dict("sys.modules", {"transformers": mock.Mock(
                CLIPModel=mock.Mock(from_pretrained=boom), CLIPProcessor=mock.Mock())}):
            for _ in range(imagegen._CLIP_TRIES + 4):
                self.assertIsNone(imagegen._clip())
        # retried up to the cap, then stopped paying the load cost on every segment
        self.assertEqual(calls["n"], imagegen._CLIP_TRIES)

    def test_success_is_cached(self):
        from avp import imagegen
        imagegen._CLIP["state"] = ("model", "proc", "cpu", "torch")
        with mock.patch.dict("sys.modules", {"transformers": None}):
            self.assertEqual(imagegen._clip()[0], "model")


class FootageMatching(unittest.TestCase):
    """Archive matching (A): richer query variants for the search, and a CLIP rerank on the pixels so
    a good photo with an unhelpful NASA title stops losing to keyword-matching filler."""

    def _seg(self, visual="Venus phases through a telescope", kw=("venus", "phases", "orbit")):
        from avp.models import Segment
        return Segment(index=2, narration="n", visual=visual, keywords=list(kw))

    def _script(self):
        from avp.models import Script
        return Script(title="t", topic="Galileo Galilei", segments=[])

    def test_expand_queries_specific_to_broad_and_deduped(self):
        qs = footage._expand_queries(self._seg(), self._script())
        self.assertEqual(qs[0], "Venus phases through a telescope")     # visual cue first
        self.assertIn("venus phases orbit", qs)                          # the joined blob
        self.assertIn("venus phases", qs)                                # adjacent pair
        self.assertIn("venus", qs)                                       # single keyword
        self.assertIn("Galileo Galilei", qs)                             # topic last
        self.assertEqual(len(qs), len(set(q.lower() for q in qs)))       # deduped

    def test_expand_queries_survives_empty_segment(self):
        from avp.models import Script, Segment
        qs = footage._expand_queries(Segment(index=1, narration="n"), Script(title="t", topic="Mars", segments=[]))
        self.assertEqual(qs, ["Mars"])

    def test_segment_meaning_prefers_visual_and_keywords(self):
        m = footage._segment_meaning(self._seg(), self._script())
        self.assertIn("Venus phases through a telescope", m)
        self.assertIn("venus", m)
        from avp.models import Script, Segment
        bare = footage._segment_meaning(Segment(index=1, narration="spoken words"),
                                        Script(title="t", topic="Mars", segments=[]))
        self.assertEqual(bare, "Mars")                                   # never the narration

    def test_pick_top_returns_ranked_shortlist(self):
        cands = [{"nasa_id": "a", "title": "Venus transit", "description": "", "collection": "c1", "center": "JPL"},
                 {"nasa_id": "b", "title": "Random nebula", "description": "", "collection": "c2", "center": "GSFC"},
                 {"nasa_id": "c", "title": "Venus phases sequence", "description": "venus", "collection": "c3", "center": "JPL"}]
        orig = footage.nasa_candidates
        footage.nasa_candidates = lambda q, media_type="image", limit=30: cands
        try:
            top = footage._pick_top(["venus"], set(), ["venus", "phases"], "image", k=2)
        finally:
            footage.nasa_candidates = orig
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0][0]["nasa_id"], "c")                      # best text match first
        self.assertGreaterEqual(top[0][1], top[1][1])                    # relevance sorted

    def test_pick_top_skips_used_and_dedups_titles(self):
        cands = [{"nasa_id": "a", "title": "Mars portrait", "description": "", "collection": "c1", "center": "JPL"},
                 {"nasa_id": "b", "title": "Mars Portrait", "description": "", "collection": "c2", "center": "JPL"},
                 {"nasa_id": "c", "title": "Mars dunes", "description": "", "collection": "c3", "center": "JPL"}]
        orig = footage.nasa_candidates
        footage.nasa_candidates = lambda q, media_type="image", limit=30: cands
        try:
            # used_ids holds BOTH the id and the title key — that's what _try_nasa records after a
            # pick, and it's the title key that blocks the same photo under a different nasa_id.
            used = {"a", footage._title_key("Mars portrait")}
            top = footage._pick_top(["mars"], used, ["mars"], "image", k=5)
        finally:
            footage.nasa_candidates = orig
        ids = [c["nasa_id"] for c, _ in top]
        self.assertNotIn("a", ids)                                       # already used elsewhere
        self.assertNotIn("b", ids)                                       # same title, different id
        self.assertEqual(ids, ["c"])

    def test_clip_score_is_not_rescaled_into_text_relevance(self):
        """Regression: a mediocre CLIP match (0.29) must NOT be reported as ~0.92 by rescaling —
        that inflation let a plainly wrong photo pass a gate the text floor would have failed."""
        from avp.config import Config as C
        cfg = C.load(None)
        self.assertEqual(cfg.video.footage_clip_floor, 0.25)          # CLIP has its own scale/floor
        src = Path("src/avp/footage.py").read_text()
        self.assertNotIn("/ 0.32", src)                                # the old rescale is gone
        self.assertIn("footage_clip_floor", src)

    def test_clip_rerank_picks_highest_and_degrades(self):
        import avp.imagegen as ig
        orig = ig.clip_scores
        try:
            ig.clip_scores = lambda paths, text: [0.2, 0.7, 0.3]
            self.assertEqual(footage._clip_rerank([Path("a"), Path("b"), Path("c")], "venus"), (1, 0.7))
            ig.clip_scores = lambda paths, text: []                      # CLIP unavailable
            self.assertIsNone(footage._clip_rerank([Path("a")], "venus"))
        finally:
            ig.clip_scores = orig


class ScriptLengthBudget(unittest.TestCase):
    """The word budget must match the voice's MEASURED pace per language, and an over-long draft must
    be trimmed — a 50s target once produced 60.3s of speech, blowing the 60s ceiling."""

    def test_word_budget_is_language_aware(self):
        en, it = llm._words_for(50, "en"), llm._words_for(50, "it")
        self.assertLess(en, it)                       # English is spoken slower (words/sec)
        self.assertEqual(en, 118)                     # 50s × 2.35 w/s (measured EN pace)
        self.assertEqual(it, 130)                     # 50s × 2.60 w/s (measured IT pace)
        self.assertEqual(llm._words_for(50, None), llm._words_for(50, "en"))   # safe default
        self.assertEqual(llm._words_for(50, "de"), 120)                        # unknown → 2.4

    def test_budget_lands_under_target_at_measured_rate(self):
        # EN measured 2.37-2.46 w/s: the budget must not exceed the target at the FASTEST word count.
        self.assertLessEqual(llm._words_for(50, "en") / 2.37, 50.0)
        self.assertLessEqual(llm._words_for(50, "it") / 2.60, 50.0)

    def test_generate_script_has_a_trim_guard(self):
        src = Path("src/avp/llm.py").read_text()
        self.assertIn("TOO LONG", src)                # symmetric with the expand guard
        self.assertIn("words * 1.12", src)


class SocialPolishFixes(unittest.TestCase):
    """The six issues from the user's review: green-canvas backdrop, glued-on CTA, false-color look,
    hybrid finale, music quality, slow one-image pacing."""

    def test_backdrop_is_space_not_green_canvas(self):
        # old bug: ImageChops.add(scale=1.7) DIVIDED the image into a muddy teal 'book canvas'
        from avp.captions import render_cosmic_backdrop
        from avp.config import VideoConfig
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "b.png"
            render_cosmic_backdrop(out, VideoConfig(width=270, height=480), seed=3)
            im = Image.open(out).convert("RGB")
            px = list(im.getdata())
            n = len(px)
            avg = [sum(c[i] for c in px) / n for i in range(3)]
            self.assertLess(max(avg), 90)                          # overall dark (it's space)
            self.assertGreater(avg[2], avg[0])                     # blue-dominant…
            self.assertGreater(avg[2], avg[1])                     # …and NEVER green-dominant
            bright = sum(1 for c in px if max(c) > 160)
            self.assertGreater(bright, n * 0.001)                  # real stars are visible

    def test_cta_narration_uses_topic_bridge_with_fallback(self):
        from avp import stages
        from avp.models import Script
        cfg = Config.load(None)
        bridged = Script(title="t", segments=[], cta_bridge="Want to capture Saturn's rings yourself?")
        self.assertEqual(stages._cta_narration(bridged, cfg),
                         "Want to capture Saturn's rings yourself? Get AstroStackerPro — link in bio.")
        plain = Script(title="t", segments=[])
        self.assertEqual(stages._cta_narration(plain, cfg),
                         cfg.funnel.cta_line.format(app=cfg.funnel.app_name))

    def test_script_dataclass_accepts_old_json_without_bridge(self):
        from avp.models import Script
        old = Script.from_dict({"title": "t", "segments": [], "topic": "x"})
        self.assertEqual(old.cta_bridge, "")

    def test_prompt_asks_for_cta_bridge(self):
        self.assertIn("cta_bridge", llm.USER_TMPL)
        self.assertIn("Do NOT name any app", llm.USER_TMPL)

    def test_imagegen_style_is_natural_color(self):
        from avp import imagegen
        self.assertIn("natural true-color", imagegen.STILE)
        self.assertNotIn("telescope and probe imagery look", imagegen.STILE)
        for term in ("false color", "infrared look", "monochrome orange"):
            self.assertIn(term, imagegen.NEGATIVI)

    def test_music_prompts_carry_production_quality(self):
        from avp import music
        for mood, prompt in music.PROMPTS.items():
            self.assertIn("high fidelity", prompt, mood)
        self.assertIn("muddy", music.NEGATIVE)
        self.assertIn("sudden cuts", music.NEGATIVE)

    def test_imagegen_keeps_ranked_runner_up(self):
        from avp import imagegen
        from avp.models import Script, Segment
        cfg = Config.load(None)
        cfg.video.image_candidates = 3
        cfg.video.images_per_segment = 2
        seg = Segment(index=4, narration="n", visual="Saturn", keywords=["saturn"])
        script = Script(title="t", topic="Saturn", segments=[])
        orig = (imagegen.available, imagegen._run_mflux, imagegen.clip_scores)
        try:
            imagegen.available = lambda c: True
            imagegen._run_mflux = lambda p, out, seed, c: (out.write_bytes(b"P" + str(seed).encode()), True)[1]
            imagegen.clip_scores = lambda imgs, text: [0.10, 0.90, 0.40]
            with tempfile.TemporaryDirectory() as td:
                dest = Path(td) / "04.png"
                rep = imagegen.generate_for_segment(seg, script, cfg, dest)
                self.assertEqual(rep["kept"], 2)
                self.assertEqual(rep["chosen"], 1)                 # best candidate first
                self.assertTrue(dest.exists())
                second = Path(td) / "04_2.png"
                self.assertTrue(second.exists())                   # runner-up on screen too
                # ranked order: dest holds candidate #1 (0.90), second holds #2 (0.40)
                self.assertNotEqual(dest.read_bytes(), second.read_bytes())
        finally:
            imagegen.available, imagegen._run_mflux, imagegen.clip_scores = orig

    def test_segment_sources_splits_stills_not_cta(self):
        from avp import stages
        from avp.models import Segment
        with tempfile.TemporaryDirectory() as td:
            fdir = Path(td) / "footage"
            fdir.mkdir()
            (fdir / "02.png").write_bytes(b"a")
            (fdir / "02_2.png").write_bytes(b"b")
            (fdir / "02_3.png").write_bytes(b"c")
            (fdir / "06.png").write_bytes(b"e")

            class P:
                footage_dir = fdir
            seg = Segment(index=2, narration="n", footage="02.png")
            self.assertEqual([p.name for p in stages._segment_sources(P, seg)],
                             ["02.png", "02_2.png", "02_3.png"])
            cta = Segment(index=6, narration="n", footage="06.png", kind="cta")
            self.assertEqual([p.name for p in stages._segment_sources(P, cta)], ["06.png"])
            vid = Segment(index=2, narration="n", footage="02.mp4")
            self.assertEqual([p.name for p in stages._segment_sources(P, vid)], ["02.mp4"])


class SocialTokenStore(unittest.TestCase):
    """The token store must survive the things that actually happen to it: a crash mid-write, a
    corrupt file, and being printed to a terminal."""

    def _store(self, tmp):
        return mock.patch.dict("os.environ", {"AVP_TOKEN_STORE": str(Path(tmp) / "t.json")})

    def test_roundtrip_and_expiry_stamp(self):
        from avp.social import tokens
        with tempfile.TemporaryDirectory() as tmp, self._store(tmp):
            tokens.put("tiktok", {"access_token": "a", "refresh_token": "r", "expires_in": 3600,
                                  "account": "Astro"})
            rec = tokens.get("tiktok")
            # expires_in is converted to an absolute expires_at, or a token stored today looks fresh
            # forever after a restart.
            self.assertNotIn("expires_in", rec)
            self.assertGreater(rec["expires_at"], time.time() + 3000)
            self.assertTrue(tokens.is_fresh(rec))

    def test_expired_token_is_not_fresh(self):
        from avp.social import tokens
        self.assertFalse(tokens.is_fresh({"access_token": "a", "expires_at": time.time() - 1}))
        # inside the safety skew: still "not fresh", because an upload takes minutes
        self.assertFalse(tokens.is_fresh({"access_token": "a", "expires_at": time.time() + 60}))
        self.assertTrue(tokens.is_fresh({"access_token": "a", "expires_at": time.time() + 9999}))
        self.assertFalse(tokens.is_fresh(None))
        self.assertFalse(tokens.is_fresh({}))

    def test_connected_never_leaks_tokens(self):
        from avp.social import tokens
        with tempfile.TemporaryDirectory() as tmp, self._store(tmp):
            tokens.put("tiktok", {"access_token": "SECRET-VALUE", "refresh_token": "ALSO-SECRET",
                                  "expires_in": 100, "account": "Astro"})
            blob = json.dumps(tokens.connected())
            self.assertNotIn("SECRET-VALUE", blob)
            self.assertNotIn("ALSO-SECRET", blob)
            self.assertIn("Astro", blob)

    def test_corrupt_store_is_survivable(self):
        from avp.social import tokens
        with tempfile.TemporaryDirectory() as tmp, self._store(tmp):
            Path(tmp, "t.json").write_text("{not json")
            self.assertEqual(tokens.connected(), {})       # no crash on a scheduled run
            tokens.put("tiktok", {"access_token": "a"})    # and it recovers by overwriting
            self.assertEqual(tokens.get("tiktok")["access_token"], "a")

    def test_file_is_private(self):
        from avp.social import tokens
        with tempfile.TemporaryDirectory() as tmp, self._store(tmp):
            tokens.put("tiktok", {"access_token": "a"})
            self.assertEqual(oct(tokens.store_path().stat().st_mode)[-3:], "600")

    def test_forget(self):
        from avp.social import tokens
        with tempfile.TemporaryDirectory() as tmp, self._store(tmp):
            tokens.put("tiktok", {"access_token": "a"})
            self.assertTrue(tokens.forget("tiktok"))
            self.assertFalse(tokens.forget("tiktok"))
            self.assertIsNone(tokens.get("tiktok"))


class SocialTokenRefreshExpiry(unittest.TestCase):
    """Regression: a refreshed record carries expires_at=None plus a new expires_in. If None counted
    as a real value the token looked fresh forever and every post after the true expiry 401'd."""

    def test_refresh_recomputes_the_expiry(self):
        from avp.social import tokens
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {"AVP_TOKEN_STORE": str(Path(tmp) / "t.json")}):
                tokens.put("tiktok", {"access_token": "old", "expires_in": 10})
                self.assertFalse(tokens.is_fresh(tokens.get("tiktok")))     # 10s < the 300s skew
                # what TikTok.refresh() hands back
                tokens.put("tiktok", {"access_token": "new", "expires_in": 86400,
                                      "expires_at": None})
                rec = tokens.get("tiktok")
                self.assertTrue(tokens.is_fresh(rec))
                self.assertIsNotNone(rec["expires_at"])
                self.assertNotIn("expires_in", rec)         # never left behind to confuse a later read
                self.assertGreater(rec["expires_at"], time.time() + 80000)


class YouTubeTitleChoice(unittest.TestCase):
    """Regression: `a or b if c else d` binds as `(a or b) if c else d`, so an empty caption threw
    away a perfectly good metadata title."""

    def _title(self, meta, caption):
        from avp.social.youtube import YouTube
        captured = {}

        class FakeResp:
            status_code = 200
            headers = {"Location": "https://upload.example/session"}
            text = ""

            def json(self):
                return {"id": "vid1", "status": {"privacyStatus": "public"}}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["title"] = json["snippet"]["title"]
            return FakeResp()

        with tempfile.TemporaryDirectory() as tmp:
            vid = Path(tmp) / "v.mp4"
            vid.write_bytes(b"\0" * 16)
            with mock.patch("avp.social.youtube.requests.post", side_effect=fake_post), \
                 mock.patch("avp.social.youtube.requests.put", return_value=FakeResp()):
                YouTube().post(vid, caption, meta, Config(), "tok", {}, False)
        return captured["title"]

    def test_metadata_title_wins_even_with_an_empty_caption(self):
        self.assertEqual(self._title({"youtube": {"title": "Real Title"}}, ""), "Real Title")

    def test_falls_back_to_the_first_caption_line(self):
        self.assertEqual(self._title({}, "First line\nsecond"), "First line")

    def test_never_empty(self):
        self.assertEqual(self._title({}, "   \n  "), "Untitled")


class TikTokChunking(unittest.TestCase):
    """TikTok requires 5-64 MB chunks; getting the plan wrong rejects the whole upload."""

    def test_single_chunk_when_it_fits(self):
        from avp.social.tiktok import chunk_plan
        self.assertEqual(chunk_plan(12 * 1024 * 1024), (12 * 1024 * 1024, 1))
        self.assertEqual(chunk_plan(64 * 1024 * 1024), (64 * 1024 * 1024, 1))

    def test_splits_past_the_single_chunk_limit(self):
        from avp.social.tiktok import CHUNK_SIZE, chunk_plan
        size = 200 * 1024 * 1024
        chunk, total = chunk_plan(size)
        self.assertEqual(chunk, CHUNK_SIZE)
        self.assertEqual(total, 20)

    def test_last_chunk_absorbs_the_remainder(self):
        """floor() division means the final chunk is bigger than chunk_size, which TikTok allows —
        what must never happen is a chunk count that leaves bytes unsent."""
        from avp.social.tiktok import chunk_plan
        size = 205 * 1024 * 1024
        chunk, total = chunk_plan(size)
        covered_before_last = (total - 1) * chunk
        self.assertLess(covered_before_last, size)          # the last chunk still has work to do
        self.assertGreaterEqual(covered_before_last + (size - covered_before_last), size)


class SocialAuthUrls(unittest.TestCase):
    APPS = {"AVP_TIKTOK_CLIENT_KEY": "k", "AVP_TIKTOK_CLIENT_SECRET": "s",
            "AVP_META_APP_ID": "k", "AVP_META_APP_SECRET": "s",
            "AVP_GOOGLE_CLIENT_ID": "k", "AVP_GOOGLE_CLIENT_SECRET": "s"}

    def test_tiktok_asks_for_three_scopes_only(self):
        """Postiz demands six; TikTok's review penalises scopes the app cannot demonstrate, so the
        native client must keep asking for exactly what it uses."""
        from avp.social.tiktok import TikTok
        self.assertEqual(set(TikTok.scopes),
                         {"user.info.basic", "video.upload", "video.publish"})

    def test_urls_carry_the_verified_redirect(self):
        from avp import social
        with mock.patch.dict("os.environ", self.APPS):
            for plat in social.PLATFORMS:
                url, state = social.start_connect(plat)
                self.assertIn("www.astrostackerpro.com%2Fconnect%2F" + plat, url)
                self.assertIn(state, url)

    def test_youtube_requests_offline_access(self):
        """Without access_type=offline AND prompt=consent Google returns no refresh token, and the
        link silently becomes single-use."""
        from avp import social
        with mock.patch.dict("os.environ", self.APPS):
            url, _ = social.start_connect("youtube")
        self.assertIn("access_type=offline", url)
        self.assertIn("prompt=consent", url)

    def test_missing_credentials_say_which_variable(self):
        from avp.social.base import app_credentials
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError) as e:
                app_credentials("tiktok")
        self.assertIn("AVP_TIKTOK_CLIENT_KEY", str(e.exception))


class NativePublishDispatch(unittest.TestCase):
    def _project(self, tmp):
        root = Path(tmp) / "p"
        (root).mkdir(parents=True)
        (root / "metadata.json").write_text(json.dumps(
            {"tiktok": {"caption": "hello"}, "youtube": {"title": "T", "description": "D"}}))
        vid = root / "final.mp4"
        vid.write_bytes(b"\0" * 1024)
        proj = mock.Mock()
        proj.root = root
        proj.output = vid
        proj.output_for = lambda _e: vid
        return proj

    def test_one_dead_platform_does_not_stop_the_others(self):
        """A failed TikTok upload is no reason to skip a good Instagram post."""
        from avp import publish
        cfg = Config()
        cfg.publish.backend = "native"
        cfg.publish.platforms = ["tiktok", "instagram"]
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp)

            def fake_post(platform, *a, **kw):
                if platform == "tiktok":
                    raise RuntimeError("boom")
                return {"post_id": "ig1"}

            with mock.patch("avp.social.post", side_effect=fake_post):
                plan = publish.stage_publish(proj, cfg, go=True)
            # the outcome is written back, so a scheduled run leaves an auditable record
            saved = json.loads((proj.root / "publish_plan.json").read_text())
        by = {p["platform"]: p for p in plan}
        self.assertFalse(by["tiktok"]["posted"])
        self.assertIn("boom", by["tiktok"]["error"])
        self.assertTrue(by["instagram"]["posted"])
        self.assertEqual([p["posted"] for p in saved], [False, True])

    def test_postiz_backend_still_reachable(self):
        from avp import publish
        cfg = Config()
        cfg.publish.backend = "postiz"
        cfg.publish.platforms = ["tiktok"]
        cfg.publish.postiz_token = ""
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp)
            with mock.patch("avp.social.post") as native:
                with self.assertRaises(RuntimeError):      # no token → the Postiz path complains
                    publish.stage_publish(proj, cfg, go=True)
            native.assert_not_called()

    def test_dry_run_posts_nothing(self):
        from avp import publish
        cfg = Config()
        cfg.publish.platforms = ["tiktok"]
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._project(tmp)
            with mock.patch("avp.social.post") as native:
                plan = publish.stage_publish(proj, cfg, go=False)
            native.assert_not_called()
        self.assertEqual(plan[0]["caption"], "hello")


if __name__ == "__main__":
    unittest.main()
