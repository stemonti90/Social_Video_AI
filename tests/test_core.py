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

    def test_en_both(self):
        self.assertEqual([p.name for p in tts.get_providers(self._cfg("en", "both"))],
                         ["kokoro", "chatterbox"])

    def test_it_both_skips_chatterbox(self):
        self.assertEqual([p.name for p in tts.get_providers(self._cfg("it", "both"))], ["kokoro"])

    def test_primary_falls_back_for_it(self):
        c = self._cfg("it", "both")
        c.tts.primary = "chatterbox"
        self.assertEqual(tts.primary_engine(c), "kokoro")


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
            first = footage._pick(["mars"], used, ["mars"], "image")
            self.assertEqual(first["nasa_id"], "A1")            # best relevance for "mars"
            used.add(first["nasa_id"])
            used.add(footage._title_key(first["title"]))         # as _try_nasa does
            second = footage._pick(["mars"], used, ["mars"], "image")
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
            out = llm.OllamaClient(LLMConfig()).chat("sys", "usr")
        self.assertEqual(out, "{}")
        self.assertEqual(captured["json"]["options"]["num_ctx"], 8192)
        self.assertEqual(captured["timeout"], (10, 300))      # (connect, read) — no 10-min hang


if __name__ == "__main__":
    unittest.main()
