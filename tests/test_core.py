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


if __name__ == "__main__":
    unittest.main()
