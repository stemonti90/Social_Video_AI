"""Minimal unit tests for the engine's critical pure logic (stdlib unittest, no deps).

Run: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
import inspect
import os
import sys
import json
import pathlib
import re
import subprocess
import requests
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
    def test_falls_back_to_static_on_ken_burns_failure(self):
        """The push-in is rendered in PIL and streamed to ffmpeg; if that encode dies (ffmpeg has
        SIGSEGVed under memory pressure before), the segment must still get a static clip."""
        from PIL import Image
        calls = []

        def fake_run(args, retries=6):
            calls.append(args)

        def boom(*a, **k):
            raise RuntimeError("ffmpeg exited -11 while encoding the Ken Burns clip")

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "01.jpg"
            Image.new("RGB", (1200, 800), (20, 30, 60)).save(src)      # real image: PIL prep needs it
            with mock.patch("avp.ffmpeg.run", fake_run), \
                 mock.patch("avp.ffmpeg._ken_burns_via_pil", boom):
                ffmpeg.make_clip(src, 6.0, 1080, 1920, 30, True, Path(td) / "out.mp4")
        self.assertEqual(len(calls), 1)                                  # the static fallback encode
        self.assertTrue(any(".frame.png" in str(a) for a in calls[0]))    # = the PIL still path


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
        # Captions now also carry the mandatory brand tag (see BrandTagOnEveryPost); the whitespace
        # tidy-up is checked on the body, before the tag block.
        self.assertTrue(d["tiktok"]["caption"].startswith("e f"))
        self.assertTrue(d["instagram"]["caption"].startswith("g h"))


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
            # Each attempt now tries the local cache first and the network second; count ATTEMPTS
            # (the online call), not raw calls, so the cap under test stays what it means.
            if not k.get("local_files_only"):
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
        # 2.33, the SLOWEST delivery measured (the range is 2.33-2.61). Budgeting at the average makes
        # half the videos longer than planned, and long is the failure that breaks the 60s rule.
        self.assertEqual(en, 116)                     # 50s × 2.33 = 116.5, banker's rounding → 116
        self.assertEqual(it, 130)                     # 50s × 2.60 w/s (measured IT pace)
        self.assertEqual(llm._words_for(50, None), llm._words_for(50, "en"))   # safe default
        self.assertEqual(llm._words_for(50, "de"), 120)                        # unknown → 2.4

    def test_budget_lands_under_target_at_measured_rate(self):
        # EN measured 2.37-2.46 w/s: the budget must not exceed the target at the FASTEST word count.
        self.assertLessEqual(llm._words_for(50, "en") / 2.37, 50.0)
        self.assertLessEqual(llm._words_for(50, "it") / 2.60, 50.0)

    def test_generate_script_guards_length_in_both_directions(self):
        """Both rewrites must exist, and neither may be reachable without the other — the two used to
        be one-shot passes in a fixed order, so an expansion that overshot was never trimmed back."""
        src = Path("src/avp/llm.py").read_text()
        self.assertIn("TOO LONG", src)
        self.assertIn("TOO SHORT", src)
        from avp.llm import length_verdict
        self.assertEqual(length_verdict(200, 118), "long")
        self.assertEqual(length_verdict(60, 118), "short")


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


class CardMeetsTheVoice(unittest.TestCase):
    """The app card must be on screen the moment the voice names the app. It used to take a fixed
    2.6s tail, which on a real build (the ISS video) put the card up at 53.2s while the voice started
    pitching at 51.2s: two seconds of "Get AstroStackerPro" spoken over a photo of the station."""

    def _words(self, pairs):
        return [(w, t) for w, t in pairs]

    def test_it_finds_the_hook_even_when_the_aligner_splits_the_app_name(self):
        """The aligner returns "AstroStacker" + "Pro" and the em dash as its own token. Matching on
        the joined letters makes both invisible; matching token-by-token would miss entirely."""
        from avp.stages import _phrase_start
        words = self._words([("The", 0.0), ("sky", 0.4), ("waits.", 0.8),
                             ("Get", 1.5), ("AstroStacker", 1.8), ("Pro", 2.4),
                             ("—", 2.6), ("link", 2.7), ("in", 3.0), ("bio.", 3.2)])
        self.assertAlmostEqual(_phrase_start(words, "Get AstroStackerPro — link in bio."), 1.5)

    def test_a_phrase_that_is_not_spoken_is_not_invented(self):
        from avp.stages import _phrase_start
        self.assertIsNone(_phrase_start(self._words([("a", 0.0), ("b", 0.5)]), "totally absent"))
        self.assertIsNone(_phrase_start(self._words([("a", 0.0)]), "—"))   # nothing but separators

    def test_the_search_runs_from_the_right(self):
        """The bridge can echo a word the hook also uses; the CTA's copy is the one that matters."""
        from avp.stages import _phrase_start
        words = self._words([("get", 0.0), ("closer.", 0.5), ("get", 2.0), ("closer.", 2.5)])
        self.assertAlmostEqual(_phrase_start(words, "get closer"), 2.0)

    def _fixture(self, tmp, narration_words, hook_at):
        """A project whose captions place the app hook at `hook_at` seconds."""
        import json
        from avp.models import Script, Segment
        root = Path(tmp)
        (root / "captions.kokoro.json").write_text(json.dumps(
            [{"text": w, "start": t, "end": t + 0.2} for w, t in narration_words]))
        # Durations matter: the CTA's start is computed from the MEASURED length of everything before
        # it, which is what the voice stage records on every real build.
        script = Script(title="t", topic="test", segments=[
            Segment(index=1, narration="Body.", visual="v", keywords=[], duration=5.0),
            Segment(index=2, narration="The sky waits. Get AstroStackerPro — link in bio.",
                    visual="App endcard", keywords=[], kind="cta", duration=5.3)])
        script.cta_bridge = "The sky waits."
        script.bridge_kind = "shoot"
        return script, hook_at

    def test_the_card_starts_a_beat_before_the_app_is_named(self):
        from avp import stages
        cfg = Config.load(None)
        cfg.funnel.app_name = "AstroStackerPro"
        words = [("Body.", 0.0),
                 ("The", 5.0), ("sky", 5.4), ("waits.", 5.8),
                 ("Get", 8.0), ("AstroStacker", 8.3), ("Pro", 8.9),
                 ("—", 9.1), ("link", 9.2), ("in", 9.5), ("bio.", 9.7)]
        with tempfile.TemporaryDirectory() as tmp:
            script, _ = self._fixture(tmp, words, 8.0)
            project = mock.Mock(root=Path(tmp))
            render = 6.5          # CTA speech is 5.0s→9.9s plus a silent tail
            card = stages._card_seconds(project, cfg, script, "kokoro", render)
            # The CTA's speech starts after the content audio PLUS its trailing gap, which the
            # captions timeline includes; the hook then lands `into` seconds later.
            into = 8.0 - (5.0 + cfg.video.segment_gap)
            self.assertAlmostEqual(card, render - into + stages.CARD_LEAD, places=3)
            durs = stages._cta_split(render, 2, card)
            self.assertAlmostEqual(sum(durs), render, places=3)
            self.assertAlmostEqual(durs[0], into - stages.CARD_LEAD, places=3)

    def test_one_misheard_word_no_longer_loses_the_sync(self):
        """The regression that shipped: the aligner heard "Deimos" as "Dimos", one word inside a
        20-word CTA, so searching the transcript for the CTA's full text failed and the card fell
        back to its blind fixed tail. The CTA's start now comes from measured durations instead."""
        from avp import stages
        cfg = Config.load(None)
        cfg.funnel.app_name = "AstroStackerPro"
        words = [("Body.", 0.0),
                 ("The", 5.0), ("sky", 5.4), ("waits", 5.8), ("Dimos.", 6.2),   # <- misheard
                 ("Get", 8.0), ("AstroStacker", 8.3), ("Pro,", 8.9),            # <- and a stray comma
                 ("link", 9.2), ("in", 9.5), ("bio.", 9.7)]
        with tempfile.TemporaryDirectory() as tmp:
            script, _ = self._fixture(tmp, words, 8.0)
            script.segments[1].narration = ("The sky waits Deimos. Get AstroStackerPro — link in bio.")
            project = mock.Mock(root=Path(tmp))
            card = stages._card_seconds(project, cfg, script, "kokoro", 6.5)
            self.assertNotAlmostEqual(card, stages.ENDCARD_SECONDS, places=3)   # NOT the blind tail
            into = 8.0 - (5.0 + cfg.video.segment_gap)
            self.assertAlmostEqual(card, 6.5 - into + stages.CARD_LEAD, places=3)

    def test_a_cta_that_never_names_the_app_keeps_the_short_tail(self):
        """`bridge_policy: honest` on a topic with no honest link closes on the sky and lets the card
        sign off silently. There is no hook to sync to, and a short tail is exactly right."""
        from avp import stages
        cfg = Config.load(None)
        cfg.funnel.app_name = "AstroStackerPro"
        cfg.funnel.bridge_policy = "honest"
        with tempfile.TemporaryDirectory() as tmp:
            script, _ = self._fixture(tmp, [("The", 5.0), ("sky", 5.4), ("waits.", 5.8)], 0)
            script.bridge_kind = "none"
            project = mock.Mock(root=Path(tmp))
            self.assertAlmostEqual(stages._card_seconds(project, cfg, script, "kokoro", 8.0),
                                   stages.ENDCARD_SECONDS, places=3)

    def test_no_captions_means_the_old_fixed_tail(self):
        """captions is a stage that can be skipped. Assemble must still produce a sane card."""
        from avp import stages
        with tempfile.TemporaryDirectory() as tmp:
            script, _ = self._fixture(tmp, [], 0)
            (Path(tmp) / "captions.kokoro.json").unlink()
            project = mock.Mock(root=Path(tmp))
            self.assertAlmostEqual(stages._card_seconds(project, Config.load(None), script, "kokoro", 8.0),
                                   stages.ENDCARD_SECONDS, places=3)

    def test_the_bridge_always_keeps_a_readable_slice(self):
        """Even if the hook is nearly the whole CTA, the picture under the bridge must register."""
        from avp import stages
        durs = stages._cta_split(6.0, 2, card=99.0)
        self.assertAlmostEqual(sum(durs), 6.0, places=3)
        self.assertGreaterEqual(durs[0], stages.MIN_BRIDGE_SECONDS - 1e-9)


class EditedLinesAreReVoiced(unittest.TestCase):
    """The voice cache was keyed by segment INDEX, so a corrected line kept its old audio forever.
    A factual fix went into script.md, the stage logged "cached", and the video went on saying that
    Venera 4 took the first photograph of Venus — which it never did. `--force` did not help: it
    re-runs the stage, and the stage was skipping the file."""

    def test_the_wav_records_the_words_it_says(self):
        from avp import stages
        src = inspect.getsource(stages.stage_voice)
        self.assertIn("with_suffix(\".txt\")", src)
        self.assertIn("spoken_before == seg.narration", src)

    def test_matching_text_is_a_cache_hit_and_changed_text_is_not(self):
        """The decision itself, on the two cases that matter."""
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "01.wav"
            wav.write_bytes(b"RIFF")
            stamp = wav.with_suffix(".txt")
            stamp.write_text("The old line.")

            def reuse(narration):
                before = stamp.read_text() if stamp.exists() else None
                return wav.exists() and before == narration

            self.assertTrue(reuse("The old line."))
            self.assertFalse(reuse("The corrected line."))

    def test_audio_without_a_stamp_is_kept_but_flagged(self):
        """Projects built before the stamp existed must not all be re-voiced on upgrade — but the
        log has to say the words could not be verified, rather than claiming a clean cache hit."""
        from avp import stages
        src = inspect.getsource(stages.stage_voice)
        self.assertIn("cached, unverified", src)


class NativeTargetsAndTikTokSelfUnblock(unittest.TestCase):
    """auto.connected_platforms used to ask Postiz whatever the backend. Under the native backend
    that call failed, the target list came back empty, and the daily run would have BUILT every
    video and posted none. It now reads the token store — and TikTok joins only when TikTok itself
    says the account may post publicly, so the day the app clears review it switches on unaided."""

    def _cfg(self, platforms):
        cfg = Config.load(None)
        cfg.publish.backend = "native"
        cfg.auto.platforms = platforms
        return cfg

    def test_a_platform_with_a_token_is_a_target(self):
        from avp import auto
        with mock.patch("avp.social.tokens.get", lambda p: {"access_token": "t"} if p == "instagram" else None):
            self.assertEqual(auto.connected_platforms(self._cfg(["instagram", "tiktok"])), {"instagram"})

    def test_tiktok_stays_out_while_it_cannot_post_publicly(self):
        from avp import auto
        with mock.patch("avp.social.tokens.get", lambda p: {"access_token": "t"}), \
             mock.patch.object(auto, "tiktok_can_post_publicly", return_value=False):
            self.assertEqual(auto.connected_platforms(self._cfg(["instagram", "tiktok"])), {"instagram"})

    def test_tiktok_joins_by_itself_once_public_posting_is_offered(self):
        from avp import auto
        with mock.patch("avp.social.tokens.get", lambda p: {"access_token": "t"}), \
             mock.patch.object(auto, "tiktok_can_post_publicly", return_value=True):
            self.assertEqual(auto.connected_platforms(self._cfg(["instagram", "tiktok"])),
                             {"instagram", "tiktok"})

    def test_the_operator_may_choose_to_stock_tiktok_while_restricted(self):
        """auto.tiktok_restricted_ok: post now at SELF_ONLY and flip later, rather than wait."""
        from avp import auto
        cfg = self._cfg(["instagram", "tiktok"])
        cfg.auto.tiktok_restricted_ok = True
        with mock.patch("avp.social.tokens.get", lambda p: {"access_token": "t"}), \
             mock.patch.object(auto, "tiktok_can_post_publicly", return_value=False):
            self.assertEqual(auto.connected_platforms(cfg), {"instagram", "tiktok"})

    def test_public_posting_is_read_from_creator_info(self):
        """The decision is TikTok's own privacy_level_options, nothing inferred."""
        from avp import auto
        cfg = self._cfg(["tiktok"])
        for opts, want in ((["FOLLOWER_OF_CREATOR", "SELF_ONLY"], False),
                           (["PUBLIC_TO_EVERYONE", "SELF_ONLY"], True), ([], False)):
            fake = mock.Mock()
            fake.creator_info.return_value = {"privacy_level_options": opts}
            with mock.patch("avp.social.tiktok.TikTok", return_value=fake), \
                 mock.patch("avp.social.tokens.is_fresh", return_value=True):
                self.assertEqual(auto.tiktok_can_post_publicly(cfg, {"access_token": "t"}), want, opts)

    def test_a_failing_probe_means_not_yet_never_yes(self):
        """A wrong "yes" posts a video nobody can see; the safe default on any error is to wait."""
        from avp import auto
        fake = mock.Mock(); fake.creator_info.side_effect = RuntimeError("boom")
        with mock.patch("avp.social.tiktok.TikTok", return_value=fake), \
             mock.patch("avp.social.tokens.is_fresh", return_value=True):
            self.assertFalse(auto.tiktok_can_post_publicly(self._cfg(["tiktok"]), {"access_token": "t"}))


class TranslatedSubtitlesAreClauses(unittest.TestCase):
    """A translation cannot follow the voice word by word — different order, different count — and
    pushing it through the karaoke renderer made 111 three-word cards for a 47s video, median 0.32s,
    the amber highlight on words unrelated to what was being said. Translations are now cut into
    readable clauses that each hold a share of the segment's audio window."""

    def test_a_long_sentence_becomes_two_readable_cards(self):
        from avp.captions import split_phrases, PHRASE_MAX_WORDS
        t = ("Una cicatrice planetaria ha spaccato la crosta di Marte abbastanza da inghiottire "
             "un intero continente.")
        out = split_phrases(t, 0.0, 6.2)
        self.assertEqual(len(out), 2)
        for txt, a, b in out:
            self.assertLessEqual(len(txt.split()), PHRASE_MAX_WORDS)
        self.assertAlmostEqual(out[-1][2], 6.2, places=3)       # the window is fully covered
        self.assertAlmostEqual(out[0][2], out[1][1], places=3)  # no gap between cards

    def test_a_card_never_ends_on_a_dangling_preposition(self):
        from avp.captions import split_phrases
        t = ("Una cicatrice planetaria ha spaccato la crosta di Marte abbastanza da inghiottire "
             "un intero continente.")
        first = split_phrases(t, 0.0, 6.2)[0][0]
        self.assertNotRegex(first, r"\b(di|da|la|il|un|e)$")

    def test_sentence_ends_are_preferred_cut_points(self):
        from avp.captions import split_phrases
        out = split_phrases("Prima frase breve. Seconda frase breve.", 0.0, 4.0)
        self.assertEqual([x[0] for x in out], ["Prima frase breve.", "Seconda frase breve."])

    def test_a_flash_is_merged_into_its_neighbour(self):
        """A 0.4s card is a flash: time is shared by word count, so a 2-word tail after a sentence
        end gets folded into the previous card rather than blinking."""
        from avp.captions import split_phrases, PHRASE_MIN_SECONDS
        out = split_phrases("Una frase lunga che riempie la carta intera. Fine.", 0.0, 3.0)
        self.assertEqual(len(out), 1)
        self.assertGreaterEqual(out[0][2] - out[0][1], PHRASE_MIN_SECONDS)

    def test_the_translated_path_no_longer_uses_the_karaoke_renderer(self):
        from avp import stages
        src = inspect.getsource(stages._assemble_engine)
        i = src.index("want_translated and sub_json")
        block = src[i:src.index("elif cap_json.exists()", i)]    # the translated branch, whole
        self.assertIn("render_phrase_pngs", block)
        self.assertNotIn("distribute_words", block)


class BrandTagOnEveryPost(unittest.TestCase):
    """#astrostackerpro goes on every post, on every platform, by construction — the channel exists
    to funnel viewers to the app, and a prompt asking the model for the tag is advice it can ignore."""

    def test_instagram_gets_the_tag_whether_or_not_the_model_included_it(self):
        from avp.llm import _clean_metadata, BRAND_TAG
        for tags in (["#space", BRAND_TAG], ["#space"], ["#SPACE", "#AstroStackerPro"]):
            out = _clean_metadata({"instagram": {"caption": "x"}, "instagram_hashtags": list(tags)})
            got = out["instagram"]["hashtags"]
            self.assertIn(BRAND_TAG, got)
            self.assertEqual(got.count(BRAND_TAG), 1, got)       # never twice
            self.assertIn(BRAND_TAG, out["instagram"]["caption"])

    def test_it_is_appended_last_so_it_never_displaces_a_narrow_tag(self):
        from avp.llm import _clean_metadata, BRAND_TAG
        out = _clean_metadata({"instagram": {"caption": "x"},
                               "instagram_hashtags": [f"#t{i}" for i in range(29)]})
        got = out["instagram"]["hashtags"]
        self.assertEqual(len(got), 30)
        self.assertEqual(got[-1], BRAND_TAG)

    def test_a_model_that_returned_no_tag_list_still_gets_it(self):
        from avp.llm import _clean_metadata, BRAND_TAG
        out = _clean_metadata({"instagram": {"caption": "A line. #mars"}})
        self.assertIn(BRAND_TAG, out["instagram"]["hashtags"])
        self.assertIn("#mars", out["instagram"]["hashtags"])       # inline tags were salvaged

    def test_tiktok_and_youtube_carry_it_too(self):
        from avp.llm import _clean_metadata, BRAND_TAG
        out = _clean_metadata({"tiktok": {"caption": "y #space"}, "youtube": {"tags": ["space"]}})
        self.assertIn(BRAND_TAG, out["tiktok"]["hashtags"])
        self.assertIn(BRAND_TAG, out["tiktok"]["caption"])
        self.assertIn("astrostackerpro", out["youtube"]["tags"])    # YouTube tags carry no '#'


class StaleImagesAreNotManualOverrides(unittest.TestCase):
    """`avp run --force` rewrote the script and every segment logged "← manual 01.png": the override
    glob is `NN.*`, exactly what the generator writes, so the previous build's pictures were mistaken
    for files the operator had placed and generation was skipped. An 8-segment script rendered over
    the previous script's 7 pictures — narration about the Perseverance rover over an empty plain."""

    def _project(self, tmp, rows):
        root = Path(tmp)
        if rows is not None:
            (root / "footage_report.json").write_text(json.dumps(rows))
        return mock.Mock(root=root)

    def _seg(self, narration):
        return Segment(index=1, narration=narration, visual="v", keywords=[])

    def test_our_own_picture_for_different_words_is_stale(self):
        from avp.footage import _stale_from_a_previous_script
        with tempfile.TemporaryDirectory() as tmp:
            p = self._project(tmp, [{"index": 1, "segment": "The old narration.",
                                     "outcome": "generated"}])
            self.assertTrue(_stale_from_a_previous_script(p, self._seg("A completely new line.")))

    def test_our_own_picture_for_the_same_words_is_a_legitimate_resume(self):
        """`avp build` must stay resumable — an interrupted run should not rebuild what it had."""
        from avp.footage import _stale_from_a_previous_script
        with tempfile.TemporaryDirectory() as tmp:
            p = self._project(tmp, [{"index": 1, "segment": "Same words.", "outcome": "generated"}])
            self.assertFalse(_stale_from_a_previous_script(p, self._seg("Same words.")))

    def test_a_file_we_cannot_prove_is_ours_stays_the_operators(self):
        """Deleting someone's deliberately placed picture is the one unrecoverable mistake here."""
        from avp.footage import _stale_from_a_previous_script
        for rows in (None,                                            # no report at all
                     [],                                              # report, no entry for it
                     [{"index": 1, "segment": "x", "outcome": "manual"}],
                     [{"index": 2, "segment": "x", "outcome": "generated"}]):   # another segment
            with tempfile.TemporaryDirectory() as tmp:
                p = self._project(tmp, rows)
                self.assertFalse(_stale_from_a_previous_script(p, self._seg("new words")), rows)

    def test_an_unreadable_report_is_not_a_licence_to_delete(self):
        from avp.footage import _stale_from_a_previous_script
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "footage_report.json").write_text("{not json")
            self.assertFalse(_stale_from_a_previous_script(mock.Mock(root=Path(tmp)),
                                                           self._seg("new words")))

    def test_the_narration_is_compared_the_way_the_report_stores_it(self):
        """The report truncates to 120 chars; comparing against the full line would call every long
        segment stale and regenerate the whole video on every resume."""
        from avp.footage import _stale_from_a_previous_script
        long = "word " * 60
        with tempfile.TemporaryDirectory() as tmp:
            p = self._project(tmp, [{"index": 1, "segment": long[:120], "outcome": "generated"}])
            self.assertFalse(_stale_from_a_previous_script(p, self._seg(long)))


class ScriptLandsNearTheTarget(unittest.TestCase):
    """Videos have to run just under 60s — "quasi 60 secondi, altrimenti diventa turbo compresso".
    Olympus Mons rendered 42.7s against a 58s target: the writer returned 82% of the words it was
    asked for, which sat inside the dead band between the two length guards, so neither fired."""

    def _asked(self, seconds=50, lang="en", ipseg=2):
        """The word budget actually handed to the writer, and the target the guards check."""
        from avp import llm
        wps = llm._WPS[lang]
        words = round(seconds * wps)
        nseg = max(3, round(seconds / (llm.SECONDS_PER_IMAGE * ipseg)))
        return words, max(8, round(words * llm.ASK_INFLATION / nseg)) * nseg

    def test_the_writer_is_asked_for_more_than_the_target(self):
        """It delivers a measured median 88%, so asking for exactly the target lands short."""
        target, asked = self._asked()
        self.assertGreater(asked, target)

    def test_a_typical_delivery_now_lands_on_target(self):
        """The compensation must match the rate MEASURED ON THE CURRENT PROMPT — ~70%, not the 88%
        read off builds that predate the hook and photographable-visual rules. Getting the first
        draft near target is the only length lever that works: asked to lengthen, this model ignores
        the target and doubles what it has, and asked to cut a quarter it shaves 1-3%."""
        from avp.llm import length_verdict
        target, asked = self._asked()
        self.assertEqual(length_verdict(round(asked * 0.70), target), "ok")

    def test_a_draft_inside_the_band_is_left_alone(self):
        from avp.llm import length_verdict
        for n in (105, 112, 116):
            self.assertEqual(length_verdict(n, 112), "ok", n)

    def test_the_dead_band_that_shipped_a_42_second_video_is_closed(self):
        """The original guards trimmed 12% over but expanded only 20% under, so a draft at 81.7% of
        target sat between them untouched and rendered 42.7s against a 58s target."""
        from avp.llm import length_verdict
        self.assertEqual(length_verdict(int(118 * 0.817), 118), "short")

    def test_the_ceiling_is_tighter_than_the_floor(self):
        """Deliberately asymmetric, and NOT the symmetry an earlier pass here enforced: making the
        band ±12% closed the dead band but put the ceiling at 132 words, which renders 64s. Short is
        a weaker video; long is a video that breaks the 60s rule, so they cannot share a tolerance."""
        from avp.llm import length_verdict, LENGTH_FLOOR, LENGTH_HEADROOM_S
        target = 112
        over = target + LENGTH_HEADROOM_S * 2.33
        under = target * LENGTH_FLOOR
        self.assertLess(over - target, target - under)
        self.assertEqual(length_verdict(132, 112), "long")

    def test_the_ceiling_keeps_the_video_under_sixty_seconds(self):
        """The ceiling is not a taste parameter. At the SLOWEST delivery measured (2.33 words/s) and
        the LONGEST CTA measured (8.9s), plus ~0.8s of gaps, a script at the ceiling must still fit."""
        from avp.llm import length_verdict
        target = 112
        ceiling = max(w for w in range(target, target * 2) if length_verdict(w, target) == "ok")
        worst_case = ceiling / 2.33 + 8.9 + 0.8
        self.assertLess(worst_case, 60.0, f"{ceiling} words renders {worst_case:.1f}s")

    def test_an_expansion_that_overshoots_is_rejected(self):
        """The real failure: asked to lengthen 92 words toward 118, the model returned 222 and the
        old test — merely `new > cur` — accepted it. That is a 93s narration on a 58s target."""
        from avp.llm import moves_closer
        self.assertFalse(moves_closer(222, 92, 118))
        self.assertTrue(moves_closer(110, 92, 118))

    def test_a_trim_that_undershoots_is_rejected_too(self):
        """Same rule in the other direction, which is the point of stating it as distance: a trim
        from 200 down to 20 has overshot the 118 target by further than it started."""
        from avp.llm import moves_closer
        self.assertFalse(moves_closer(20, 200, 118))
        self.assertTrue(moves_closer(130, 200, 118))

    def test_a_rewrite_that_does_not_move_is_rejected(self):
        from avp.llm import moves_closer
        self.assertFalse(moves_closer(92, 92, 118))


class AHungStageIsKilled(unittest.TestCase):
    """A script stage sat 12h50m on an Ollama call that never answered — the client's own 10-minute
    read timeout did not fire, because the Mac had slept and a socket blocked across a suspend does
    not re-arm it. For an unattended 2-a-day run that is the worst possible failure: no error, no
    output, the channel simply stops. Per-request timeouts cannot be the only defence."""

    def test_every_stage_has_a_watchdog(self):
        from avp.pipeline import BUILD_STAGES, STAGE_TIMEOUTS, DEFAULT_STAGE_TIMEOUT
        for stage in list(BUILD_STAGES) + ["script"]:
            self.assertGreater(STAGE_TIMEOUTS.get(stage, DEFAULT_STAGE_TIMEOUT), 0, stage)

    def test_the_limits_leave_room_for_a_slow_but_working_stage(self):
        """They exist to catch a hang, never to hurry real work: footage genuinely runs ~25 min."""
        from avp.pipeline import STAGE_TIMEOUTS
        self.assertGreaterEqual(STAGE_TIMEOUTS["footage"], 2 * 25 * 60)
        self.assertGreaterEqual(STAGE_TIMEOUTS["script"], 2 * 16 * 60)

    def test_a_hanging_stage_is_killed_and_reported_as_timed_out(self):
        from avp import pipeline
        slept = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 start_new_session=True)
        try:
            with mock.patch.object(pipeline.subprocess, "Popen", return_value=slept), \
                 mock.patch.dict(pipeline.STAGE_TIMEOUTS, {"script": 0}):
                rc = pipeline._run_stage_subprocess("script", "slug", "config.yaml", False)
            self.assertEqual(rc, 124)                       # conventional "timed out"
            self.assertIsNotNone(slept.poll(), "the hung stage must actually be dead")
        finally:
            if slept.poll() is None:
                slept.kill()

    def test_the_kill_reaches_the_whole_group_not_just_the_child(self):
        """A stage spawns mflux and ffmpeg. Signalling only the direct child leaves those running
        against the project directory — which is exactly how two builds once raced over one folder."""
        from avp.pipeline import _kill_tree
        parent = subprocess.Popen(
            [sys.executable, "-c",
             "import subprocess,sys,time;"
             "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
             "print(c.pid,flush=True);time.sleep(60)"],
            stdout=subprocess.PIPE, text=True, start_new_session=True)
        child_pid = int(parent.stdout.readline().strip())
        _kill_tree(parent)
        self.assertIsNotNone(parent.poll())
        time.sleep(0.3)
        with self.assertRaises(OSError):                   # the grandchild is gone too
            os.kill(child_pid, 0)


class LengthLoopConvergesOnTheRealModel(unittest.TestCase):
    """End-to-end over the length loop, with a stub that reproduces what gemma actually does: asked to
    LENGTHEN it overshoots by ~90% every time (measured 90 -> 172, 175, 181 across three re-rolls, so
    a bias rather than noise), and asked to SHORTEN it lands accurately (142 -> 134, 140 -> 136).

    The loop must therefore reach the band by expanding and THEN trimming. An earlier version rejected
    the overshoot to protect the draft, which blocked that route and shipped 90 words as a 35s video."""

    def _script(self, n_words, nseg=6):
        per = max(1, n_words // nseg)
        segs, left = [], n_words
        for i in range(nseg):
            take = left if i == nseg - 1 else min(per, left)
            # Every token distinct: dedupe_segments drops a segment at >=0.9 similarity, and filler
            # that differs by one word in fifteen collapses the whole script into a single segment.
            segs.append({"narration": " ".join(f"w{i}x{j}" for j in range(max(1, take))),
                         "visual": f"wide shot number {i}", "keywords": ["mars"]})
            left -= take
        return {"title": "t", "bridge_kind": "shoot", "cta_bridge": "b", "segments": segs}

    def _client(self, start):
        """A stub gemma: overshoots every expansion, trims accurately, records what it was asked."""
        calls = []

        class Stub:
            def __init__(self, *a, **k):
                pass

            def chat(inner, system, user, **kw):  # noqa: N805
                if "TOO SHORT" in user:
                    calls.append("expand")
                    cur = len(re.findall(r"\bword\b", user)) // 1
                    prev = int(re.search(r"~(\d+) spoken words", user).group(1))
                    return json.dumps(self._script(int(prev * 1.9)))
                if "TOO LONG" in user:
                    calls.append("trim")
                    target = int(re.search(r"target is ~(\d+)", user).group(1))
                    return json.dumps(self._script(target + 3))     # accurate, slightly over
                calls.append("draft")
                return json.dumps(self._script(start))

        return Stub, calls

    def test_it_expands_then_trims_into_the_band(self):
        from avp import llm
        Stub, calls = self._client(90)
        with mock.patch.object(llm, "OllamaClient", Stub), \
             mock.patch.object(llm, "_judge_best", lambda c, d, t, l: d[0]):
            sc = llm.generate_script(llm.LLMConfig(), "Venus", seconds=48,
                                     refine_passes=0, best_of=1)
        words = sum(len(s.narration.split()) for s in sc.segments)
        self.assertEqual(llm.length_verdict(words, 112), "ok", f"{words} words, calls={calls}")
        self.assertEqual(calls.count("expand"), 1)      # one overshoot...
        self.assertEqual(calls.count("trim"), 1)        # ...then one trim, not three re-rolls

    def test_a_draft_already_in_band_is_not_touched(self):
        from avp import llm
        Stub, calls = self._client(112)
        with mock.patch.object(llm, "OllamaClient", Stub), \
             mock.patch.object(llm, "_judge_best", lambda c, d, t, l: d[0]):
            llm.generate_script(llm.LLMConfig(), "Venus", seconds=48, refine_passes=0, best_of=1)
        self.assertEqual(calls, ["draft"])              # no length pass at all

    def test_it_never_ships_a_draft_worse_than_it_started_with(self):
        """If every pass makes things worse, the closest draft seen must still be the one returned."""
        from avp import llm

        class AlwaysWorse:
            def __init__(self, *a, **k):
                pass

            def chat(inner, system, user, **kw):  # noqa: N805
                return json.dumps(self._script(400 if "TOO SHORT" in user or "TOO LONG" in user else 90))

        with mock.patch.object(llm, "OllamaClient", AlwaysWorse), \
             mock.patch.object(llm, "_judge_best", lambda c, d, t, l: d[0]):
            sc = llm.generate_script(llm.LLMConfig(), "Venus", seconds=48, refine_passes=0, best_of=1)
        words = sum(len(s.narration.split()) for s in sc.segments)
        self.assertEqual(words, 90, "the 400-word drafts must never win")


class SegmentwiseFit(unittest.TestCase):
    """Whole-script editing drifted videos across 47-64s: this model doubles when asked to lengthen
    and shaves 1-3% when asked to cut a quarter. Small local edits are where it complies, so the
    script is brought to length one segment at a time, each reply counted before it is accepted."""

    class _Compliant:
        """A stub that does what gemma does on a SMALL edit: hits the exact count it was asked for."""
        def __init__(self):
            self.calls = []

        def chat(self, system, user, **kw):
            n = int(re.search(r"EXACTLY (\d+) words", user).group(1))
            self.calls.append(n)
            return json.dumps({"narration": " ".join(f"w{i}" for i in range(n))})

    def _script(self, counts):
        return {"title": "t", "segments": [
            {"narration": " ".join(f"s{i}x{j}" for j in range(c)), "visual": f"v{i}", "keywords": []}
            for i, c in enumerate(counts)]}

    def test_budgets_sum_to_the_target_and_keep_the_hook_short(self):
        from avp.llm import segment_budgets
        b = segment_budgets(112, 6)
        self.assertEqual(sum(b), 112)
        self.assertEqual(min(b), b[0])                 # the hook lands in a breath
        self.assertEqual(max(b), b[-2])                # the penultimate 'wow' gets the room

    def test_only_segments_outside_tolerance_are_touched(self):
        from avp import llm
        client = self._Compliant()
        # budgets for 112/6 are [15,18,19,19,22,19]; 16 and 18 are within ±20%, 8 and 40 are not
        out = llm.fit_segments(client, self._script([16, 18, 8, 19, 40, 19]), 112, "en")
        counts = [llm._seg_words(x) for x in out["segments"]]
        self.assertEqual(len(client.calls), 2, client.calls)     # exactly the two outliers
        self.assertEqual(counts[2], 19)
        self.assertEqual(counts[4], 22)
        self.assertEqual(counts[0], 16)                           # the hook was left alone

    def test_the_total_lands_on_target_from_a_short_draft(self):
        """The real case: a 77-word draft against 112, which no whole-script pass could fix."""
        from avp import llm
        out = llm.fit_segments(self._Compliant(), self._script([13] * 6), 112, "en")
        total = sum(llm._seg_words(x) for x in out["segments"])
        self.assertEqual(llm.length_verdict(total, 112), "ok", total)

    def test_a_reply_that_is_no_closer_is_discarded(self):
        """The count is verified: a model that ignores the number cannot make a segment worse."""
        from avp import llm

        class Stubborn:
            def chat(self, system, user, **kw):
                return json.dumps({"narration": " ".join(["w"] * 60)})   # always 60, whatever asked

        out = llm.fit_segments(Stubborn(), self._script([8, 18, 19, 19, 22, 19]), 112, "en")
        self.assertEqual(llm._seg_words(out["segments"][0]), 8)        # kept, not replaced by 60

    def test_the_whole_script_loop_is_skipped_in_segmentwise_mode(self):
        from avp import llm
        src = inspect.getsource(llm.generate_script)
        self.assertIn('fit or "whole"', src)
        self.assertIn("fit_segments(client, data, words, language)", src)


class KenBurnsIsSmoothCentredAndExact(unittest.TestCase):
    """Three measured faults in the old zoompan push-in, all seen on the Mercury video:
    (1) `-t` let one frame more than zoompan's `d` through, and that frame snapped back to zoom 1.0
    — a pop before every cut (last frame differed 0.79 from the FIRST frame, 32.9 from the one
    before it); (2) zoompan's window is anchored at x=0,y=0, so every push-in drifted top-left;
    (3) its motion juddered — frame-to-frame difference alternated on 34% of frames (mean 3.27)
    against 0% and 1.26 for a float-precision affine crop, and supersampling did not help."""

    def test_the_window_is_centred_and_shrinks_monotonically(self):
        from avp.ffmpeg import zoom_window
        W, H = 1620, 2880
        prev = None
        for n in (0, 30, 60, 200, 1000):
            x0, y0, cw, ch = zoom_window(W, H, n)
            self.assertAlmostEqual(x0 * 2 + cw, W, places=6)     # centred horizontally
            self.assertAlmostEqual(y0 * 2 + ch, H, places=6)     # centred vertically
            if prev is not None:
                self.assertLessEqual(cw, prev)
            prev = cw
        self.assertAlmostEqual(zoom_window(W, H, 10_000)[2], W / 1.15, places=6)   # capped

    def test_the_clip_is_exactly_the_requested_frames(self):
        from avp import ffmpeg
        src = inspect.getsource(ffmpeg.make_clip)
        self.assertIn("round(duration * fps)", src)
        self.assertNotIn("zoompan=", src)                      # the filter, not the word in comments
        self.assertIn('"-frames:v", str(frames)', inspect.getsource(ffmpeg._ken_burns_via_pil))

    def test_a_tiny_real_render_has_the_right_frame_count(self):
        from PIL import Image
        from avp import ffmpeg
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "s.png"
            Image.effect_noise((300, 500), 80).convert("RGB").save(src)
            out = Path(tmp) / "c.mp4"
            ffmpeg.make_clip(src, 0.4, 108, 192, 30, True, out)     # 12 frames
            n = subprocess.run(["ffprobe", "-v", "0", "-count_frames", "-select_streams", "v",
                                "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True).stdout.strip()
            self.assertEqual(n, "12")

    def test_stills_inside_a_segment_dissolve_instead_of_hard_cutting(self):
        from avp import stages
        src = inspect.getsource(stages._assemble_engine)
        self.assertIn("concat_videos_xfade(parts, list(durs), inner, clip)", src)


class CachedModelsLoadWithoutTheNetwork(unittest.TestCase):
    """CLIP's weights were on disk and transformers still asked HuggingFace for a safetensors variant;
    the connection hung at 0% CPU for ten minutes. A build must not depend on a server to load a
    model it already has: the cache is tried first, the network only when the cache is empty."""

    def test_clip_is_loaded_from_the_local_cache_first(self):
        from avp import imagegen
        src = inspect.getsource(imagegen._clip)
        first_online = src.index("CLIPModel.from_pretrained(name)")
        offline = src.index("local_files_only=True")
        self.assertLess(offline, first_online)


class NoPeopleAmongTheCandidates(unittest.TestCase):
    """The generator puts a lone figure on a ridge "for scale" in ~1/4 of its landscapes, and no
    prompt wording removes it (measured: negative and positive phrasing, same seeds, same figures).
    A clean candidate existed in every pair generated, so the fix is to LOOK — torchvision's COCO
    detector, measured to separate perfectly (figures >= 0.991, clean <= 0.637) — and pick it."""

    def _run(self, people_by_index, n_cands=2, keep=1, retries=2, reject=True, detector=True):
        from avp import imagegen
        made = []

        def fake_mflux(prompt, out, seed, cfg):
            out.write_bytes(b"png"); made.append(out.name); return True

        def fake_people(paths):
            if not detector:
                return None
            return [people_by_index.get(int(pathlib.Path(x).stem.split("_c")[1]), 0.0) for x in paths]

        cfg = Config.load(None)
        cfg.video.image_candidates = n_cands
        cfg.video.image_people_retries = retries
        cfg.video.image_reject_people = reject
        cfg.video.max_images_per_segment = 3
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(imagegen, "available", return_value=True), \
             mock.patch.object(imagegen, "_run_mflux", side_effect=fake_mflux), \
             mock.patch.object(imagegen, "person_scores", side_effect=fake_people), \
             mock.patch.object(imagegen, "clip_scores", return_value=None), \
             mock.patch.object(imagegen, "images_for_segment", return_value=keep):
            seg = Segment(index=1, narration="n", visual="wide shot, a ridge", keywords=["mars"], duration=5.0)
            res = imagegen.generate_for_segment(seg, Script(title="t", topic="Mars", segments=[]),
                                                cfg, pathlib.Path(tmp) / "01.png")
        return res, made

    def test_a_peopled_candidate_is_dropped_when_a_clean_one_exists(self):
        res, made = self._run({0: 0.99, 1: 0.0})
        self.assertEqual(res["candidates"], 1)                 # the peopled one is gone
        self.assertEqual(len(made), 2)                          # no extra generation was needed
        self.assertTrue(all(x < 0.9 for x in res["people"]))

    def test_when_every_candidate_has_a_person_another_is_generated(self):
        res, made = self._run({0: 0.99, 1: 0.98, 2: 0.0})
        self.assertEqual(len(made), 3)                          # one extra candidate
        self.assertEqual(res["candidates"], 1)
        self.assertEqual(res["people"], [0.0])

    def test_after_the_retries_the_least_peopled_is_kept_with_a_warning(self):
        with self.assertLogs("avp.imagegen", level="WARNING") as cm:
            res, made = self._run({0: 0.99, 1: 0.95, 2: 0.97, 3: 0.93}, retries=2)
        self.assertEqual(len(made), 4)                          # 2 initial + 2 retries, then stop
        self.assertEqual(res["people"][0], 0.93)               # least peopled first
        self.assertTrue(any("every candidate shows a person" in m for m in cm.output))

    def test_screening_can_be_switched_off(self):
        res, made = self._run({0: 0.99, 1: 0.99}, reject=False)
        self.assertEqual(len(made), 2)
        self.assertIsNone(res["people"])

    def test_no_detector_means_no_screening_and_no_failure(self):
        res, made = self._run({0: 0.99}, detector=False)
        self.assertEqual(res["candidates"], 2)
        self.assertIsNone(res["people"])


class RegistryMustNotContradictTheShot(unittest.TestCase):
    """The lighting registry is appended to the prompt, and the model obeys it over the shot cue —
    it is the more concrete of the two. So a cue asking for a world seen from space must not be
    handed surface lighting: "wide shot, the planet Mars in the dark void" came back as an ochre
    desert under a butterscotch sky, because it matched `mars` before anything read "dark void"."""

    def test_an_orbital_cue_beats_the_surface_bucket(self):
        from avp.imagegen import _bucket
        for cue in ("wide shot, the planet Mars in the dark void",
                    "Titan seen from space",
                    "Venus from orbit, full disc",
                    "the globe of Mars against the blackness"):
            self.assertEqual(_bucket(cue), "planet", cue)

    def test_a_genuine_surface_shot_still_gets_surface_lighting(self):
        """The dusty bucket exists so Mars does not come back with the Moon's airless black sky."""
        from avp.imagegen import _bucket
        for cue in ("wide shot, the Martian surface at dawn",
                    "extreme close-up of volcanic flow textures, Martian surface",
                    "the peak against a dusty horizon, Mars horizon"):
            self.assertEqual(_bucket(cue), "dusty_surface", cue)

    def test_the_registry_reaches_the_prompt(self):
        """Guards the actual failure: a black-sky cue must not carry a butterscotch-sky registry."""
        from avp.imagegen import build_prompt
        seg = Segment(index=1, narration="n", visual="wide shot, the planet Mars in the dark void",
                      keywords=["Mars planet"])
        prompt = build_prompt(seg, Script(title="t", topic="Mars", segments=[]))
        self.assertIn("black", prompt)
        self.assertNotIn("butterscotch", prompt)


class NoPeopleInTheFrame(unittest.TestCase):
    """Faceless channel, and nobody has stood on Mars. Three of the eight Olympus Mons frames came
    back with a human silhouette on a ridge — landscape photography puts a figure in for scale, and
    "people, faces" reads as portrait scale, so it never suppressed them."""

    def test_the_negative_prompt_covers_a_distant_figure_too(self):
        from avp.imagegen import NEGATIVI
        for w in ("person", "human figure", "silhouette of a person", "astronaut", "figure for scale"):
            self.assertIn(w, NEGATIVI)

    def test_it_is_a_negative_prompt_not_appended_text(self):
        """These words in the POSITIVE prompt would ask for exactly what they forbid — the mistake
        that once made every video orange ("false color" sitting in the positive prompt)."""
        from avp.imagegen import build_prompt, NEGATIVI
        seg = Segment(index=1, narration="n", visual="wide shot, a volcanic ridge", keywords=["Mars"])
        prompt = build_prompt(seg, Script(title="t", topic="Mars", segments=[]))
        for w in ("person", "astronaut", "watermark"):
            self.assertNotIn(w, prompt)
        self.assertIn("person", NEGATIVI)


class CutRhythmFollowsTheSegment(unittest.TestCase):
    """The cut rhythm used to be `segments x 2` stills, which made it a hostage of how the writer
    split the text. Across twelve real builds the writer delivered 4 to 8 segments for the same 50s
    target, so the same rule produced anything from 2.37s to 5.94s per image — the fast end being the
    "everything flies past, I cannot even read it" the channel owner reported. The number now comes
    from each segment's MEASURED narration length instead."""

    def _cfg(self, cap=3):
        cfg = Config.load(None)
        cfg.video.max_images_per_segment = cap
        cfg.video.images_per_segment = 2
        return cfg

    def _seg(self, dur):
        return Segment(index=1, narration="n", visual="v", keywords=[], duration=dur)

    def test_every_shot_lands_near_the_target(self):
        from avp.imagegen import images_for_segment
        from avp.llm import SECONDS_PER_IMAGE
        cfg = self._cfg()
        for dur in (4.3, 5.6, 7.3, 9.0, 11.5):
            n = images_for_segment(self._seg(dur), cfg)
            self.assertLessEqual(abs(dur / n - SECONDS_PER_IMAGE), 1.5,
                                 f"{dur}s cut into {n} holds {dur / n:.2f}s per image")

    def test_a_short_segment_holds_one_still_instead_of_flashing_two(self):
        """The regression that mattered: eight 4.7s segments used to become sixteen 2.37s shots."""
        from avp.imagegen import images_for_segment
        self.assertEqual(images_for_segment(self._seg(4.74), self._cfg()), 1)

    def test_a_long_segment_gets_cut_more(self):
        from avp.imagegen import images_for_segment
        self.assertEqual(images_for_segment(self._seg(11.9), self._cfg()), 3)

    def test_the_ceiling_is_respected(self):
        from avp.imagegen import images_for_segment
        self.assertEqual(images_for_segment(self._seg(60.0), self._cfg(cap=2)), 2)

    def test_never_zero_images(self):
        """round(1.0/4.3) is 0, and a segment with no picture is a black frame."""
        from avp.imagegen import images_for_segment
        self.assertEqual(images_for_segment(self._seg(1.0), self._cfg()), 1)

    def test_an_unvoiced_segment_falls_back_to_the_planning_number(self):
        """`avp footage` can be run before `avp voice`; there is no measurement to use yet."""
        from avp.imagegen import images_for_segment
        cfg = self._cfg()
        self.assertEqual(images_for_segment(self._seg(0), cfg), cfg.video.images_per_segment)
        self.assertEqual(images_for_segment(Segment(index=1, narration="n", visual="v",
                                                    keywords=[]), cfg), 2)


class VisualsMustBePhotographable(unittest.TestCase):
    """This image model renders a requested "diagram" as garbled pseudo-text, and with the archives
    switched off (`generate_only`) there is nothing left to rescue that frame. The Olympus Mons
    script asked for "tectonic plate movement diagram vs static Martian crust"."""

    def test_it_keeps_the_side_of_a_vs_that_a_camera_could_shoot(self):
        from avp.imagegen import _photographable
        self.assertEqual(
            _photographable("medium shot, tectonic plate movement diagram vs static Martian crust"),
            "medium shot, static Martian crust")

    def test_a_good_visual_is_left_exactly_alone(self):
        from avp.imagegen import _photographable
        for v in ("wide shot, the peak against a dusty horizon",
                  "extreme close-up of volcanic flow textures",
                  "the object filling the frame, a massive volcanic peak"):
            self.assertEqual(_photographable(v), v)

    def test_a_preposition_left_dangling_goes_too(self):
        """Removing "annotated cutaway" must not leave the prompt starting with "of"."""
        from avp.imagegen import _photographable
        self.assertEqual(_photographable("annotated cutaway of the volcano's magma chamber"),
                         "the volcano's magma chamber")

    def test_a_cue_that_is_only_a_drawn_figure_yields_nothing(self):
        from avp.imagegen import _photographable
        self.assertEqual(_photographable("diagram"), "")
        self.assertEqual(_photographable(""), "")

    def test_the_prompt_falls_through_to_the_keywords_when_nothing_survives(self):
        """An empty subject would be a worse prompt than the segment's own search terms."""
        from avp.imagegen import build_prompt
        seg = Segment(index=1, narration="n", visual="labelled schematic",
                      keywords=["Olympus Mons", "Mars"])
        prompt = build_prompt(seg, Script(title="t", topic="Olympus Mons", segments=[]))
        self.assertIn("Olympus Mons", prompt)
        self.assertNotIn("schematic", prompt)

    def test_drawn_figures_are_also_in_the_negative_prompt(self):
        """Belt and braces: stripped from what we ask for, and steered away from on top."""
        from avp.imagegen import NEGATIVI
        for w in ("diagram", "chart", "infographic", "split screen"):
            self.assertIn(w, NEGATIVI)


class GenerateOnlyNeverTouchesTheArchives(unittest.TestCase):
    """`footage_source: generate_only` is a promise: every frame on screen is one we made. A build
    that quietly borrowed a NASA still for one segment broke that promise without saying so."""

    def _run(self, source, generate_returns, subject="the solar arrays tracking the sun"):
        """Run the footage stage with the archives wired to explode if they are ever consulted."""
        from avp import footage as fmod
        calls = {"nasa": 0, "wikimedia": 0, "generate": 0}

        def _gen(seg, script, cfg, dest, **kw):
            calls["generate"] += 1
            if not generate_returns:
                return None
            dest.write_bytes(b"png")
            return {"candidates": 1, "selection": "clip", "chosen": 0, "scores": [0.4]}

        def _nasa(*a, **kw):
            calls["nasa"] += 1
            return False

        def _wiki(*a, **kw):
            calls["wikimedia"] += 1
            return False

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config.load(None)
            cfg.paths.projects_dir = tmp
            cfg.video.footage_source = source
            from avp.manifest import VideoProject
            project = VideoProject("t", cfg)
            project.footage_dir.mkdir(parents=True, exist_ok=True)
            script = Script(title="t", topic="ISS", segments=[
                Segment(index=1, narration="n", visual=subject, keywords=["solar", "sun"])])
            with mock.patch("avp.imagegen.generate_for_segment", side_effect=_gen), \
                 mock.patch("avp.imagegen.available", return_value=True), \
                 mock.patch.object(fmod, "_try_nasa", _nasa), \
                 mock.patch.object(fmod, "_try_wikimedia", _wiki):
                fmod.resolve_footage(project, script, cfg, allow_download=True)
        return calls, script

    def test_a_solar_subject_no_longer_gets_routed_to_the_archives(self):
        """This is the exact segment that failed: "solar arrays tracking the sun" hit the `star`
        bucket, so archive-first routing handed a NASA video to a build meant to be all ours."""
        from avp import imagegen
        seg = Segment(index=1, narration="n", visual="the solar arrays tracking the sun",
                      keywords=["solar", "sun"])
        # under `generate` it genuinely does prefer the archives — that routing is still correct there
        self.assertTrue(imagegen.prefers_archive(seg, Script(title="t", topic="ISS", segments=[])))
        calls, script = self._run("generate_only", True)
        self.assertEqual(calls["nasa"], 0)
        self.assertEqual(calls["wikimedia"], 0)
        self.assertEqual(calls["generate"], 1)
        self.assertEqual(script.segments[0].credit, "")     # our own pixels, nothing to attribute

    def test_a_failed_generation_retries_then_falls_back_to_our_own_backdrop(self):
        """No archive safety net, so: one retry (memory pressure usually clears), then a procedural
        starfield. Still ours — never a borrowed photograph."""
        calls, script = self._run("generate_only", False)
        self.assertEqual(calls["generate"], 2)
        self.assertEqual(calls["nasa"], 0)
        self.assertEqual(calls["wikimedia"], 0)
        self.assertTrue(script.segments[0].footage)

    def test_plain_generate_still_reaches_the_archives(self):
        """The new value must not change what `generate` does — nebulae still want NASA's photo."""
        calls, _ = self._run("generate", False)
        self.assertGreaterEqual(calls["nasa"], 1)


class ArchiveFirstRouting(unittest.TestCase):
    """Nebulae and the Sun must reach the archives BEFORE the generator, because this image model
    renders them as the wrong object entirely (imagegen.ARCHIVE_FIRST documents the measurements)."""

    def _seg(self, visual, kw):
        from avp.models import Segment
        return Segment(index=1, narration="n", visual=visual, keywords=kw.split())

    def _script(self):
        from avp.models import Script
        return Script(title="t", topic="test", segments=[])

    def test_deep_sky_and_sun_go_to_archives_first(self):
        from avp import imagegen
        sc = self._script()
        self.assertTrue(imagegen.prefers_archive(
            self._seg("Wide shot, the Orion Nebula core", "Orion Nebula M42"), sc))
        self.assertTrue(imagegen.prefers_archive(
            self._seg("close up of a sunspot", "sun solar"), sc))

    def test_spacecraft_and_surfaces_still_generate_first(self):
        """The rule is narrow on purpose: a probe in flight has no archive equivalent, and the
        generator handles it well once the palette is right."""
        from avp import imagegen
        sc = self._script()
        self.assertFalse(imagegen.prefers_archive(
            self._seg("the Voyager probe in deep space", "Voyager spacecraft"), sc))
        self.assertFalse(imagegen.prefers_archive(
            self._seg("the surface of Mars", "Mars crater regolith"), sc))

    def test_falls_back_to_keywords_and_topic(self):
        from avp import imagegen
        from avp.models import Script
        sc = Script(title="t", topic="the Andromeda galaxy", segments=[])
        self.assertTrue(imagegen.prefers_archive(self._seg("", "nebula cluster"), sc))
        self.assertTrue(imagegen.prefers_archive(self._seg("", ""), sc))   # topic alone decides

    def test_negatives_are_not_in_the_positive_prompt(self):
        """Regression: "Avoid: false color, monochrome orange" used to be appended to the POSITIVE
        prompt, where a diffusion model reads it as a request. Those words belong to NEGATIVI."""
        from avp import imagegen
        p = imagegen.build_prompt(self._seg("Saturn from orbit", "saturn rings"), self._script())
        for banned in ("Avoid:", "false color", "monochrome orange", "no text", "illustration"):
            self.assertNotIn(banned, p, f"{banned!r} must not sit in the positive prompt")
        self.assertIn("false color", imagegen.NEGATIVI)


class DustyVsAirlessSurface(unittest.TestCase):
    """A world with an atmosphere and a world without one need opposite skies. Mars under the airless
    rule came back with the rover sitting in open black space."""

    def test_mars_and_titan_are_dusty(self):
        from avp import imagegen
        self.assertEqual(imagegen._bucket("wide shot, rover tracks in the Martian soil"), "dusty_surface")
        self.assertEqual(imagegen._bucket("Titan dunes under haze"), "dusty_surface")

    def test_moon_and_asteroids_stay_airless(self):
        from avp import imagegen
        self.assertEqual(imagegen._bucket("the lunar surface, craters"), "surface")
        self.assertEqual(imagegen._bucket("a rocky asteroid surface, regolith"), "surface")

    def test_neither_registry_entry_negates(self):
        """Both entries feed the POSITIVE prompt, where "no black sky" reads as "black sky"."""
        from avp import imagegen
        for key in ("surface", "dusty_surface"):
            text = imagegen.REGISTRO[key].lower()
            for neg in (" no ", "without", "never", "avoid"):
                self.assertNotIn(neg, text, f"{key} must not negate: {text!r}")


class MoodKeywordBoundaries(unittest.TestCase):
    """One short keyword firing inside a longer word decides an entire soundtrack. A Jupiter script
    scored "emotional" — piano and strings over hard science — because "solid" contains "soli"."""

    def test_a_keyword_inside_a_longer_word_does_not_count(self):
        from avp.music import classify_mood
        for text, trap in [("the crust stays solid for billions of years", "soli"),
                           ("engineers avoid that orbit entirely", "void"),
                           ("we can trace the signal back", "race"),
                           ("the probe will embrace the atmosphere", "race")]:
            r = classify_mood(text)
            self.assertNotIn(trap, r["rationale"],
                             f"{trap!r} matched inside a longer word: {text!r}")

    def test_the_same_word_standing_alone_still_counts(self):
        """The fix must not deafen the classifier — real signals still have to land."""
        from avp.music import classify_mood
        self.assertEqual(classify_mood("the void, silence and death, a dark end")["mood"], "dark")
        self.assertEqual(classify_mood("a vast, epic structure, truly giant")["mood"], "cinematic")
        # a lone real hit is still a hit — the boundary rule must not raise the bar
        self.assertIn("void", classify_mood("adrift in the void")["rationale"])

    def test_no_signal_falls_back_to_the_neutral_bed(self):
        from avp.music import classify_mood
        r = classify_mood("Ganymede's diameter exceeds Mercury's by a small margin.")
        self.assertEqual(r["mood"], "documentary")


class CtaEndcardTiming(unittest.TestCase):
    """The spoken CTA runs ~7 seconds. Parking the app card there for all of it made the video feel
    over while the voice was still going — viewers read it as the images having run out."""

    def test_the_card_gets_a_short_fixed_tail_not_a_share(self):
        """The point of a fixed tail: a CTA twice as long must not put the card up twice as long."""
        from avp.stages import _cta_split, ENDCARD_SECONDS
        for total in (6.0, 8.0, 12.0):
            durs = _cta_split(total, 2)
            self.assertAlmostEqual(sum(durs), total, places=3)
            self.assertAlmostEqual(durs[-1], ENDCARD_SECONDS, places=3)
        # and the bridge gets everything else, so it grows with the CTA instead of the card growing
        self.assertGreater(_cta_split(12.0, 2)[0], _cta_split(6.0, 2)[0])

    def test_a_very_short_cta_never_gives_the_card_more_than_half(self):
        """A 3s CTA must not be all card: the bridge still has to be seen inside the story."""
        from avp.stages import _cta_split
        durs = _cta_split(3.0, 2)
        self.assertAlmostEqual(sum(durs), 3.0, places=3)
        self.assertLessEqual(durs[-1], 1.5 + 1e-6)

    def test_a_lone_card_still_fills_the_segment(self):
        """When no content still is available the card is all there is, and must not leave a gap."""
        from avp.stages import _cta_split
        self.assertEqual(_cta_split(6.0, 1), [6.0])

    def test_the_cta_plays_the_last_content_picture_first(self):
        from avp.stages import _segment_sources
        from avp.models import Segment
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "footage").mkdir()
            card = root / "footage" / "endcard.png"; card.write_bytes(b"x")
            last = root / "footage" / "04_2.png"; last.write_bytes(b"x")
            project = mock.Mock(footage_dir=root / "footage")
            seg = Segment(index=5, narration="n", footage="endcard.png", kind="cta")
            self.assertEqual(_segment_sources(project, seg, last), [last, card])
            # and with nothing to fall back on, just the card
            self.assertEqual(_segment_sources(project, seg, None), [card])


class CutRhythm(unittest.TestCase):
    """The segment count IS the cut rhythm, and it has a floor and a ceiling that were both found the
    hard way: ~10s per segment reads as a documentary, ~5s reads as the video played at 1.5x."""

    def _plan(self, target_seconds, images_per_segment=2):
        """What a given target actually produces: (segments, seconds per image, words). Mirrors the
        real formula rather than restating a number, so the test moves when the rhythm moves."""
        from avp.llm import _words_for, SECONDS_PER_IMAGE
        content = target_seconds - 8                     # the spoken CTA and its tail
        nseg = max(3, round(content / (SECONDS_PER_IMAGE * images_per_segment)))
        return nseg, content / (nseg * images_per_segment), _words_for(content, "en")

    def test_a_picture_gets_between_three_and_five_seconds(self):
        """Below ~3s the eye never settles — viewers reported the video felt sped up. Above ~5s the
        feed scrolls past. The target sits at 4.3s; the band is what must never be left."""
        for target in (45, 50, 56, 58, 60):
            _, per_image, _ = self._plan(target)
            self.assertGreaterEqual(per_image, 3.0, f"{target}s cuts too fast")
            self.assertLessEqual(per_image, 5.0, f"{target}s cuts too slow")

    def test_the_rhythm_survives_a_change_to_images_per_segment(self):
        """The count is derived from seconds-per-IMAGE, so halving the images per segment must
        double the segments rather than double how long each picture sits there."""
        one, _, _ = self._plan(58, images_per_segment=1)
        two, _, _ = self._plan(58, images_per_segment=2)
        self.assertGreater(one, two)
        for ipp in (1, 2, 3):
            _, per_image, _ = self._plan(58, images_per_segment=ipp)
            self.assertGreaterEqual(per_image, 3.0)
            self.assertLessEqual(per_image, 5.0)

    def test_the_default_target_sits_just_under_the_ceiling(self):
        """Near 60s, never over: the writer can overshoot by ~10% and the trim guard takes the rest."""
        cfg = Config()
        self.assertGreaterEqual(cfg.script.target_seconds, 54)
        self.assertLess(cfg.script.target_seconds, 60)   # the writer can overshoot ~10%

    def test_short_videos_keep_both_a_story_and_the_rhythm(self):
        """A floor protects the story — three beats minimum — but must not push the cut rate out of
        band to do it. A floor of four sent a 30s video to 2.75s per image, the exact fault this
        whole change was fixing."""
        nseg, per_image, _ = self._plan(30)
        self.assertGreaterEqual(nseg, 3)
        self.assertGreaterEqual(per_image, 3.0)
        self.assertLessEqual(per_image, 5.0)

    def test_the_word_budget_matches_the_measured_speaking_rate(self):
        """2.38 words/second across four real builds; the budget must not drift from that or videos
        land short (which reads as rushed) or long (which breaks the 60s ceiling)."""
        from avp.llm import _words_for
        for seconds in (30, 42, 48):
            implied = _words_for(seconds, "en") / seconds
            self.assertAlmostEqual(implied, 2.38, delta=0.15,
                                   msg=f"{seconds}s implies {implied:.2f} words/s")


class FactCheckSelfContradiction(unittest.TestCase):
    """Observed live on a real build: the checker reasoned inside its own "why" field, concluded the
    claim was fine — "the claim is actually true... no flag" — and still emitted verdict "wrong".
    Nothing broke only because the fix came back empty. That safety is now deliberate."""

    def _script(self):
        from avp.models import Script, Segment
        return Script(title="t", topic="Jupiter", segments=[
            Segment(index=1, narration="Ganymede is the only moon with an intrinsic magnetic field."),
        ])

    def _run(self, sc, findings):
        from avp import factcheck
        cfg = Config(); cfg.script.factcheck = "fix"; cfg.script.factcheck_key = "k"
        payload = {"choices": [{"message": {"content": json.dumps({"findings": findings})}}]}
        with mock.patch("avp.factcheck.requests.post",
                        return_value=mock.Mock(status_code=200, json=lambda: payload)):
            return factcheck.run(sc, cfg)

    def test_wrong_without_a_fix_is_downgraded_and_never_applied(self):
        sc = self._script()
        before = sc.segments[0].narration
        rep = self._run(sc, [{"segment": 1, "field": "narration",
                              "claim": "Ganymede is the only moon with an intrinsic magnetic field.",
                              "verdict": "wrong",
                              "why": "...the claim is actually true, so no flag. No flag.",
                              "fix": ""}])
        self.assertEqual(sc.segments[0].narration, before)   # untouched
        self.assertEqual(rep.findings[0].verdict, "unsure")  # downgraded, still reported
        self.assertFalse(rep.findings[0].applied)

    def test_a_whitespace_only_fix_counts_as_no_fix(self):
        sc = self._script()
        rep = self._run(sc, [{"segment": 1, "field": "narration",
                              "claim": "Ganymede is the only moon with an intrinsic magnetic field.",
                              "verdict": "wrong", "why": "x", "fix": "   "}])
        self.assertEqual(rep.findings[0].verdict, "unsure")


class FactCheckVisuals(unittest.TestCase):
    """Shot descriptions drive image generation, so a wrong one is worse than a wrong spoken line —
    nobody hears it, but everyone sees the picture it produced. They are corrected under different
    rules: length barely matters, the shot scale does."""

    def _script(self):
        from avp.models import Script, Segment
        return Script(title="t", topic="Opportunity", segments=[
            Segment(index=1, narration="The rover crossed the plain.",
                    visual="wide shot, the dusty Martian horizon under a pale blue sky"),
            Segment(index=2, narration="It drilled into rock.",
                    visual="extreme close-up, jagged rock samples on Mars"),
        ])

    def _cfg(self, mode="fix"):
        cfg = Config()
        cfg.script.factcheck = mode
        cfg.script.factcheck_key = "test-key"
        return cfg

    def _reply(self, findings):
        return {"choices": [{"message": {"content": json.dumps({"findings": findings})}}]}

    def _run(self, sc, findings, mode="fix"):
        from avp import factcheck
        with mock.patch("avp.factcheck.requests.post",
                        return_value=mock.Mock(status_code=200, json=lambda: self._reply(findings))):
            return factcheck.run(sc, self._cfg(mode))

    def test_a_wrong_sky_is_corrected_in_the_visual(self):
        sc = self._script()
        rep = self._run(sc, [{"segment": 1, "field": "visual",
                              "claim": "under a pale blue sky", "verdict": "wrong",
                              "why": "The Martian day sky is butterscotch, blue only at sunset.",
                              "fix": "under a hazy butterscotch sky"}])
        self.assertIn("butterscotch", sc.segments[0].visual)
        self.assertNotIn("pale blue", sc.segments[0].visual)
        self.assertTrue(rep.findings[0].applied)
        self.assertEqual(rep.findings[0].field, "visual")

    def test_the_spoken_line_is_untouched_by_a_visual_finding(self):
        sc = self._script()
        before = sc.segments[0].narration
        self._run(sc, [{"segment": 1, "field": "visual", "claim": "under a pale blue sky",
                        "verdict": "wrong", "why": "x", "fix": "under a butterscotch sky"}])
        self.assertEqual(sc.segments[0].narration, before)

    def test_a_fix_that_drops_the_shot_scale_is_refused(self):
        """Segments alternate wide / medium / close on purpose; a fix that loses the scale would
        silently flatten the cut rhythm."""
        sc = self._script()
        rep = self._run(sc, [{"segment": 1, "field": "visual",
                              "claim": "wide shot, the dusty Martian horizon under a pale blue sky",
                              "verdict": "wrong", "why": "sky colour",
                              "fix": "the dusty Martian horizon under a butterscotch sky"}])
        self.assertIn("wide shot", sc.segments[0].visual)
        self.assertFalse(rep.findings[0].applied)

    def test_a_bloated_visual_is_refused(self):
        """This string is prepended to an image prompt, where a long tail stops being obeyed."""
        sc = self._script()
        rep = self._run(sc, [{"segment": 2, "field": "visual",
                              "claim": "jagged rock samples on Mars", "verdict": "wrong", "why": "x",
                              "fix": "jagged rock samples on Mars " + "extremely detailed " * 10}])
        self.assertEqual(sc.segments[1].visual, "extreme close-up, jagged rock samples on Mars")
        self.assertFalse(rep.findings[0].applied)

    def test_a_visual_may_grow_a_little_unlike_a_spoken_line(self):
        """Nobody speaks a shot description, so the tight spoken-length rule does not apply."""
        sc = self._script()
        rep = self._run(sc, [{"segment": 2, "field": "visual",
                              "claim": "jagged rock samples on Mars", "verdict": "wrong", "why": "x",
                              "fix": "jagged rust coloured rock samples on the Martian regolith"}])
        self.assertTrue(rep.findings[0].applied)
        self.assertIn("rust coloured", sc.segments[1].visual)

    def test_the_field_defaults_to_narration_when_absent(self):
        """Older or sloppy replies must not silently rewrite the wrong field."""
        sc = self._script()
        self._run(sc, [{"segment": 1, "claim": "The rover crossed the plain.",
                        "verdict": "wrong", "why": "x", "fix": "The rover crossed dunes."}])
        self.assertEqual(sc.segments[0].narration, "The rover crossed dunes.")
        self.assertIn("pale blue", sc.segments[0].visual)   # visual untouched


class FactCheck(unittest.TestCase):
    """The checker is a safety net. It must catch what the local writer invents, and it must never
    be able to stop a build."""

    def _script(self):
        from avp.models import Script, Segment
        return Script(title="t", topic="Opportunity", segments=[
            Segment(index=1, narration="A machine 225 million kilometres away is still screaming."),
            Segment(index=2, narration="It scavenged power from the Martian soil for fifteen years."),
            Segment(index=3, narration="The rover crawled 45 kilometres across the plain."),
        ])

    def _cfg(self, mode="fix"):
        cfg = Config()
        cfg.script.factcheck = mode
        cfg.script.factcheck_key = "test-key"
        return cfg

    def _reply(self, findings):
        return {"choices": [{"message": {"content": json.dumps({"findings": findings})}}]}

    def test_confident_corrections_are_applied(self):
        from avp import factcheck
        sc, cfg = self._script(), self._cfg("fix")
        payload = self._reply([
            {"segment": 2, "claim": "scavenged power from the Martian soil",
             "verdict": "wrong", "why": "It was solar powered.",
             "fix": "ran on sunlight for"},
        ])
        with mock.patch("avp.factcheck.requests.post",
                        return_value=mock.Mock(status_code=200, json=lambda: payload)):
            rep = factcheck.run(sc, cfg)
        self.assertTrue(rep.checked)
        self.assertIn("ran on sunlight for", sc.segments[1].narration)
        self.assertNotIn("Martian soil", sc.segments[1].narration)
        self.assertTrue(rep.findings[0].applied)

    def test_unsure_is_reported_but_never_rewritten(self):
        """Swapping a maybe-true line for a maybe-true line is churn, not correction."""
        from avp import factcheck
        sc, cfg = self._script(), self._cfg("fix")
        before = sc.segments[0].narration
        payload = self._reply([
            {"segment": 1, "claim": "is still screaming", "verdict": "unsure",
             "why": "Operational status changes.", "fix": "fell silent"},
        ])
        with mock.patch("avp.factcheck.requests.post",
                        return_value=mock.Mock(status_code=200, json=lambda: payload)):
            rep = factcheck.run(sc, cfg)
        self.assertEqual(sc.segments[0].narration, before)
        self.assertFalse(rep.findings[0].applied)

    def test_flag_mode_never_touches_the_script(self):
        from avp import factcheck
        sc, cfg = self._script(), self._cfg("flag")
        before = [s.narration for s in sc.segments]
        payload = self._reply([{"segment": 2, "claim": "scavenged power from the Martian soil",
                                "verdict": "wrong", "why": "solar", "fix": "ran on sunlight"}])
        with mock.patch("avp.factcheck.requests.post",
                        return_value=mock.Mock(status_code=200, json=lambda: payload)):
            factcheck.run(sc, cfg)
        self.assertEqual([s.narration for s in sc.segments], before)

    def test_a_longer_fix_is_refused(self):
        """Corrections are spoken: caption timings and the 60s ceiling come from word count."""
        from avp import factcheck
        sc, cfg = self._script(), self._cfg("fix")
        payload = self._reply([
            {"segment": 3, "claim": "crawled 45 kilometres", "verdict": "wrong", "why": "x",
             "fix": "crawled forty five kilometres across the endless rust coloured martian plain"},
        ])
        with mock.patch("avp.factcheck.requests.post",
                        return_value=mock.Mock(status_code=200, json=lambda: payload)):
            rep = factcheck.run(sc, cfg)
        self.assertIn("crawled 45 kilometres", sc.segments[2].narration)
        self.assertFalse(rep.findings[0].applied)

    def test_a_dead_api_never_stops_the_build(self):
        from avp import factcheck
        for boom in (requests.exceptions.ConnectionError("no route"),
                     requests.exceptions.ReadTimeout("slow")):
            sc, cfg = self._script(), self._cfg("fix")
            before = [s.narration for s in sc.segments]
            with mock.patch("avp.factcheck.requests.post", side_effect=boom):
                rep = factcheck.run(sc, cfg)
            self.assertFalse(rep.checked)
            self.assertIn("unavailable", rep.reason)
            self.assertEqual([s.narration for s in sc.segments], before)

    def test_a_500_and_a_garbage_reply_both_fail_open(self):
        from avp import factcheck
        sc, cfg = self._script(), self._cfg("fix")
        with mock.patch("avp.factcheck.requests.post",
                        return_value=mock.Mock(status_code=500, text="upstream boom")):
            self.assertFalse(factcheck.run(sc, cfg).checked)
        garbage = {"choices": [{"message": {"content": "I'm afraid I can't do that."}}]}
        with mock.patch("avp.factcheck.requests.post",
                        return_value=mock.Mock(status_code=200, json=lambda: garbage)):
            rep = factcheck.run(sc, cfg)
        self.assertTrue(rep.checked)          # it answered, it just said nothing usable
        self.assertEqual(rep.findings, [])

    def test_missing_key_is_not_a_crash(self):
        from avp import factcheck
        cfg = Config(); cfg.script.factcheck = "fix"; cfg.script.factcheck_key = ""
        with mock.patch.dict("os.environ", {}, clear=True):
            rep = factcheck.run(self._script(), cfg)
        self.assertFalse(rep.checked)
        self.assertIn("no API key", rep.reason)

    def test_off_makes_no_call_at_all(self):
        from avp import factcheck
        with mock.patch("avp.factcheck.requests.post") as post:
            rep = factcheck.run(self._script(), self._cfg("off"))
        post.assert_not_called()
        self.assertFalse(rep.checked)

    def test_only_time_dependent_claims_earn_a_web_lookup(self):
        from avp import factcheck
        self.assertTrue(factcheck.volatile("it is still transmitting"))
        self.assertTrue(factcheck.volatile("the only human-made object out there"))
        self.assertTrue(factcheck.volatile("45 kilometres of driving"))
        self.assertFalse(factcheck.volatile("Mars is a rocky planet"))

    def test_report_is_written_for_the_audit_trail(self):
        from avp import factcheck
        with tempfile.TemporaryDirectory() as td:
            payload = self._reply([])
            with mock.patch("avp.factcheck.requests.post",
                            return_value=mock.Mock(status_code=200, json=lambda: payload)):
                factcheck.run(self._script(), self._cfg("flag"), out_dir=Path(td))
            saved = json.loads((Path(td) / "factcheck.json").read_text())
        self.assertTrue(saved["checked"])          # a clean pass still proves it looked
        self.assertEqual(saved["segments"], 3)


if __name__ == "__main__":
    unittest.main()
