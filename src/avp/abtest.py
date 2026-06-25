"""A/B harness: render short samples under different render configs and compare them on MEASURABLE
metrics (wall time, peak RSS, success/error) so the best fp16 / voice setting is choosable without
reading code. Perceived quality still needs an ear, so the harness also writes the actual A and B
samples side by side and a metric-based recommendation (with a 'best auto' that the caller may save
as the default, override always wins).

Each variant runs in its OWN subprocess (`python -m avp.abtest worker <json>`) so memory is isolated
and the peak RSS is real — critical on the 24GB machine where fp16 vs fp32 is exactly a memory call.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .log import get_logger

log = get_logger("avp.abtest")


def _peak_mb() -> float:
    """This process's peak resident memory in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)


# ── variant workers (run inside the subprocess) ──────────────────────────────────────────────────
def _worker_fp16(args: dict) -> dict:
    """Generate a short Stable Audio bed at the requested dtype; report time + peak RSS."""
    from . import music
    dtype = args["dtype"]                       # "float32" | "float16"
    out = Path(args["out"])
    t0 = time.time()
    music.generate_track(out, mood="documentary", seconds=float(args.get("seconds", 6.0)),
                         steps=int(args.get("steps", 40)), seed=7,
                         device=args.get("device", "mps"), dtype=dtype)
    return {"variant": dtype, "ok": out.exists() and out.stat().st_size > 1000,
            "seconds": round(time.time() - t0, 1), "peak_mb": _peak_mb(), "file": str(out)}


def _worker_voice(args: dict) -> dict:
    """Synthesize a short line with one voice config; report time + peak RSS + audio length."""
    from . import ffmpeg, tts
    out = Path(args["out"])
    prov = tts.KokoroProvider(args.get("lang_code", "a"), args.get("voice", "af_heart"),
                              args.get("device", "mps"), float(args.get("speed", 1.0)))
    t0 = time.time()
    prov.synthesize(args["text"], out)
    dur = ffmpeg.ffprobe_duration(out) if out.exists() else 0.0
    return {"variant": args["label"], "ok": out.exists() and out.stat().st_size > 1000,
            "seconds": round(time.time() - t0, 1), "peak_mb": _peak_mb(),
            "audio_seconds": round(dur, 2), "voice": args.get("voice"),
            "speed": args.get("speed"), "file": str(out)}


_WORKERS = {"fp16": _worker_fp16, "voice": _worker_voice}


def _run_variant(kind: str, args: dict) -> dict:
    """Spawn one isolated subprocess for a variant and return its metrics (or an error record)."""
    src_dir = str(Path(__file__).resolve().parent.parent)
    payload = json.dumps({"kind": kind, "args": args})
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-m", "avp.abtest", "worker", payload],
                           cwd=src_dir, capture_output=True, text=True, timeout=900,
                           env={"PYTHONPATH": src_dir, **_os_environ()})
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if r.returncode == 0 and line:
            return json.loads(line[-1])
        return {"variant": args.get("dtype") or args.get("label", "?"), "ok": False,
                "seconds": round(time.time() - t0, 1), "peak_mb": 0.0,
                "error": (r.stderr or "no output").strip().splitlines()[-1][:200]}
    except Exception as e:  # noqa: BLE001
        return {"variant": args.get("dtype") or args.get("label", "?"), "ok": False,
                "seconds": round(time.time() - t0, 1), "peak_mb": 0.0, "error": str(e)[:200]}


def _os_environ() -> dict:
    import os
    return dict(os.environ)


# ── recommendation (pure, unit-tested) ───────────────────────────────────────────────────────────
def recommend_fp16(results: list[dict]) -> dict:
    """Pick fp16 only when it succeeds AND is at least as fast and lighter than fp32 (fp16 halves the
    ~5GB Stable Audio footprint — a real win on 24GB) — but quality still wants an ear, so we say so."""
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return {"choice": None, "reason": "all variants failed — keep current"}
    f16 = next((r for r in ok if r["variant"] == "float16"), None)
    f32 = next((r for r in ok if r["variant"] == "float32"), None)
    if f16 and f32:
        if f16["seconds"] <= f32["seconds"] * 1.05 and f16["peak_mb"] < f32["peak_mb"]:
            return {"choice": "float16",
                    "reason": (f"fp16 faster/lighter ({f16['seconds']}s/{f16['peak_mb']}MB vs "
                               f"{f32['seconds']}s/{f32['peak_mb']}MB) — verify the sample by ear")}
        return {"choice": "float32", "reason": "fp16 not clearly better; fp32 is the safe default"}
    return {"choice": ok[0]["variant"], "reason": "only one variant succeeded"}


def recommend_voice(results: list[dict]) -> dict:
    """Voice naturalness is the ear's call, so the harness recommends the stable variant that rendered
    fastest and reports each one's pace (audio length) — the user picks from the side-by-side samples."""
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return {"choice": None, "reason": "all voice variants failed"}
    best = min(ok, key=lambda r: r["seconds"])      # fastest clean render; quality compared by ear
    return {"choice": best["variant"],
            "reason": f"rendered cleanly ({best.get('audio_seconds')}s of audio) — compare A/B by ear for naturalness"}


# ── orchestration ────────────────────────────────────────────────────────────────────────────────
def run_ab(out_dir: Path, cfg, text: str = "Otto pianeti orbitano il Sole a velocità diverse.",
           kinds=("fp16", "voice")) -> dict:
    """Run the requested A/B comparisons, write the samples + ab_report.json into out_dir, return the
    report. Real renders in isolated subprocesses; safe to call from a CLI."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.tts.device
    report: dict = {"text": text, "fp16": None, "voice": None}

    if "fp16" in kinds:
        results = [_run_variant("fp16", {"dtype": dt, "out": str(out_dir / f"fp16_{dt}.wav"),
                                         "device": device})
                   for dt in ("float32", "float16")]
        report["fp16"] = {"results": results, "recommendation": recommend_fp16(results)}
        log.info("A/B fp16 → %s", report["fp16"]["recommendation"])

    if "voice" in kinds:
        lang = cfg.script.language
        lc, v = tts_lang(lang)
        variants = [
            {"label": "current", "voice": v, "lang_code": lc, "speed": 0.94 if lang == "it" else 1.0},
            {"label": "slower",  "voice": v, "lang_code": lc, "speed": 0.88 if lang == "it" else 0.92},
        ]
        results = [_run_variant("voice", {**vr, "text": text, "device": device,
                                          "out": str(out_dir / f"voice_{vr['label']}.wav")})
                   for vr in variants]
        report["voice"] = {"results": results, "recommendation": recommend_voice(results)}
        log.info("A/B voice → %s", report["voice"]["recommendation"])

    (out_dir / "ab_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def tts_lang(language: str):
    from .tts import LANG_KOKORO
    return LANG_KOKORO.get(language, LANG_KOKORO["en"])


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "worker":
        spec = json.loads(argv[1])
        result = _WORKERS[spec["kind"]](spec["args"])
        print(json.dumps(result))
        return 0
    print("usage: python -m avp.abtest worker <json>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
