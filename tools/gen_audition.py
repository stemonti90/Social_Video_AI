"""Generate 3 music auditions (one per mood) with Stable Audio Open. Run:
PYTHONPATH=src .venv/bin/python tools/gen_audition.py
"""
import time
from pathlib import Path

from avp.music import PROMPTS, generate_track

out = Path("auditions/music")
out.mkdir(parents=True, exist_ok=True)
for i, mood in enumerate(["ethereal", "cinematic", "dark"]):
    t0 = time.time()
    print(f"[{i+1}/3] generating '{mood}' …", flush=True)
    generate_track(out / f"{mood}.wav", mood=mood, seconds=40.0, steps=100, seed=42 + i)
    print(f"    done in {time.time()-t0:.0f}s", flush=True)
print("ALL DONE ->", out.resolve())
