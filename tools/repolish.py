"""Rewrite an already-built video's narration in the channel's current voice, keeping its images.

Usage: PYTHONPATH=src .venv/bin/python tools/repolish.py <slug>

Why. The back catalogue was written before the wonder voice: "The Gutted Moon of Saturn", "The World
That Bleeds Fire", "The Moon's Dead Mask" — exactly the lexicon the channel has since banned. Rebuilding
those videos for TikTok with the old narration would republish it. This runs the same chain a new
script gets — fact sheet → polish → fact-check — on the existing script, leaves VISUAL/KEYWORDS (and
so the generated images) untouched, and saves script.json + script.md. The voice stage then re-
synthesises only the lines that changed (its cache is keyed by text), captions re-adapt (theirs is
keyed by source text and duration), assemble picks it all up.

Exit 0 whether or not the polish was accepted — a rejected polish keeps the original lines, and the
caller decides what to do with a still-morbid script (it warns loudly).
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avp import brief, factcheck, polish, stages  # noqa: E402
from avp.config import Config  # noqa: E402
from avp.llm import morbid_in_script  # noqa: E402
from avp.manifest import VideoProject  # noqa: E402
from avp.models import Segment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("repolish")


def main(slug: str) -> int:
    cfg = Config.load("config.yaml")
    project = VideoProject(slug, cfg)
    script = stages.load_script(project)
    topic = project.manifest.data.get("topic") or script.topic or script.title
    before = {s.index: s.narration for s in script.segments}
    old_cta = next((s for s in script.segments if s.kind == "cta"), None)  # keeps its endcard picture
    script.segments = [s for s in script.segments if s.kind != "cta"]      # the CTA is re-composed below
    facts = brief.build(topic, cfg, out_dir=project.root)
    new = polish.run(script, facts, cfg, out_dir=project.root)
    accepted = new is not script
    if accepted:
        rep = factcheck.run(new, cfg, out_dir=project.root)
        wrong = [f for f in rep.findings if f.verdict == "wrong"]
        log.info("fact-check after polish: %d wrong, %d unsure", len(wrong), len(rep.findings) - len(wrong))
    if cfg.funnel.enabled:
        new.segments.append(Segment(index=len(new.segments) + 1, narration=stages._cta_narration(new, cfg),
                                    visual="App endcard", keywords=[], kind="cta",
                                    footage=old_cta.footage if old_cta else None))
    project.script_json.write_text(stages._json(new.to_dict()))
    stages.emit_script_md(new, project.script_md)
    project.manifest.data["title"] = new.title
    project.manifest.save() if hasattr(project.manifest, "save") else None
    changed = sum(1 for s in new.segments if before.get(s.index) != s.narration)
    log.info("%s: polish %s — title %r, %d line(s) changed", slug, "ACCEPTED" if accepted else "REJECTED (original kept)",
             new.title, changed)
    mw = morbid_in_script({"title": new.title, "segments": [{"narration": s.narration} for s in new.segments],
                           "cta_bridge": new.cta_bridge})
    if mw:
        log.warning("%s: the script STILL carries the morbid word %r", slug, mw)
    for s in new.segments:
        log.info("  %d: %s", s.index, s.narration)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
