"""Command-line interface.

Typical flow (semi-automatic, human-in-the-loop):
    avp new saturn-rings --topic "Why Saturn's rings are disappearing"
    # -> edit projects/saturn-rings/script.md  (check the facts!)
    avp build saturn-rings
    # -> projects/saturn-rings/saturn-rings.mp4

Per-stage commands (script, voice, footage, captions, assemble) let you re-run one step.
'run' does the whole thing without stopping for review — handy for smoke tests.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

from . import pipeline, stages
from .config import Config
from .log import get_logger, setup_logging
from .manifest import VideoProject


def _build_parser() -> argparse.ArgumentParser:
    # Common options live on a parent parser added to every subcommand, so they work
    # whether typed before OR after the subcommand (e.g. `avp voice slug --config x`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml", help="path to config.yaml")
    common.add_argument("-v", "--verbose", action="store_true")

    p = argparse.ArgumentParser(
        prog="avp", parents=[common],
        description="AUT Video Pipeline — local faceless-video engine",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    new = sub.add_parser("new", parents=[common],
                         help="create project + generate script, then STOP for your review")
    new.add_argument("slug")
    new.add_argument("--topic", required=True)

    sc = sub.add_parser("script", parents=[common], help="(re)generate the script")
    sc.add_argument("slug")
    sc.add_argument("--topic", default=None)

    for name, helptext in [
        ("voice", "synthesize narration (TTS)"),
        ("footage", "resolve/download footage"),
        ("captions", "generate karaoke captions"),
        ("assemble", "render the final mp4"),
        ("metadata", "generate platform titles/descriptions/hashtags"),
    ]:
        s = sub.add_parser(name, parents=[common], help=helptext)
        s.add_argument("slug")

    b = sub.add_parser("build", parents=[common],
                       help="voice -> footage -> captions -> assemble (resumable)")
    b.add_argument("slug")
    b.add_argument("--force", action="store_true", help="rerun even completed stages")

    r = sub.add_parser("run", parents=[common],
                       help="full pipeline incl. script, no review stop (smoke test)")
    r.add_argument("slug")
    r.add_argument("--topic", required=True)
    r.add_argument("--force", action="store_true")

    pub = sub.add_parser("publish", parents=[common],
                         help="publish to socials via Postiz (dry-run unless --go)")
    pub.add_argument("slug")
    pub.add_argument("--go", action="store_true", help="actually post (needs Postiz configured)")
    pub.add_argument("--platforms", nargs="*", default=None, help="override platforms")

    sub.add_parser("list", parents=[common], help="list projects and stage status")

    sub.add_parser("config-get", parents=[common], help="print the current config as JSON")
    cs = sub.add_parser("config-set", parents=[common], help="merge a JSON patch into config.yaml")
    cs.add_argument("patch", help="JSON object to merge into config.yaml")
    return p


def _cmd_list(cfg: Config) -> int:
    root = Path(cfg.paths.projects_dir)
    manifests = sorted(root.glob("*/manifest.json")) if root.exists() else []
    if not manifests:
        print("(no projects yet — try: avp new my-first --topic \"...\")")
        return 0
    for mpath in manifests:
        m = json.loads(mpath.read_text())
        status = " ".join(f"{k}:{v.get('state', '?')[:4]}" for k, v in m["stages"].items())
        print(f"{m['slug']:<28} {status}")
    return 0


def _deep_merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def _config_set(path: str, patch: dict) -> None:
    import yaml
    from pathlib import Path
    p = Path(path)
    data = (yaml.safe_load(p.read_text()) if p.exists() else {}) or {}
    _deep_merge(data, patch)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def _quiet_libs() -> None:
    """Silence noisy ML library warnings/telemetry so the build log stays clean."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore")


def main(argv: list[str] | None = None) -> int:
    _quiet_libs()
    args = _build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)
    log = get_logger()

    if args.cmd == "list":
        return _cmd_list(cfg)

    if args.cmd == "config-get":
        print(json.dumps(cfg.to_dict()))
        return 0
    if args.cmd == "config-set":
        try:
            patch = json.loads(args.patch)
        except json.JSONDecodeError as e:
            log.error("config-set: patch is not valid JSON (%s)", e)
            return 1
        if not isinstance(patch, dict):
            log.error("config-set: patch must be a JSON object, got %s", type(patch).__name__)
            return 1
        _config_set(args.config, patch)
        print("ok")
        return 0

    if args.cmd == "new":
        project = VideoProject.create(args.slug, cfg)
        setup_logging(level, project.log_file)
        stages.stage_script(project, cfg, args.topic)
        print(f"\n📝 Review & edit:  {project.script_md}")
        print(f"   Then build:     avp build {args.slug}\n")
        return 0

    project = VideoProject(args.slug, cfg)
    setup_logging(level, project.log_file)

    try:
        if args.cmd == "script":
            stages.stage_script(project, cfg, args.topic)
            print(f"📝 {project.script_md} — review, then: avp build {args.slug}")
        elif args.cmd == "voice":
            stages.stage_voice(project, cfg)
        elif args.cmd == "footage":
            stages.stage_footage(project, cfg)
        elif args.cmd == "captions":
            stages.stage_captions(project, cfg)
        elif args.cmd == "assemble":
            out = stages.stage_assemble(project, cfg)
            print(f"✅ {out}")
        elif args.cmd == "metadata":
            stages.stage_metadata(project, cfg)
            print(f"📋 {project.root / 'metadata.md'}")
        elif args.cmd == "publish":
            from . import publish as publish_mod
            publish_mod.stage_publish(project, cfg, go=args.go, platforms=args.platforms)
            print(f"📤 {project.root / 'publish_plan.json'}")
        elif args.cmd == "build":
            out = pipeline.build(project, cfg, force=args.force)
            print(f"✅ {out}")
        elif args.cmd == "run":
            stages.stage_script(project, cfg, args.topic)
            out = pipeline.build(project, cfg, force=args.force)
            print(f"✅ {out}")
    except Exception as e:  # noqa: BLE001 — present a clean message, full trace in -v
        log.error("%s", e)
        if args.verbose:
            raise
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
