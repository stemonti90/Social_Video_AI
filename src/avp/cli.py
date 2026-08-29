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
import re
import shutil
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
    pub.add_argument("--at", default=None,
                     help="ISO 8601 time to schedule (e.g. 2026-07-01T18:00:00Z); omit = post now")

    sub.add_parser("list", parents=[common], help="list projects and stage status")

    dl = sub.add_parser("delete", parents=[common],
                        help="permanently delete a project and its folder")
    dl.add_argument("slug")

    sub.add_parser("config-get", parents=[common], help="print the current config as JSON")
    cs = sub.add_parser("config-set", parents=[common], help="merge a JSON patch into config.yaml")
    cs.add_argument("patch", help="JSON object to merge into config.yaml")

    ab = sub.add_parser("ab", parents=[common], help="A/B compare fp16 vs fp32 and voice variants")
    ab.add_argument("--kinds", nargs="*", default=["fp16", "voice"], help="which comparisons to run")
    ab.add_argument("--text", default=None, help="sample line to synthesize for the voice A/B")

    au = sub.add_parser("auto", parents=[common],
                        help="generate N videos and schedule them to Postiz (daily automation)")
    au.add_argument("--count", type=int, default=None, help="override auto.count for this run")
    au.add_argument("--dry-run", action="store_true",
                    help="pick topics + show the plan; do NOT generate or post")
    au.add_argument("--no-publish", action="store_true",
                    help="generate the videos but do not schedule them to Postiz")

    tp = sub.add_parser("topics", parents=[common], help="show, add to, or refill the topic queue")
    tp.add_argument("--refill", action="store_true", help="top the queue up via the LLM now")
    tp.add_argument("--add", nargs="*", default=None, help="append one or more topics to the queue")

    cn = sub.add_parser("connect", parents=[common],
                        help="link a social account (OAuth) so `publish --go` can post to it")
    cn.add_argument("platform", choices=["tiktok", "instagram", "youtube"])
    cn.add_argument("--code", default=None,
                    help="the authorisation code from the callback page (second step)")

    sub.add_parser("accounts", parents=[common], help="show the linked social accounts")

    dc = sub.add_parser("disconnect", parents=[common], help="forget a linked social account")
    dc.add_argument("platform", choices=["tiktok", "instagram", "youtube"])

    wk = sub.add_parser("worker", parents=[common],
                        help="claim + render jobs from the control server (Mac GPU worker)")
    wk.add_argument("--server", default=None, help="control server base URL (or env AVP_CONTROL_URL)")
    wk.add_argument("--token", default=None, help="control token (or env AVP_CONTROL_TOKEN)")
    wk.add_argument("--once", action="store_true", help="process a single job then exit")
    wk.add_argument("--poll", type=int, default=60, help="seconds between polls when idle")
    wk.add_argument("--name", default="mac-worker", help="worker id reported to the server")
    return p


def _cmd_social(args, cfg: Config) -> int:
    """connect / accounts / disconnect.

    `connect` is deliberately two steps. The OAuth redirect has to land on an HTTPS URL the platform
    trusts, so it lands on a static page on the main site; that page shows the code and the exact
    command to paste back. It costs one paste roughly once a year (TikTok refresh tokens last 365
    days) and saves running a callback service just to catch a query string."""
    from . import social

    if args.cmd == "accounts":
        linked = social.tokens.connected()
        if not linked:
            print("No linked accounts. Run `avp connect tiktok` (or instagram / youtube).")
            return 0
        for plat, info in sorted(linked.items()):
            left = info["expires_in"]
            # TikTok access tokens last 24h (the refresh token does the real work), so "<1d" is the
            # normal healthy state — show hours instead of a scary-looking "0d".
            when = ("expired" if info["expired"]
                    else "no expiry" if not left
                    else f"{left // 86400}d left" if left >= 86400
                    else f"{left // 3600}h left")
            print(f"  {plat:<10} {info['account']:<28} {when}")
        return 0

    if args.cmd == "disconnect":
        print(f"Forgot {args.platform}." if social.tokens.forget(args.platform)
              else f"{args.platform} was not linked.")
        return 0

    if args.code:
        info = social.finish_connect(args.platform, args.code, cfg)
        print(f"Linked {args.platform}: {info.get('account', '?')}")
        return 0

    url, state = social.start_connect(args.platform, cfg)
    print(f"\n1. Open this and authorise the account:\n\n{url}\n")
    print(f"2. You land on {social.redirect_uri(args.platform)} — copy the command it shows,")
    print(f"   or run:  avp connect {args.platform} --code <CODE>\n")
    print(f"   (state = {state}; the page shows the same value if the link is fresh)")
    return 0


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


# A safe project folder name: lowercase/digits/_/- only, first char alnum or _ (never a leading
# '-' that looks like a flag, never a leading '.'). Real slugs come from slugify ([a-z0-9-]); the
# leading '_' covers existing test projects like '_smoke'. Traversal is blocked here AND below.
_SLUG_RE = re.compile(r"^[a-z0-9_][a-z0-9_-]*$")


def _cmd_delete(cfg: Config, slug: str) -> int:
    """Permanently remove a project folder. Deliberately defensive — this is destructive:
    the slug must be a single safe path component, the resolved path must sit *directly*
    inside the projects dir, and it must actually be a project (have a manifest)."""
    log = get_logger()
    if not slug or ".." in slug or "/" in slug or "\\" in slug or not _SLUG_RE.match(slug):
        log.error("delete: invalid project slug %r", slug)
        return 1
    projects = Path(cfg.paths.projects_dir).resolve()
    target = (projects / slug).resolve()
    if target.parent != projects:                       # never escape the projects directory
        log.error("delete: refusing to delete outside the projects directory (%s)", target)
        return 1
    if not target.is_dir():
        log.error("delete: project %r not found", slug)
        return 1
    if not (target / "manifest.json").exists():         # only delete real projects, never stray dirs
        log.error("delete: %r has no manifest.json — not a project, not deleting", slug)
        return 1
    shutil.rmtree(target)
    log.info("Deleted project %s (%s)", slug, target)
    print(f"deleted {slug}")
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
    if args.cmd == "delete":
        return _cmd_delete(cfg, args.slug)

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

    if args.cmd == "ab":
        from . import abtest
        out_dir = Path(cfg.paths.projects_dir).expanduser() / "_ab"
        setup_logging(level, out_dir / "ab.log")
        text = args.text or "Otto pianeti orbitano il Sole a velocità molto diverse."
        report = abtest.run_ab(out_dir, cfg, text=text, kinds=tuple(args.kinds))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "topics":
        from . import auto as auto_mod
        path = auto_mod._queue_path(cfg)
        if args.add:
            q = auto_mod.load_queue(path)
            q += [t.strip() for t in args.add if t.strip()]
            auto_mod.save_queue(path, q)
        if args.refill:
            auto_mod.next_topics(cfg, 0, consume=True)     # refill to threshold without popping
        q = auto_mod.load_queue(path)
        print(f"# {path}  ({len(q)} topics)")
        for t in q:
            print(t)
        return 0

    if args.cmd == "auto":
        from . import auto as auto_mod
        setup_logging(level, Path(cfg.paths.projects_dir).expanduser() / "_auto" / "auto.log")
        report = auto_mod.run_daily(cfg, count=args.count, dry_run=args.dry_run,
                                    publish=not args.no_publish, config_path=args.config)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.cmd in ("connect", "accounts", "disconnect"):
        # These are the commands a user runs while credentials are still half-configured, so a clean
        # one-line message beats a traceback. Same convention as the pipeline stages below.
        try:
            return _cmd_social(args, cfg)
        except Exception as e:  # noqa: BLE001
            log.error("%s", e)
            if args.verbose:
                raise
            return 1

    if args.cmd == "worker":
        from . import worker as worker_mod
        server = args.server or os.getenv("AVP_CONTROL_URL")
        token = args.token or os.getenv("AVP_CONTROL_TOKEN", "")
        if not server:
            log.error("worker: no control server (use --server or set AVP_CONTROL_URL)")
            return 1
        setup_logging(level, Path(cfg.paths.projects_dir).expanduser() / "_auto" / "worker.log")
        worker_mod.run_worker(cfg, server, token, once=args.once, poll=args.poll,
                              name=args.name, config_path=args.config)
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
            publish_mod.stage_publish(project, cfg, go=args.go, platforms=args.platforms, when=args.at)
            print(f"📤 {project.root / 'publish_plan.json'}")
        elif args.cmd == "build":
            out = pipeline.build(project, cfg, force=args.force,
                                 config_path=args.config, verbose=args.verbose)
            print(f"✅ {out}")
        elif args.cmd == "run":
            stages.stage_script(project, cfg, args.topic)
            out = pipeline.build(project, cfg, force=args.force,
                                 config_path=args.config, verbose=args.verbose)
            print(f"✅ {out}")
    except Exception as e:  # noqa: BLE001 — present a clean message, full trace in -v
        log.error("%s", e)
        if args.verbose:
            raise
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
