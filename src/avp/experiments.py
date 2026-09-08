"""Structured A/B tests: one variable at a time, assigned per video, measured in the report.

The growth loop learns only if a change can be attributed. An experiment names ONE variable and its
variants; every new video is assigned a variant (alternating, so the arms stay balanced), the
assignment is written into the project's manifest and into the publish record, and the weekly report
groups the numbers by variant. Variables the pipeline can actually act on:

  tiktok_offset     minutes after the Instagram slot at which the TikTok copy is scheduled (Upload-Post)
  target_seconds    the length the script is written for
  ig_hashtags_max   how many hashtags the Instagram caption carries

State lives in projects/_auto/experiments.json; `avp experiment start|stop|show` manages it.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

VARIABLES = {"tiktok_offset": int, "target_seconds": int, "ig_hashtags_max": int}


def _path(cfg) -> Path:
    return Path(cfg.paths.projects_dir).expanduser() / "_auto" / "experiments.json"


def _load(cfg) -> dict:
    p = _path(cfg)
    if not p.exists():
        return {"active": None, "history": []}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {"active": None, "history": []}


def _save(cfg, state: dict) -> None:
    p = _path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=1, ensure_ascii=False))


def active(cfg) -> dict | None:
    return _load(cfg).get("active")


def start(cfg, name: str, variable: str, variants: list[str]) -> dict:
    if variable not in VARIABLES:
        raise ValueError(f"unknown variable {variable!r}; choose from {sorted(VARIABLES)}")
    if len(variants) < 2:
        raise ValueError("an experiment needs at least two variants")
    cast = VARIABLES[variable]
    exp = {"name": name, "variable": variable, "variants": [cast(v) for v in variants],
           "started": date.today().isoformat(), "assigned": {}}
    state = _load(cfg)
    if state.get("active"):
        state["history"].append(dict(state["active"], stopped=date.today().isoformat()))
    state["active"] = exp
    _save(cfg, state)
    log.info("Experiment %r started: %s ∈ %s", name, variable, exp["variants"])
    return exp


def stop(cfg) -> dict | None:
    state = _load(cfg)
    exp = state.get("active")
    if exp:
        state["history"].append(dict(exp, stopped=date.today().isoformat()))
        state["active"] = None
        _save(cfg, state)
    return exp


def assign(cfg, slug: str) -> dict | None:
    """The active experiment's variant for a new video — alternating so the arms stay balanced — or
    None when nothing is running. Idempotent for a slug already assigned."""
    state = _load(cfg)
    exp = state.get("active")
    if not exp:
        return None
    if slug in exp["assigned"]:
        variant = exp["assigned"][slug]
    else:
        variant = exp["variants"][len(exp["assigned"]) % len(exp["variants"])]
        exp["assigned"][slug] = variant
        _save(cfg, state)
    return {"name": exp["name"], "variable": exp["variable"], "variant": variant}


def value(project, variable: str, default):
    """The variant a project carries for `variable`, else `default`."""
    try:
        exp = project.manifest.data.get("experiment") or {}
    except Exception:  # noqa: BLE001
        return default
    if exp.get("variable") == variable and exp.get("variant") is not None:
        return VARIABLES.get(variable, str)(exp["variant"])
    return default


def describe(cfg) -> str:
    state = _load(cfg)
    exp = state.get("active")
    lines = []
    if exp:
        lines.append(f"attivo: {exp['name']} — {exp['variable']} ∈ {exp['variants']} (dal {exp['started']}, "
                     f"{len(exp['assigned'])} video assegnati)")
        for slug, v in exp["assigned"].items():
            lines.append(f"  {slug}: {v}")
    else:
        lines.append("nessun esperimento attivo")
    for h in state.get("history", [])[-3:]:
        lines.append(f"chiuso: {h['name']} — {h['variable']} ∈ {h['variants']} ({h['started']} → {h.get('stopped')})")
    return "\n".join(lines)
