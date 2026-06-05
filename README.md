# AUT Video Pipeline (`avp`)

A **local, open-source engine** for faceless short-form videos (astronomy/space niche),
built to run entirely on an Apple-Silicon Mac. No paid SaaS, no cloud, no API bills.
Every default model is **commercial-license-clean**.

```
topic ──▶ script ──▶ [you review] ──▶ voice (TTS) ──▶ footage ──▶ captions ──▶ assemble ──▶ 9:16 .mp4
         (LLM)        script.md           Kokoro/         NASA       karaoke      ffmpeg
                                          Chatterbox      (public     (ASS)
                                                          domain)
```

## Why it's built this way

- **Python engine, thin UI later.** The local AI tools (Kokoro, Chatterbox, Parakeet)
  are Python-only, so the core is a clean Python CLI. A desktop (Electron) shell that
  *calls* this CLI is the next increment — built on an engine that already works.
- **Stage-based & resumable.** Each video is a folder under `projects/<slug>/` with a
  `manifest.json` tracking stage status, the models used, the **license attributions**
  owed, and whether **AI disclosure** is required at publish time. Re-running skips
  completed stages.
- **Human-in-the-loop.** After the script is generated it **stops** so you can fact-check
  and rewrite (`script.md`). This is what keeps the channel original — and monetizable
  under YouTube's "inauthentic content" policy.
- **Provider abstraction.** TTS, STT and the LLM sit behind interfaces; swapping engines
  is a config change.

## Requirements (already present on this Mac ✓)

`brew`, `uv`, `python3.11`, `ollama` (≥0.19), `node`. The setup script installs the rest
(`ffmpeg`, `espeak-ng`, Python deps, and the `qwen3:14b` model).

## Install

```bash
./setup.sh            # ffmpeg + .venv (py3.11) + AI deps + qwen3:14b + config.yaml
source .venv/bin/activate
```

## Use

```bash
# 1) generate a script, then STOP for review
avp new saturn-rings --topic "Why Saturn's rings are disappearing"

# 2) open projects/saturn-rings/script.md  — check the facts, tighten the hook

# 3) build everything after the script
avp build saturn-rings
#    -> projects/saturn-rings/saturn-rings.mp4

avp list                      # project + stage status
avp run smoke --topic "..."   # full run with no review stop (smoke test)
```

Per-stage commands let you iterate: `avp voice <slug>`, `avp footage <slug>`,
`avp captions <slug>`, `avp assemble <slug>`, `avp script <slug>`.

### The two voices

`tts.engine: both` synthesizes the script with **Kokoro** *and* **Chatterbox** and renders
**one video per voice** — `projects/<slug>/<slug>.kokoro.mp4` and `…chatterbox.mp4`, each
with its own WhisperX-aligned captions — so you choose per video. `<slug>.mp4` mirrors
`tts.primary`. Set a `chatterbox_ref` wav to clone a voice (you must own the rights to it).

### Funnel & metadata

With `funnel.enabled`, every video gets a spoken + on-screen **endcard** promoting your app
(`app_name` / `tagline` / `url` / `handle` / `cta_line`) — this drives installs. The
`metadata` stage writes `metadata.json` + `metadata.md` with per-platform **titles,
descriptions and hashtags** (app link included), plus the AI-disclosure flag. Aim for
`script.target_seconds: 75` so the spoken track clears the 60s TikTok minimum.

## Licenses (all defaults are commercial-safe)

| Stage | Engine | License |
|------|--------|---------|
| Script | Qwen3 (Ollama) | Apache-2.0 |
| Voice | Kokoro / Chatterbox | Apache-2.0 / MIT |
| Captions | Parakeet (opt) / WhisperX (opt) | CC-BY-4.0¹ / BSD |
| Footage | NASA image library | Public Domain² |
| Assembly | FFmpeg | LGPL |

¹ Parakeet weights are CC-BY-4.0 → **attribute NVIDIA** if you enable it.
² NASA media is public domain; the build records an attribution per asset anyway.
**Avoid** (non-commercial, not wired in): XTTS-v2, F5-TTS, Fish Speech for TTS; FLUX
\[dev]/\[Kontext-dev] for images; generic ESA website footage. Hubble/Webb/ESO are CC BY
(commercial OK *with on-screen credit*) — wire them in later as a footage source.

## Footage & attribution

`footage` first uses any file you drop at `projects/<slug>/footage/NN.jpg` (NN = segment
number); otherwise it searches NASA's public-domain library by the segment keywords and
downloads the best still. Each download is logged in the manifest's `attributions`.

## Publishing (not automated yet)

Per the toolchain research: **YouTube** auto-upload is easy (cheap API quota since Dec
2025). **TikTok/Instagram** require app audits/business-account review, so start
semi-manual there and add **Postiz** (AGPL, self-host) once the channel is rolling.

## Known first-run checks (honest notes)

These are isolated and commented in the code; confirm against installed versions:
- **Kokoro** (`tts.py`): `KPipeline` yield shape / voice names per the installed `kokoro`.
- **Chatterbox** (`tts.py`): `ChatterboxTTS.from_pretrained(device="mps")` may need a
  float32 workaround on MPS; outputs carry an (imperceptible, legal) watermark.
- **Parakeet** (`stt.py`): CLI flags / JSON schema. Default `stt.engine: even` needs no
  install; switch to `whisperx` (accurate) or `parakeet` (fast) once installed.
- **ffmpeg `zoompan`** (`ffmpeg.py`): Ken Burns params are conservative; tune to taste.
- **NASA asset size**: we pick the largest still in each item; verify resolution for 9:16.

## Layout

```
src/avp/
  cli.py         # commands
  pipeline.py    # orchestrator (build = voice→footage→captions→assemble)
  stages.py      # the 5 stages
  llm.py         # Ollama script generation
  tts.py         # Kokoro + Chatterbox providers
  stt.py         # word timing: even / whisperx / parakeet
  footage.py     # NASA search + download + attribution
  captions.py    # ASS karaoke writer
  ffmpeg.py      # render primitives
  config.py manifest.py models.py log.py
```

## Desktop UI

An accessible Electron control panel lives in `electron/` (Projects · New · Script review ·
Build with live log · Preview · Publish). It wraps this CLI — no separate backend.

```bash
cd electron && npm install && npm start
```

Or just **double-click `AUT Video Pipeline.app`** in the project root — a native launcher
with a custom icon (drag it to your Dock or Applications). Unsigned, so if macOS warns on
the very first run, right-click → Open once.

Built to **WCAG 2.1 AA / Legge Stanca (L. 4/2004)**: semantic landmarks, skip-link, keyboard-
operable ARIA tabs, `aria-live` build log, visible focus, ≥4.5:1 contrast, reduced-motion.
The renderer degrades to mock data in a plain browser for design preview.

## Publishing

`avp publish <slug>` builds a per-platform plan (caption/hashtags from `metadata.json` + the
video) and writes `publish_plan.json` — a **dry run** by default. With a running
[Postiz](https://github.com/gitroomhq/postiz-app) (open-source; TikTok/IG/YouTube + more),
set `publish.postiz_url` / `postiz_token` / `integrations` and run `avp publish <slug> --go`.
Real public posting still requires each platform's app approval (TikTok audit, Meta review).

## Roadmap

1. First validated video (this milestone).
2. Electron desktop shell calling this CLI (reuses the original `Social_Video_AI` idea).
3. Hubble/Webb/ESO footage sources (with CC-BY on-screen credit) + video clips.
4. Postiz integration for scheduling; YouTube auto-publish first.
