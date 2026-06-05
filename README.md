# AUT Video Pipeline (`avp`)

**A local, open-source pipeline for faceless short-form videos** (astronomy / space niche),
running entirely on an Apple-Silicon Mac. No paid SaaS, no cloud, no API bills — every
default model is **commercial-license-clean**. Two goals: ad monetization, and driving
installs of the companion app **AstroStackerPro**.

It ships as both a **Python engine** (`avp` CLI) and an **accessible Electron desktop app**
that drives it — both working and validated end-to-end.

```
topic ─▶ script ─▶ [you review] ─▶ voice ─▶ footage ─▶ captions ─▶ assemble ─▶ 9:16 .mp4
        (Ollama)     script.md      Kokoro/   NASA +      karaoke     ffmpeg
                                    Chatterbox Wikimedia   (PNG/ASS)
                                               + fallback
```

Language **ITA / ENG** selectable per video. Footage **never goes black**:
NASA → Wikimedia Commons → generated cosmic backdrop.

## Highlights

- **All-local OSS stack**, Apple-Silicon (M-series) friendly (~24 GB RAM).
- **Stage-based & resumable** — each video is `projects/<slug>/` + a `manifest.json` tracking
  stage status, the models used, the **license attributions** owed, and whether **AI
  disclosure** is required at publish. Re-runs skip completed stages.
- **Human-in-the-loop** — generation stops after the script so you fact-check and tighten
  `script.md` (what keeps the channel original and monetizable).
- **Two TTS voices** (Kokoro + Chatterbox) — render one video per voice, choose per upload.
- **App funnel** — spoken + on-screen endcard, plus per-platform metadata.
- **Accessible desktop UI** (Electron) wrapping the CLI — no separate backend.
- **Hardened** — tolerant of brew-ffmpeg/ffprobe SIGSEGV, footage dedup, graceful
  fallbacks; **23 unit tests**.

## Requirements

`brew`, `uv`, `python3.11`, `ollama` (≥0.19), `node`. `setup.sh` installs the rest
(`ffmpeg`, `espeak-ng`, Python deps, and the `qwen3:14b` model).

## Install

```bash
./setup.sh                     # ffmpeg + .venv (py3.11) + AI deps + qwen3:14b + config.yaml
cd electron && npm install     # desktop UI deps
```

## Use — CLI

```bash
# 1) generate a script, then STOP for review
avp new saturn-rings --topic "Why Saturn's rings are disappearing"

# 2) open projects/saturn-rings/script.md — check the facts, tighten the hook

# 3) build everything after the script
avp build saturn-rings         # → projects/saturn-rings/saturn-rings.mp4

avp list                       # project + stage status
```

Per-stage commands let you iterate: `avp script|voice|footage|captions|assemble|metadata <slug>`.
`avp run <slug> --topic "…"` does a full run with no review stop (smoke test).

## Use — Desktop app

```bash
cd electron && npm start
```

…or double-click **AUT Video Pipeline.app** (drag it to /Applications or the Dock). The UI is
a numbered **stepper** — Projects · New · ① Review · ② Build (live log) · ③ Preview · ④ Publish —
with an always-visible footer (← Back / step / primary action / Next →) so the next action is
never hidden below long content. **Every setting** (language, voices, captions, platforms, app
funnel) is editable in the UI; you never have to touch config files. Unsigned, so on the very
first run macOS may warn → right-click → Open once.

> **Accessibility.** Built targeting **WCAG 2.1 AA / Legge Stanca (L. 4/2004)**: semantic
> landmarks, skip-link, keyboard-operable ARIA tabs (arrows/Home/End), `aria-live` build log,
> visible focus, ≥4.5:1 contrast, status conveyed by **shape + text, not color alone**, and
> reduced-motion support. Programmatic and runtime checks pass; a full **manual screen-reader
> audit (VoiceOver/NVDA) and an automated axe-core pass are still pending**, so conformance is
> reported as **partial** until those are done. The renderer degrades to mock data in a plain
> browser for design preview.

## Language & voices

`script.language: en|it` (Kokoro voice `af_heart` for EN / `if_sara` for IT; **Chatterbox is
EN-only and auto-skipped for IT**; footage keywords stay English for archive search).
`tts.engine: both` synthesizes with **Kokoro** *and* **Chatterbox** and renders one video per
voice — `<slug>.kokoro.mp4` + `<slug>.chatterbox.mp4`, each with its own captions — so you
choose per video. `<slug>.mp4` mirrors `tts.primary`. Set a `chatterbox_ref` wav to clone a
voice (you must own the rights to it). Captions default to **Parakeet (MLX)**; `whisperx` and
`even` are selectable.

## Footage (never black) & attribution

The `footage` stage fills each segment via a fallback chain so a clip is **never left black**:

1. any file you drop at `projects/<slug>/footage/NN.jpg` (NN = segment number), else
2. **NASA** public-domain library — relevance-scored, diagram-filtered, **deduped by id *and*
   title** (NASA stores the same photo under several ids), else
3. **Wikimedia Commons** — free licenses, incl. ESA / Hubble / ESO, else
4. a **generated cosmic backdrop** (Pillow nebula + starfield).

Every download is recorded in the manifest's `attributions`. Optional (toggle in
`config.video`): real NASA **video** clips with mid-clip seek to skip title cards, crossfades,
**EBU R128 loudness** (−14 LUFS), inter-segment micro-gaps, on-screen credits, auto music.

## Funnel & metadata

With `funnel.enabled`, every video gets a spoken + on-screen **endcard** promoting your app
(`app_name` / `tagline` / `url` / `handle` / `cta_line`). The `metadata` stage writes
`metadata.json` + `metadata.md` with per-platform **titles, descriptions and hashtags** (app
link included), plus the AI-disclosure flag. Aim for `script.target_seconds: 75` so the spoken
track clears the 60 s TikTok minimum.

## Publishing

`avp publish <slug>` builds a per-platform plan (caption/hashtags from `metadata.json` + the
video) and writes `publish_plan.json` — a **dry run** by default. With a running
[Postiz](https://github.com/gitroomhq/postiz-app) (open-source; TikTok/IG/YouTube + more), set
`publish.postiz_url` / `postiz_token` / `integrations` and run `avp publish <slug> --go`. Real
public posting still requires each platform's app approval (TikTok audit, Meta review). In the
desktop app, **"Generate plan (dry run)"** is the safe default; **"Publish now (live)"** is a
separate, confirmation-gated action.

## Licenses (all defaults are commercial-safe)

| Stage | Engine | License |
|------|--------|---------|
| Script | Qwen3 (Ollama) | Apache-2.0 |
| Voice | Kokoro / Chatterbox | Apache-2.0 / MIT |
| Captions | Parakeet (default) / WhisperX | CC-BY-4.0¹ / BSD |
| Footage | NASA / Wikimedia Commons | Public Domain² / free licenses |
| Assembly | FFmpeg | LGPL |

¹ Parakeet weights are CC-BY-4.0 → **attribute NVIDIA** when you enable it.
² NASA media is public domain; the build records an attribution per asset anyway.
**Avoid** (non-commercial, not wired in): XTTS-v2, F5-TTS, Fish Speech for TTS; FLUX
\[dev]/\[Kontext-dev] for images; generic ESA website footage. Hubble/Webb/ESO are CC-BY
(commercial OK *with on-screen credit*).

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

23 unit tests cover footage scoring / dedup / URL-encoding, LLM JSON extraction, caption
timing, WAV-duration (header-read, no ffprobe), `ffmpeg` retry-on-signal, and config
round-trips.

## Layout

```
src/avp/
  cli.py         # commands
  pipeline.py    # orchestrator (build = script→voice→footage→captions→assemble→metadata)
  stages.py      # the stages
  llm.py         # Ollama script + metadata generation
  tts.py         # Kokoro + Chatterbox providers
  stt.py         # word timing: even / whisperx / parakeet
  footage.py     # NASA + Wikimedia search/download + dedup + attribution
  captions.py    # karaoke caption PNGs, endcard, cosmic backdrop
  ffmpeg.py      # render primitives (SIGSEGV-hardened)
  config.py manifest.py models.py log.py
electron/        # main.js + preload.js + renderer/ (accessible UI) + server.js (browser preview)
tests/           # test_core.py
```

## Engineering notes (Apple-Silicon)

brew's **minimal ffmpeg** lacks libass (so captions are rendered as Pillow PNG overlays) and
can transiently **SIGSEGV** under load — WAV durations are therefore read straight from the
file header (no `ffprobe`), and `ffmpeg.run()` retries on signal death. Pin **torch 2.6.0 ↔
torchvision 0.21.0**. Always run the CLI with `PYTHONPATH=src`.

## Roadmap

- [x] Validated end-to-end video (Kokoro + Chatterbox).
- [x] Accessible Electron desktop app driving the CLI.
- [x] Footage fallback chain (no black frames) + ITA/ENG language switch.
- [ ] Hubble / Webb / ESO footage sources (CC-BY on-screen credit) + more video clips.
- [ ] Wire the real AstroStackerPro App Store URL into the funnel.
- [ ] Postiz live posting; YouTube auto-publish first.
- [ ] Full manual screen-reader + automated axe-core accessibility audit.
