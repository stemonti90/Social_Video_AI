# Music for AUT Video Pipeline — commercial-safe sources

Drop `.mp3` / `.wav` / `.m4a` files in this folder. The pipeline auto-picks a track during
`assemble` and **ducks it under the narration** (sidechain compression) with a gentle fade-out.

⚠️ This is a **monetized** channel → every track must be cleared for **commercial use** and,
ideally, **not registered with YouTube Content ID** (otherwise you'll have to dispute claims).

## Verified options (June 2026)

| Source | License | Commercial | Attribution | Notes |
|---|---|---|---|---|
| **Stable Audio Open** (local generation) | Stability AI Community | ✅ (< $1M ARR) | none | 100% **original** → can never get a Content-ID claim. Best for a monetized channel. Needs a one-time Hugging Face license accept + token (see below). |
| **Pixabay Music** | Pixabay Content License | ✅ | none | Free account to download. A few tracks are Content-ID registered → keep the license certificate to dispute. |
| **Mixkit** | Mixkit Free License | ✅ | none | Free to use inside videos; no standalone redistribution. |
| **Incompetech** (Kevin MacLeod) | CC-BY 4.0 | ✅ *with credit* | **required** | High quality, direct download — but **frequently Content-ID claimed** (disputable with the CC-BY license). |
| **YouTube Audio Library** | YouTube license | ✅ | varies | In YouTube Studio; many tracks are claim-safe; YouTube-centric. |

❌ **FreePD.com closed in 2025** — no longer a source.
❌ Avoid anything without downloadable license proof, and "free" pop/remix tracks (almost always Content-ID).

## Picking the mood
Astronomy/space content suits **ambient / cinematic / sci-fi / atmospheres**. Keep it sparse
and slow — it's a bed under the voice, not the focus.

## Crediting CC-BY tracks
If you use a CC-BY track (e.g. Incompetech), the required attribution goes in the video
description. Ask me to wire `video.music_credit` so the `metadata` stage adds it automatically.

## Local generation (Stable Audio Open)
To enable original generated music (recommended), do the one-time Hugging Face steps:
1. Open <https://huggingface.co/stabilityai/stable-audio-open-1.0> and click **Agree and access repository**.
2. Create a **Read** token at <https://huggingface.co/settings/tokens>.
3. Authenticate locally: `huggingface-cli login` (paste the token) — *do this yourself; tokens are never entered by the assistant*.
Then tell me, and I'll install `stable-audio-tools`, download the model, and wire `video.music_source: generate`.
