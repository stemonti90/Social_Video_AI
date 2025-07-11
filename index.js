"use strict";
/**
 * AI Social‑First Video Pipeline – fully original assets, multi‑platform edition
 * Version: 2025‑07‑10
 *
 * ✨ Funzioni chiave
 * ──────────────────────────────────────────────────────────────────────────
 * • Immagini 100 % originali → Stable Diffusion XL 1.0 + Real‑ESRGAN 4×, seed casuale per ogni run, dedup via hash + CLIP similarity.
 * • Musica  100 % originale → Meta MusicGen (CLI) con seed casuale, dedup via hash.
 * • Registri JSON per prevenire il ri‑utilizzo di asset in progetti futuri.
 * • Pipeline video completa (script → immagini → audio → video → mix) con esportazione dedicata per:
 *     – TikTok (1080×1920 / 30 fps)
 *     – Instagram Reels (1080×1920 / 30 fps)
 *     – Facebook Feed 4∶5 (1080×1350 / 30 fps) + Stories 16∶9 opz.
 *     – YouTube Shorts (1080×1920 / 60 fps)
 *   Ogni piattaforma ottiene: video MP4, cover PNG, metadati JSON (+ 16×9 per FB).
 * • Output audio → loudnorm EBU R128 + ducking sidechain.
 * • Hook testuale dinamico sui primi N frame (configurabile da ENV).
 * • Registri asset in ./.asset_registry (images.json, music.json).
 *
 * Nuove variabili ENV obbligatorie:
 *   MUSICGEN_EXECUTABLE   – path assoluto al binario musicgen‑cli
 *   MUSICGEN_MODEL        – modello MusicGen (es. facebook/musicgen‑melody)
 *
 * Dipendenze npm:
 *   execa pino p-limit compromise zod dotenv nanoid axios form-data node-fetch@3
 * Tool esterni: ffmpeg ≥ 6, Stable Diffusion XL (CLI), Real‑ESRGAN CLI, CLIP‑eval CLI, MusicGen CLI
 */

// ────────────────────────────────────────────────────────────────────────────
// 📦 Imports & setup
// ────────────────────────────────────────────────────────────────────────────
const execa    = require("execa");
const pino     = require("pino");
const pLimit   = require("p-limit");
const nlp      = require("compromise");
const { z }    = require("zod");
const dotenv   = require("dotenv");
const fs       = require("fs");
const fsP      = require("fs/promises");
const path     = require("path");
const crypto   = require("crypto");
const fetch    = require("node-fetch");
const { nanoid } = require("nanoid");
const axios    = require("axios");
const FormData = require("form-data");

dotenv.config();

// ────────────────────────────────────────────────────────────────────────────
// 🔧 ENV validation
// ────────────────────────────────────────────────────────────────────────────
const EnvSchema = z.object({
  OLLAMA_MODEL: z.string(),
  STABLE_DIFFUSION_EXECUTABLE: z.string(),
  STABLE_DIFFUSION_MODEL: z.string(),
  REAL_ESRGAN_EXECUTABLE: z.string(),
  CLIP_SIM_EXECUTABLE: z.string(),
  PIPER_EXECUTABLE: z.string(),
  PIPER_VOICE_MODEL: z.string(),
  FFMPEG_FONT_PATH: z.string(),
  FFMPEG_FONT_NAME: z.string(),
  VIDEO_CHUNKS: z.string().transform(Number),
  MUSICGEN_EXECUTABLE: z.string(),
  MUSICGEN_MODEL: z.string(),
  MAX_CONCURRENCY_GPU: z.string().optional(),
  MAX_CONCURRENCY_CPU: z.string().optional(),
  // hook overlay facoltativi
  HOOK_TEXT: z.string().optional(),
  HOOK_FONT_SIZE: z.string().optional(),
  HOOK_DURATION_FRAMES: z.string().optional(),
  // Facebook 16×9
  EXPORT_STORIES: z.enum(["true","false"]).optional(),
  // TikTok autopublish
  TIKTOK_API_TOKEN: z.string().optional(),
  TIKTOK_SCHEDULE_ISO: z.string().optional(),
}).passthrough();
try { EnvSchema.parse(process.env); } catch (e) { console.error("❌ .env non valido", e); process.exit(1); }

const config = require("./config");
const log = pino({ level: process.env.LOG_LEVEL || "info" });
const limitGPU = pLimit(Number(process.env.MAX_CONCURRENCY_GPU || 1));
const limitCPU = pLimit(Number(process.env.MAX_CONCURRENCY_CPU || 3));

// ────────────────────────────────────────────────────────────────────────────
// 📂 Registri asset (originalità garantita)
// ────────────────────────────────────────────────────────────────────────────
const REG_DIR = path.join(__dirname, ".asset_registry");
const IMG_REG = path.join(REG_DIR, "images.json");
const MUS_REG = path.join(REG_DIR, "music.json");
async function loadRegistry(file) { try { return new Set(JSON.parse(await fsP.readFile(file, "utf8"))); } catch { return new Set(); } }
async function saveRegistry(set, file) { await fsP.mkdir(REG_DIR, { recursive: true }); await fsP.writeFile(file, JSON.stringify([...set])); }

// ────────────────────────────────────────────────────────────────────────────
// 🛠️ Helper functions
// ────────────────────────────────────────────────────────────────────────────
const sha256 = (data) => crypto.createHash("sha256").update(data).digest("hex");
async function safeExec(cmd, tag = "", options = {}) {
  log.debug({ tag, cmd });
  const { stdout, stderr, exitCode } = await execa.command(cmd, { all: true, ...options });
  if (exitCode !== 0) throw new Error(`${tag} failed: ${stderr}`);
  return stdout;
}
async function exists(p) { try { await fsP.access(p); return true; } catch { return false; } }

// ────────────────────────────────────────────────────────────────────────────
// ✍️ Script generation (Ollama)
// ────────────────────────────────────────────────────────────────────────────
async function runOllama(prompt) {
  const cmd = `ollama run ${process.env.OLLAMA_MODEL} ${JSON.stringify(prompt)}`;
  return safeExec(cmd, "ollama", { maxBuffer: 20 * 1024 * 1024 });
}

async function generateScript() {
  log.info("16 % – Genero script iniziale");
  return runOllama(config.OLLAMA_PROMPT_INITIAL);
}
async function validateScript(script) {
  log.info("33 % – Validazione script");
  return runOllama(config.OLLAMA_PROMPT_VALIDATE(script));
}

// ────────────────────────────────────────────────────────────────────────────
// ✂️ Split testo per chunk & sottotitoli
// ────────────────────────────────────────────────────────────────────────────
function splitScript(text) {
  const sentences = nlp(text).sentences().out("array");
  // wrap ≤ 80 char
  const wrapped = sentences.flatMap((s) => {
    if (s.length <= 80) return [s];
    const words = s.split(" ");
    const arr = [];
    let buf = "";
    for (const w of words) {
      if ((buf + " " + w).trim().length > 80) { arr.push(buf.trim()); buf = w; }
      else buf += ` ${w}`;
    }
    if (buf.trim()) arr.push(buf.trim());
    return arr;
  });
  const size = Math.ceil(wrapped.length / config.VIDEO_CHUNKS);
  return Array.from({ length: config.VIDEO_CHUNKS }, (_, i) => wrapped.slice(i * size, (i + 1) * size).join(" \n"));
}

// ────────────────────────────────────────────────────────────────────────────
// 🎨 Image generation – SDXL + ESRGAN + CLIP + dedup
// ────────────────────────────────────────────────────────────────────────────
async function generateImage(prompt, idx) {
  const seed = crypto.randomInt(1, 2 ** 30);
  const raw   = path.join(config.IMAGE_FOLDER, `raw_${seed}.png`);
  const final = path.join(config.IMAGE_FOLDER, `image${idx}.png`);
  await safeExec(`${process.env.STABLE_DIFFUSION_EXECUTABLE} -m "${process.env.STABLE_DIFFUSION_MODEL}" -p "${prompt}" -o "${raw}" --height 1024 --width 1024 --seed ${seed} -s 30`, `sdxl#${idx}`);
  const up = raw.replace("raw_", "up_4x_");
  await safeExec(`${process.env.REAL_ESRGAN_EXECUTABLE} -i "${raw}" -o "${up}" -n realesrgan-x4plus-anime`, `esrgan#${idx}`);
  await safeExec(`ffmpeg -y -i "${up}" -vf "crop=1080:1920,scale=1080:1920" "${final}"`, `crop#${idx}`);

  const hash = sha256(await fsP.readFile(final));
  const reg = await loadRegistry(IMG_REG);
  if (reg.has(hash)) {
    log.warn(`Duplicate image hash → regenerating`);
    return generateImage(prompt + " unique", idx);
  }
  reg.add(hash); await saveRegistry(reg, IMG_REG);

  const sim = parseFloat(await safeExec(`${process.env.CLIP_SIM_EXECUTABLE} "${final}" "${prompt}"`, `clip#${idx}`));
  if (sim < 0.25) {
    log.warn(`Low CLIP similarity (${sim}) → regenerating`);
    return generateImage(prompt + ", vivid", idx);
  }
  return final;
}
async function generateImages(prompts) {
  await fsP.mkdir(config.IMAGE_FOLDER, { recursive: true });
  return Promise.all(prompts.map((p, i) => limitGPU(() => generateImage(p, i))));
}

// ────────────────────────────────────────────────────────────────────────────
// 🎙️ Voice generation (Piper) + cache
// ────────────────────────────────────────────────────────────────────────────
async function generateVoice(text, outPath) {
  const h = sha256(text + process.env.PIPER_VOICE_MODEL);
  const cached = outPath.replace(/\.wav$/, `_${h}.wav`);
  if (await exists(cached)) return cached;
  const tmp = cached + ".txt";
  await fsP.writeFile(tmp, text);
  await safeExec(`${process.env.PIPER_EXECUTABLE} --model ${process.env.PIPER_VOICE_MODEL} --output_file "${cached}" --text_file "${tmp}"`, "piper" );
  await fsP.unlink(tmp);
  return cached;
}
async function getDuration(file) {
  return parseFloat(await safeExec(`ffprobe -v error -show_entries format=duration -of csv=p=0 "${file}"`, "ffprobe"));
}

// ────────────────────────────────────────────────────────────────────────────
// 🖼️ Static clip (intro / outro)
// ────────────────────────────────────────────────────────────────────────────
async function generateStaticClip(img, txt, dur, out) {
  const esc = txt.replace(/'/g, "''").replace(/:/g, "\\:");
  await safeExec(`ffmpeg -y -loop 1 -i "${img}" -t ${dur} -vf "scale=1080:1920,drawtext=text='${esc}':fontfile='${process.env.FFMPEG_FONT_PATH}':fontcolor=white:fontsize=${config.INTRO_OUTRO_FONT_SIZE}:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.5:boxborderw=10" -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart "${out}"`, "static" );
}

// ────────────────────────────────────────────────────────────────────────────
// 🎬 Main content with Ken Burns + hook overlay
// ────────────────────────────────────────────────────────────────────────────
function hookFilter() {
  if (!process.env.HOOK_TEXT) return "";
  const t = process.env.HOOK_TEXT.replace(/'/g, "''").replace(/:/g, "\\:");
  const fs = process.env.HOOK_FONT_SIZE || 64;
  const f  = process.env.HOOK_DURATION_FRAMES || 45;
  return `[v0]drawtext=text='${t}':fontfile='${process.env.FFMPEG_FONT_PATH}':fontcolor=white:fontsize=${fs}:x=(w-text_w)/2:y=H/8:enable='lte(n,${f})'[v1];[v1]`;
}

async function generateMainVideoContent(chunks) {
  const tmp = config.TEMP_FOLDER;
  await fsP.mkdir(tmp, { recursive: true });
  const parts = [];

  for (let i = 0; i < chunks.length; i++) {
    const txt = chunks[i];
    const img = path.join(config.IMAGE_FOLDER, `image${i}.png`);
    const audio = await limitCPU(() => generateVoice(txt, path.join(tmp, `v${i}.wav`)));
    const dur = await getDuration(audio);

    // Subtitle ASS
    const sub = path.join(tmp, `sub${i}.ass`);
    const lines = txt.split("\n").flatMap((l) => {
      if (l.length <= 42) return [l];
      const words = l.split(" ");
      const arr = [];
      let b = "";
      for (const w of words) {
        if ((b + " " + w).trim().length > 42) { arr.push(b.trim()); b = w; }
        else b += ` ${w}`;
      }
      if (b.trim()) arr.push(b.trim());
      return arr;
    });
    await fsP.writeFile(sub, lines.join("\n"));

    const frames = Math.ceil(dur * 30);
    const zoom = `min(1.5,1+(n/${frames})*0.5)`;
    const xPan = `iw/2-(iw/zoom/2)+(n/${frames})*50`;
    const yPan = `ih/2-(ih/zoom/2)`;
    const out = path.join(tmp, `part${i}.mp4`);

    const cmd = `ffmpeg -y -i "${img}" -i "${audio}" -filter_complex "[0:v]zoompan=z='${zoom}':x='${xPan}':y='${yPan}':d=${frames}:s=1080x1920[v0];${hookFilter()}[v0]crop=1080:1920[v];[v]subtitles='${sub}:force_style=FontName=${process.env.FFMPEG_FONT_NAME},FontSize=28,Alignment=2,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=1,Shadow=0'" -map "[v]" -map 1:a -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart -c:a aac -b:a 192k "${out}"`;
    await safeExec(cmd, `part#${i}`);
    parts.push(out);
  }

  const list = path.join(tmp, "main_list.txt");
  await fsP.writeFile(list, parts.map((p) => `file '${p.replace(/\\/g, '/')}'`).join("\n"));
  const main = path.join(tmp, "main.mp4");
  await safeExec(`ffmpeg -y -f concat -safe 0 -i "${list}" -c copy "${main}"`, "concatMain");
  return main;
}

// ────────────────────────────────────────────────────────────────────────────
// 🎵 Music generation (MusicGen) + mix
// ────────────────────────────────────────────────────────────────────────────
async function generateOriginalMusic(duration, prompt) {
  const seed = crypto.randomInt(1, 2 ** 30);
  const out = path.join(config.TEMP_FOLDER, `music_${seed}.wav`);
  const promptText = `${prompt.slice(0, 80)} instrumental background`;
  await safeExec(`${process.env.MUSICGEN_EXECUTABLE} --model ${process.env.MUSICGEN_MODEL} --prompt "${promptText}" --duration ${duration} --output "${out}" --seed ${seed}`, "musicgen");
  const hash = sha256(await fsP.readFile(out));
  const reg = await loadRegistry(MUS_REG);
  if (reg.has(hash)) {
    log.warn("Duplicate music hash → regenerating");
    return generateOriginalMusic(duration, prompt + " unique");
  }
  reg.add(hash); await saveRegistry(reg, MUS_REG);
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// 🔊 Final mix (intro + main + outro → concat + loudnorm + ducking)
// ────────────────────────────────────────────────────────────────────────────
async function finalizeVideo(intro, main, outro, scriptText) {
  const tmp = config.TEMP_FOLDER;
  const list = path.join(tmp, "cat.txt");
  await fsP.writeFile(list, [intro, main, outro].filter(Boolean).map((p) => `file '${p.replace(/\\/g, '/')}'`).join("\n"));
  const concat = path.join(tmp, "concat.mp4");
  await safeExec(`ffmpeg -y -f concat -safe 0 -i "${list}" -c copy "${concat}"`, "concat");

  const dur = await getDuration(concat);
  const bgMusic = await generateOriginalMusic(Math.ceil(dur), scriptText);
  const out = config.OUTPUT_PATH;
  await safeExec(`ffmpeg -y -i "${concat}" -i "${bgMusic}" -filter_complex "[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[a0];[1:a]volume=0.2[bg];[a0][bg]sidechaincompress=threshold=-25dB:ratio=8:attack=20:release=300[aout]" -map 0:v -map "[aout]" -c:v copy -movflags +faststart -shortest "${out}"`, "mix");
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// 📦 Platform presets & packaging
// ────────────────────────────────────────────────────────────────────────────
const PLATFORMS = [
  {
    name: "tiktok", suffix: "_tiktok", width:1080, height:1920, fps:30, maxDur:60,
    captionTpl:(t)=>`${t} 🚀`, hashtags:["#foryou","#fyp","#ai"],
    schedule: process.env.TIKTOK_SCHEDULE_ISO || null,
    publish: !!process.env.TIKTOK_API_TOKEN,
  },
  {
    name: "instagram", suffix:"_insta", width:1080, height:1920, fps:30, maxDur:90,
    captionTpl:(t)=>`${t} 😎`, hashtags:["#reels","#explore","#ai"],
  },
  {
    name:"facebook", suffix:"_fb", width:1080, height:1350, fps:30, maxDur:90,
    captionTpl:(t)=>t, hashtags:["#video","#watch","#ai"],
    exportStories16x9: process.env.EXPORT_STORIES === "true",
  },
  {
    name:"youtube", suffix:"_yt", width:1080, height:1920, fps:60, maxDur:60,
    captionTpl:(t)=>`${t} #Shorts`, hashtags:["#Shorts","#ai"],
  },
];

async function packageForPlatform(master, plat, title) {
  const { suffix, width, height, fps, name } = plat;
  const outVid = config.OUTPUT_PATH.replace(/\.mp4$/, `${suffix}.mp4`);
  const vf = (width === 1080 && height === 1920)
    ? `scale=1080:1920,fps=${fps}`
    : `scale=${width}:${height},crop=${width}:${height},fps=${fps}`;
  await safeExec(`ffmpeg -y -i "${master}" -vf "${vf}" -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart -c:a copy "${outVid}"`, `${name}-resize`);

  const cover = outVid.replace(/\.mp4$/, "_cover.png");
  await safeExec(`ffmpeg -y -ss 00:00:01 -i "${outVid}" -vframes 1 -q:v 2 "${cover}"`, `${name}-cover`);

  let stories = null;
  if (plat.exportStories16x9) {
    stories = outVid.replace(/\.mp4$/, "_16x9.mp4");
    await safeExec(`ffmpeg -y -i "${outVid}" -vf "crop=ih*(16/9):ih" -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart -c:a copy "${stories}"`, "fb-stories" );
  }

  const meta = {
    platform: name,
    video: path.basename(outVid),
    cover: path.basename(cover),
    stories16x9: stories ? path.basename(stories) : undefined,
    width, height, fps,
    caption: plat.captionTpl(title),
    hashtags: plat.hashtags,
    schedule: plat.schedule || null,
  };
  const jsonPath = outVid.replace(/\.mp4$/, ".json");
  await fsP.writeFile(jsonPath, JSON.stringify(meta, null, 2));

  // autopublish TikTok
  if (plat.publish) {
    try {
      const form = new FormData();
      form.append("video", fs.createReadStream(outVid));
      if (plat.schedule) form.append("schedule_time", plat.schedule);
      await axios.post("https://open.tiktokapis.com/v2/video/upload", form, { headers: { Authorization: `Bearer ${process.env.TIKTOK_API_TOKEN}`, ...form.getHeaders() } });
      log.info("TikTok upload OK");
    } catch (e) { log.warn("TikTok upload error", e.message); }
  }
}

// ────────────────────────────────────────────────────────────────────────────
// 🚀 Main pipeline flow
// ────────────────────────────────────────────────────────────────────────────
(async () => {
  try {
    log.info("🚀 Pipeline start");
    await fsP.mkdir(config.TEMP_FOLDER, { recursive: true });

    // 1️⃣ Script
    const rawScript = await generateScript();
    const script = await validateScript(rawScript);
    const chunks = splitScript(script);

    // 2️⃣ Images
    const imgPrompts = await Promise.all(chunks.map((c) => runOllama(config.OLLAMA_PROMPT_IMAGE_GEN(c))));
    await generateImages(imgPrompts);

    // 3️⃣ Intro & Outro
    const intro = path.join(config.TEMP_FOLDER, "intro.mp4");
    await generateStaticClip(config.INTRO_IMAGE_PATH, config.INTRO_TEXT, config.INTRO_DURATION_SECONDS, intro);
    const outro = path.join(config.TEMP_FOLDER, "outro.mp4");
    await generateStaticClip(config.OUTRO_IMAGE_PATH, config.OUTRO_TEXT, config.OUTRO_DURATION_SECONDS, outro);

    // 4️⃣ Main content
    const main = await generateMainVideoContent(chunks);

    // 5️⃣ Mix + original music
    const master = await finalizeVideo(intro, main, outro, script);

    // 6️⃣ Packaging per piattaforme
    const title = "Il Futuro dell'AI"; // oppure estratto dallo script
    for (const plat of PLATFORMS) await packageForPlatform(master, plat, title);

    log.info("✅ Output completo per tutte le piattaforme");
  } catch (e) {
    log.error("❌ Errore pipeline", e);
  }
})();
