"use strict";
/**
 * AI‑powered TikTok Video Generator – single‑file version
 *
 * Integrazione dei miglioramenti suggeriti:
 *  • execa per esecuzione processi (stream, niente limiti buffer)
 *  • pino logging JSON / pretty in dev
 *  • p‑limit per concorrenza controllata
 *  • compromise NLP per suddividere il testo
 *  • zod + dotenv per validare le variabili d’ambiente
 *  • graceful shutdown & cleanup
 *
 * Dipendenze runtime (npm i --save): execa pino p-limit compromise zod dotenv
 * Dev‑helper (opz.): pino-pretty per output leggibile → `pino-pretty | node ai_video_pipeline.js`
 */

// ────────────────────────────────────────────────────────────────────────────
// 📦  Import
// ────────────────────────────────────────────────────────────────────────────
const execa = require("execa");
const pino = require("pino");
const pLimit = require("p-limit");
const nlp = require("compromise");
const { z } = require("zod");
const dotenv = require("dotenv");
const fs = require("fs");
const fsPromises = require("fs/promises");
const path = require("path");

// Carica e valida .env -------------------------------------------------------
dotenv.config();
const EnvSchema = z
  .object({
    OLLAMA_MODEL: z.string().min(1),
    STABLE_DIFFUSION_EXECUTABLE: z.string().min(1),
    STABLE_DIFFUSION_MODEL: z.string().min(1),
    PIPER_EXECUTABLE: z.string().min(1),
    PIPER_VOICE_MODEL: z.string().min(1),
    FFMPEG_FONT_PATH: z.string().min(1),
    FFMPEG_FONT_NAME: z.string().min(1),
    VIDEO_CHUNKS: z.string().min(1),
    MAX_CONCURRENCY: z.string().optional(),
    // ...aggiungi qui se usi altre env
  })
  .passthrough();

try {
  EnvSchema.parse(process.env);
} catch (err) {
  console.error("❌  Variabili d’ambiente mancanti o non valide:\n", err);
  process.exit(1);
}

// Include la config di progetto (costanti pure) -----------------------------
const config = require("./config");

// Logger --------------------------------------------------------------------
const log = pino({
  level: process.env.LOG_LEVEL || "info",
  transport: process.env.NODE_ENV === "development" && {
    target: "pino-pretty",
    options: { translateTime: "yyyy-mm-dd HH:MM:ss.l", colorize: true },
  },
});

// Concorrenza globale --------------------------------------------------------
const limit = pLimit(parseInt(process.env.MAX_CONCURRENCY || "3", 10));

// ────────────────────────────────────────────────────────────────────────────
// 🛠️  Utilities
// ────────────────────────────────────────────────────────────────────────────
/**
 * Esegue un comando shell in modo sicuro con execa, loggando in JSON.
 * @param {string} cmd  comando da eseguire
 * @param {string} tool identificativo dello strumento (per log)
 * @param {execa.Options} opts  opz.
 * @returns {Promise<string>} stdout
 */
async function safeExec(cmd, tool, opts = {}) {
  log.info({ tool, cmd }, "🚀 avvio");
  const subprocess = execa.command(cmd, {
    all: true,
    stripFinalNewline: true,
    ...opts,
  });

  subprocess.all?.on("data", (chunk) => {
    log.debug({ tool }, chunk.toString());
  });

  const { stdout, exitCode, stderr } = await subprocess;

  if (exitCode !== 0) {
    const errorMsg = `Errore ${tool} (exit ${exitCode}): ${stderr}`;
    log.error(errorMsg);
    throw new Error(errorMsg);
  }
  return stdout;
}

// Wrapper Ollama ------------------------------------------------------------
async function runOllama(prompt) {
  const command = `ollama run ${config.OLLAMA_MODEL} ${JSON.stringify(prompt)}`;
  return safeExec(command, "Ollama", { maxBuffer: 1024 * 1024 * 10 });
}

// ────────────────────────────────────────────────────────────────────────────
// 📜  Pipeline Steps
// ────────────────────────────────────────────────────────────────────────────
async function generateScript() {
  log.info("[16%] ✍️  Generazione script iniziale...");
  return runOllama(config.OLLAMA_PROMPT_INITIAL);
}

async function validateScript(script) {
  log.info("[33%] 📚 Validazione grammatica e veridicità...");
  return runOllama(config.OLLAMA_PROMPT_VALIDATE(script));
}

function splitScript(text) {
  log.info("[50%] ✂️  Suddivisione testo...");
  const sentences = nlp(text).sentences().out("array");
  const total = sentences.length;
  const chunkSize = Math.ceil(total / config.VIDEO_CHUNKS);
  return Array.from({ length: config.VIDEO_CHUNKS }, (_, i) =>
    sentences.slice(i * chunkSize, (i + 1) * chunkSize).join(" ").trim()
  );
}

async function getAudioDuration(audioFilePath) {
  const cmd = `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${audioFilePath}"`;
  const out = await safeExec(cmd, "FFprobe");
  return parseFloat(out.trim());
}

async function generateImagePrompts(chunks) {
  log.info("[58%] 🎨  Generazione prompt immagini...");
  const results = await Promise.all(
    chunks.map((ch) => limit(() => runOllama(config.OLLAMA_PROMPT_IMAGE_GEN(ch))))
  );
  return results.map((p) => p.replace(/^"|"$/g, ""));
}

async function generateImages(prompts) {
  log.info("[66%] 🖼  Generazione immagini...");
  await fsPromises.mkdir(config.IMAGE_FOLDER, { recursive: true });
  await Promise.all(
    prompts.map((prompt, i) =>
      limit(async () => {
        const outPath = path.join(config.IMAGE_FOLDER, `image${i}.png`);
        log.info(`   -> Img ${i + 1}/${prompts.length}`);
        const cmd = `${config.STABLE_DIFFUSION_EXECUTABLE} -m "${config.STABLE_DIFFUSION_MODEL}" -p "${prompt}" -o "${outPath}" --height 1920 --width 1080 -s 25`;
        await safeExec(cmd, `StableDiffusion#${i + 1}`);
      })
    )
  );
}

async function generateVoice(text, outPath) {
  log.info("[83%] 🎙  Generazione voce...");
  const dir = path.dirname(outPath);
  await fsPromises.mkdir(dir, { recursive: true });
  const tmpText = path.join(dir, "tmp_script.txt");
  await fsPromises.writeFile(tmpText, text);
  const cmd = `${config.PIPER_EXECUTABLE} --model ${config.PIPER_VOICE_MODEL} --output_file ${outPath} --text_file ${tmpText}`;
  await safeExec(cmd, "Piper TTS");
  await fsPromises.unlink(tmpText);
}

async function generateStaticClip(imgPath, txt, duration, outPath) {
  log.info(`   -> Clip statico ${path.basename(outPath)}`);
  const escTxt = txt.replace(/'/g, "''").replace(/:/g, "\\:");
  const cmd = `ffmpeg -y -loop 1 -i "${imgPath}" -t ${duration} -vf "scale=1080:1920,drawtext=text='${escTxt}':fontfile='${config.FFMPEG_FONT_PATH}':fontcolor=white:fontsize=${config.INTRO_OUTRO_FONT_SIZE}:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.5:boxborderw=10" -c:v libx264 -pix_fmt yuv420p "${outPath}"`;
  await safeExec(cmd, "FFmpeg StaticClip");
}

async function generateMainVideoContent(chunks) {
  log.info("[100%] 🎬 Composizione contenuto principale...");
  const tempDir = config.TEMP_FOLDER;
  await fsPromises.mkdir(tempDir, { recursive: true });
  const videoParts = [];

  for (let i = 0; i < chunks.length; i++) {
    const txt = chunks[i];
    const img = path.join(config.IMAGE_FOLDER, `image${i}.png`);
    const audio = path.join(tempDir, `audio${i}.wav`);
    const part = path.join(tempDir, `part${i}.mp4`);

    await generateVoice(txt, audio);
    const duration = await getAudioDuration(audio);

    // sottotitoli plain‑text (Ass) ------------------------------------------------
    const subPath = path.join(tempDir, `sub${i}.txt`);
    await fsPromises.writeFile(subPath, txt.replace(/'/g, "''").replace(/:/g, "\\:"));

    const frames = Math.ceil(duration * 25);
    const zoom = `min(1.5, 1+ (n/${frames})*0.5)`;
    const xPan = `iw/2-(iw/zoom/2)+ (n/${frames})*50`;
    const yPan = `ih/2-(ih/zoom/2)`;

    const cmd = `ffmpeg -y -i "${img}" -i "${audio}" -i "${config.LOGO_PATH}" -filter_complex "[0:v]zoompan=z='${zoom}':x='${xPan}':y='${yPan}':d=${frames}:s=1080x1920[bg];[2:v]scale=${config.LOGO_WIDTH}:${config.LOGO_HEIGHT}[logo];[bg][logo]overlay=x=${config.LOGO_X}:y=${config.LOGO_Y}[v];[v]subtitles='${subPath}':force_style='FontName=${config.FFMPEG_FONT_NAME},FontSize=28,Alignment=10,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=1.5,Shadow=0.5'" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p "${part}"`;

    await safeExec(cmd, `FFmpeg Part#${i}`);
    videoParts.push(part);
  }

  // concat --------------------------------------------------------------
  const listPath = path.join(tempDir, "filelist_main.txt");
  await fsPromises.writeFile(listPath, videoParts.map((p) => `file '${p.replace(/\\/g, '/')}'`).join("\n"));
  const mainOut = path.join(tempDir, "main_content.mp4");
  await safeExec(`ffmpeg -y -f concat -safe 0 -i "${listPath}" -c copy "${mainOut}"`, "FFmpeg MainConcat");
  return mainOut;
}

async function finalizeVideo(intro, main, outro) {
  log.info("🎬 Finalizzazione video...");
  const tempDir = config.TEMP_FOLDER;
  const listPath = path.join(tempDir, "filelist_final.txt");
  await fsPromises.writeFile(listPath, [intro, main, outro].filter(Boolean).map((p) => `file '${p.replace(/\\/g, '/')}'`).join("\n"));
  const concatNoMusic = path.join(tempDir, "concat_no_music.mp4");
  await safeExec(`ffmpeg -y -f concat -safe 0 -i "${listPath}" -c copy "${concatNoMusic}"`, "FFmpeg FinalConcat");

  const cmdMusic = `ffmpeg -y -i "${concatNoMusic}" -i "${config.BACKGROUND_MUSIC_PATH}" -filter_complex "[0:a]volume=${config.VOICE_VOLUME}[a0];[1:a]volume=${config.MUSIC_VOLUME}[a1];[a0][a1]amix=inputs=2:duration=first" -map 0:v -map \"[a]\" -c:v copy -shortest "${config.OUTPUT_PATH}"`;
  await safeExec(cmdMusic, "FFmpeg Mix");
}

// ────────────────────────────────────────────────────────────────────────────
// 🧹  Cleanup & graceful shutdown
// ────────────────────────────────────────────────────────────────────────────
async function cleanTemp() {
  try {
    await fsPromises.rm(config.TEMP_FOLDER, { recursive: true, force: true });
    log.info("🧹 Temp directory rimossa.");
  } catch (e) {
    log.warn("Impossibile pulire temp:", e.message);
  }
}

process.on("SIGINT", async () => {
  log.warn("🛑 Interruzione (SIGINT) — cleanup...");
  await cleanTemp();
  process.exit(1);
});

process.on("unhandledRejection", (reason) => {
  log.error({ reason }, "❗ Unhandled Rejection");
});

// ────────────────────────────────────────────────────────────────────────────
// 🚀  Main Flow
// ────────────────────────────────────────────────────────────────────────────
(async () => {
  try {
    log.info("🚀 Avvio generazione video TikTok con AI...");
    await fsPromises.mkdir(config.TEMP_FOLDER, { recursive: true });

    // INTRO --------------------------------------------------------------
    const raw = await generateScript();
    const validated = await validateScript(raw);
    const chunks = splitScript(validated);

    const imgPrompts = await generateImagePrompts(chunks);
    await generateImages(imgPrompts);

    const intro = path.join(config.TEMP_FOLDER, "intro.mp4");
    await generateStaticClip(config.INTRO_IMAGE_PATH, config.INTRO_TEXT, config.INTRO_DURATION_SECONDS, intro);

    // MAIN ---------------------------------------------------------------
    const main = await generateMainVideoContent(chunks);

    // OUTRO --------------------------------------------------------------
    const outro = path.join(config.TEMP_FOLDER, "outro.mp4");
    await generateStaticClip(config.OUTRO_IMAGE_PATH, config.OUTRO_TEXT, config.OUTRO_DURATION_SECONDS, outro);

    // FINAL --------------------------------------------------------------
    await finalizeVideo(intro, main, outro);

    log.info(`✅ Video creato: ${config.OUTPUT_PATH}`);
  } catch (err) {
    log.error(err, "❌ Errore generazione video");
  } finally {
    await cleanTemp();
  }
})();

