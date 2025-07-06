const { spawn } = require('child_process');
const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');
// Carica le variabili d'ambiente dal file .env
require('dotenv').config();
const config = require('./config');
const { runChecks } = require('./preflight');


if (!runChecks()) {
    console.error('Preflight checks failed. Controlla la configurazione.');
    process.exit(1);
}

// Funzione helper per eseguire comandi esterni in modo sicuro usando spawn
async function safeSpawn(command, args, toolName, options = {}) {
    return new Promise((resolve, reject) => {
        const child = spawn(command, args, { shell: false, ...options });
        let stdout = '';
        let stderr = '';
        if (child.stdout) child.stdout.on('data', d => { stdout += d; });
        if (child.stderr) child.stderr.on('data', d => { stderr += d; });
        child.on('error', err => {
            reject(new Error(`${toolName} spawn error: ${err.message}`));
        });
        child.on('close', code => {
            if (code === 0) {
                resolve(stdout.trim());
            } else {
                reject(new Error(`${toolName} exited with code ${code}\n${stderr.trim()}`));
            }
        });
    });
}

/**
 * Esegue un prompt con Ollama in modo asincrono.
 * @param {string} prompt Il prompt da inviare a Ollama.
 * @returns {Promise<string>} L'output di Ollama.
 */
async function runOllama(prompt) {
    const stdout = await safeSpawn('ollama', ['run', config.OLLAMA_MODEL, JSON.stringify(prompt)], 'Ollama');
    return stdout.toString().trim();
}

// STEP 1 - Genera script con Ollama
async function generateScript() {
    console.log('[16%] ✍️  Generazione script iniziale...');
    return runOllama(config.OLLAMA_PROMPT_INITIAL);
}

// STEP 2 - Correggi grammatica/sintassi/verifica con Ollama
async function validateScript(script) {
    console.log('[33%] 📚 Validazione grammatica e veridicità...');
    return runOllama(config.OLLAMA_PROMPT_VALIDATE(script));
}

// STEP 3 - Divide script in 3 blocchi
function splitScript(text) {
    console.log('[50%] ✂️  Suddivisione testo...');
    // Miglioramento: regex più robusta per evitare di dividere su "es." o abbreviazioni.
    // Questa regex cerca punteggiatura finale seguita da uno spazio e una lettera maiuscola.
    // Per semplicità, manteniamo la logica originale ma la rendiamo più pulita.
    const sentences = text
        .split(/[.?!]/)
        .map(s => s.trim())
        .filter(s => s.length > 10); // Filtra frasi molto corte

    const totalSentences = sentences.length;
    const chunkSize = Math.ceil(totalSentences / config.VIDEO_CHUNKS);
    
    return Array.from({ length: config.VIDEO_CHUNKS }, (_, i) => sentences.slice(i * chunkSize, (i + 1) * chunkSize).join('. ').trim() + '.');
}

/**
 * Ottiene la durata di un file audio usando ffprobe.
 * @param {string} audioFilePath Il percorso del file audio.
 * @returns {Promise<number>} La durata del file audio in secondi.
 */
async function getAudioDuration(audioFilePath) {
    const args = ['-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audioFilePath];
    const stdout = await safeSpawn('ffprobe', args, 'FFprobe');
    return parseFloat(stdout.trim());
}

// STEP 4.1 - Genera i prompt per le immagini usando Ollama
async function generateImagePrompts(scriptChunks) {
    console.log('[58%] 🎨  Generazione prompt per le immagini (in parallelo)...');
    const promptPromises = scriptChunks.map(chunk => 
        runOllama(config.OLLAMA_PROMPT_IMAGE_GEN(chunk))
    );
    const results = await Promise.all(promptPromises);
    // Pulisce l'output di Ollama, rimuovendo eventuali virgolette esterne
    const cleanedPrompts = results.map(p => p.replace(/^"|"$/g, ''));
    console.log('Prompts generati:', cleanedPrompts);
    return cleanedPrompts;
}

// STEP 4.2 - Genera immagini con Stable Diffusion
async function generateImages(imagePrompts) {
    console.log('[66%] 🖼  Generazione immagini (in parallelo)...');
    await fsPromises.mkdir(config.IMAGE_FOLDER, { recursive: true });

    const imageGenerationPromises = imagePrompts.map((prompt, i) => {
        const outputPath = path.join(config.IMAGE_FOLDER, `image${i}.png`);
        console.log(`   -> Avvio generazione immagine ${i + 1}/${imagePrompts.length}`);
        const args = ['-m', config.STABLE_DIFFUSION_MODEL, '-p', prompt, '-o', outputPath, '--height', '1920', '--width', '1080', '-s', '25'];
        return safeSpawn(config.STABLE_DIFFUSION_EXECUTABLE, args, `Stable Diffusion (Image ${i+1})`).then(() => console.log(`   -> Immagine ${i+1} completata.`));
    });

    await Promise.all(imageGenerationPromises);
}

// STEP 5 - Sintetizza voce con Piper
async function generateVoice(text, audioOutputPath) {
    console.log('[83%] 🎙  Generazione voce...');
    const audioDir = path.dirname(audioOutputPath);
    const tempScriptPath = path.join(audioDir, 'temp_script.txt');
    await fsPromises.mkdir(audioDir, { recursive: true });
    await fsPromises.writeFile(tempScriptPath, text);

    const args = ['--model', config.PIPER_VOICE_MODEL, '--output_file', audioOutputPath, '--text_file', tempScriptPath];
    await safeSpawn(config.PIPER_EXECUTABLE, args, 'Piper TTS');
    await fsPromises.unlink(tempScriptPath); // Pulisce il file di script temporaneo
}

/**
 * Genera un clip video statico (introduzione o conclusione) con testo sovrapposto.
 * @param {string} imagePath Percorso dell'immagine di sfondo.
 * @param {string} text Testo da sovrapporre.
 * @param {number} duration Durata del clip in secondi.
 * @param {string} outputPath Percorso del file video di output.
 */
async function generateStaticClip(imagePath, text, duration, outputPath) {
    console.log(`   -> Creazione clip statico: ${outputPath}`);
    const escapedText = text.replace(/'/g, `''`).replace(/:/g, `\\:`);
    const filter = `[0:v]scale=1080:1920[bg];[1:v]scale=${config.LOGO_WIDTH}:${config.LOGO_HEIGHT}[logo];` +
        `[bg][logo]overlay=x=${config.LOGO_X}:y=${config.LOGO_Y}[with_logo];` +
        `[with_logo]drawtext=text='${escapedText}':fontfile='${config.FFMPEG_FONT_PATH}':fontcolor=white:fontsize=${config.INTRO_OUTRO_FONT_SIZE}:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.5:boxborderw=10`;
    const args = ['-y', '-loop', '1', '-i', imagePath, '-i', config.LOGO_PATH, '-t', String(duration), '-filter_complex', filter, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', outputPath];
    await safeSpawn('ffmpeg', args, `FFmpeg (Static Clip: ${path.basename(outputPath)})`);
}

// NUOVO STEP 6 - Genera il contenuto video principale (senza intro/outro) con sottotitoli e sincronizzazione
async function generateMainVideoContent(scriptChunks) {
    console.log('[100%] 🎬 Composizione video finale...');
    const tempDir = config.TEMP_FOLDER;
    await fsPromises.mkdir(tempDir, { recursive: true });

    const videoParts = [];
    // Genera un clip per ogni chunk
    for (let i = 0; i < scriptChunks.length; i++) {
        const chunkText = scriptChunks[i];
        const imagePath = path.join(config.IMAGE_FOLDER, `image${i}.png`);
        const audioPath = path.join(tempDir, `audio${i}.wav`);
        const videoPartPath = path.join(tempDir, `part${i}.mp4`);

        // 1. Genera audio per il chunk
        await generateVoice(chunkText, audioPath);

        // 2. Ottieni la durata dell'audio
        const duration = await getAudioDuration(audioPath);

        // 3. Crea il file dei sottotitoli per questo chunk
        const subtitleText = chunkText.replace(/'/g, `''`).replace(/:/g, `\\:`);
        const subtitlePath = path.join(tempDir, `subtitle${i}.txt`);
        await fsPromises.writeFile(subtitlePath, subtitleText);

        // 4. Crea il clip video con immagine, audio e sottotitoli
        // Calcola la durata in frame per il filtro zoompan (es. 25 frame al secondo)
        const durationFrames = Math.ceil(duration * 25); 
        // Espressioni per l'effetto Ken Burns (zoom in lento e leggero pan)
        const zoomExpr = `min(1.5, 1.00 + (n/${durationFrames})*0.5)`; // Zoom da 1.0 a 1.5
        const xPanExpr = `iw/2-(iw/zoom/2) + (n/${durationFrames})*50`; // Pan lento di 50px verso destra
        const yPanExpr = `ih/2-(ih/zoom/2)`; // Centra verticalmente
        console.log(`   -> Creazione clip ${i + 1}/${scriptChunks.length} (durata: ${duration.toFixed(2)}s)`);
        // Aggiungi il logo come terzo input (-i "${config.LOGO_PATH}")
        const filter = `[0:v]zoompan=z='${zoomExpr}':x='${xPanExpr}':y='${yPanExpr}':d=${durationFrames}:s=1080x1920[bg_zoomed];[2:v]scale=${config.LOGO_WIDTH}:${config.LOGO_HEIGHT}[logo_scaled];[bg_zoomed][logo_scaled]overlay=x=${config.LOGO_X}:y=${config.LOGO_Y}[video_with_logo];[video_with_logo]subtitles='${subtitlePath}':force_style='FontName=${config.FFMPEG_FONT_NAME},FontSize=${config.SUBTITLE_FONT_SIZE},Alignment=10,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2,Shadow=1'`;
        const args = ['-y', '-i', imagePath, '-i', audioPath, '-i', config.LOGO_PATH, '-filter_complex', filter, '-c:v', 'libx264', '-tune', 'stillimage', '-c:a', 'aac', '-b:a', '192k', '-pix_fmt', 'yuv420p', videoPartPath];
        await safeSpawn('ffmpeg', args, `FFmpeg (Main Content Part ${i+1})`);
        videoParts.push(videoPartPath);
    }
    
    // Concatena tutti i clip video del contenuto principale
    console.log('   -> Unione dei clip video del contenuto principale...');
    const fileListPath = path.join(tempDir, 'filelist_main.txt'); // New filelist name
    const fileListContent = videoParts.map(p => `file '${p.replace(/\\/g, '/')}'`).join('\n'); // Assicurati che i percorsi siano compatibili con ffmpeg su tutti i sistemi
    await fsPromises.writeFile(fileListPath, fileListContent);

    const mainVideoPath = path.join(tempDir, 'main_content.mp4'); // New output name
    const concatArgs = ['-y', '-f', 'concat', '-safe', '0', '-i', fileListPath, '-c', 'copy', mainVideoPath];
    await safeSpawn('ffmpeg', concatArgs, 'FFmpeg (Main Content Concatenation)');

    return mainVideoPath; // Return the path to the main video content
}

// Nuova funzione per gestire la concatenazione finale e il mixaggio audio
async function finalizeVideo(introVideoPath, mainVideoPath, outroVideoPath, outputPath, resolution) {
    console.log('🎬 Finalizzazione video (intro + contenuto + outro + musica)...');
    const tempDir = config.TEMP_FOLDER;

    // 1. Concatena intro, main, outro
    const finalFileListPath = path.join(tempDir, 'filelist_final.txt');
    const finalFileListContent = [introVideoPath, mainVideoPath, outroVideoPath]
        .filter(Boolean) // Filter out null/undefined if any part is optional
        .map(p => `file '${p.replace(/\\/g, '/')}'`).join('\n'); // Assicurati che i percorsi siano compatibili con ffmpeg su tutti i sistemi
    await fsPromises.writeFile(finalFileListPath, finalFileListContent);

    const concatenatedVideoNoMusicPath = path.join(tempDir, 'concatenated_no_music.mp4');
    const concatArgs = ['-y', '-f', 'concat', '-safe', '0', '-i', finalFileListPath, '-c', 'copy', concatenatedVideoNoMusicPath];
    await safeSpawn('ffmpeg', concatArgs, 'FFmpeg (Final Concatenation)');

    // 2. Aggiungi musica di sottofondo al video completo
    console.log('   -> Aggiunta musica di sottofondo...');
    const withMusicPath = path.join(tempDir, 'with_music.mp4');
    const musicFilter = `[0:a]volume=${config.VOICE_VOLUME}[a0];[1:a]volume=${config.MUSIC_VOLUME}[a1];[a0][a1]amix=inputs=2:duration=first[a]`;
    const musicArgs = ['-y', '-i', concatenatedVideoNoMusicPath, '-i', config.BACKGROUND_MUSIC_PATH, '-filter_complex', musicFilter, '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-shortest', withMusicPath];
    await safeSpawn('ffmpeg', musicArgs, 'FFmpeg (Audio Mixing)');

    // 3. Scala o ritaglia in base alla piattaforma di destinazione
    const scaleExpr = `scale=${resolution.width}:${resolution.height}:force_original_aspect_ratio=increase`;
    const cropExpr = `crop=${resolution.width}:${resolution.height}`;
    const scaleArgs = ['-y', '-i', withMusicPath, '-vf', `${scaleExpr},${cropExpr}`, '-c:v', 'libx264', '-c:a', 'copy', outputPath];
    await safeSpawn('ffmpeg', scaleArgs, 'FFmpeg (Resize)');
}

// Genera metadati SEO con Ollama
async function generateSocialMetadata(script, platformKey) {
    console.log(`   -> Generazione metadata per ${platformKey}...`);
    const prompt = config.OLLAMA_PROMPT_METADATA(script, platformKey);
    const raw = await runOllama(prompt);
    try {
        return JSON.parse(raw);
    } catch {
        console.warn('   -> Formato metadati non valido, uso configurazione di default');
        return {};
    }
}

// Crea un file JSON con i metadati per la piattaforma di destinazione
async function createMetadataFile(socialKey, outputPath, metadata = {}) {
    const defaults = config.SOCIAL_METADATA[socialKey] || {};
    const finalData = Object.assign({ video: path.basename(outputPath) }, defaults, metadata);
    const jsonPath = outputPath.replace(/\.mp4$/, '.json');
    await fsPromises.writeFile(jsonPath, JSON.stringify(finalData, null, 2));
    console.log(`   -> Metadata ${socialKey} salvati in ${jsonPath}`);
}

// MAIN FLOW
(async () => {
    try {
        console.log('🚀 Avvio generazione video TikTok con AI...');
        const tempDir = config.TEMP_FOLDER;
        await fsPromises.mkdir(tempDir, { recursive: true }); // Ensure temp dir exists early
        await fsPromises.mkdir(path.dirname(config.OUTPUT_TIKTOK_PATH), { recursive: true });

        // Genera Introduzione
        const rawScript = await generateScript();
        const validatedScript = await validateScript(rawScript);
        const scriptChunks = splitScript(validatedScript);

        // Metadati SEO personalizzati per ciascuna piattaforma
        const tiktokMeta = await generateSocialMetadata(validatedScript, 'TikTok');
        const instagramMeta = await generateSocialMetadata(validatedScript, 'Instagram');

        const imagePrompts = await generateImagePrompts(scriptChunks);
        await generateImages(imagePrompts);
        
        const introVideoPath = path.join(tempDir, 'intro.mp4');
        await generateStaticClip(config.INTRO_IMAGE_PATH, config.INTRO_TEXT, config.INTRO_DURATION_SECONDS, introVideoPath);

        // Genera Contenuto Principale
        const mainVideoPath = await generateMainVideoContent(scriptChunks);

        // Genera Conclusione
        const outroVideoPath = path.join(tempDir, 'outro.mp4');
        await generateStaticClip(config.OUTRO_IMAGE_PATH, config.OUTRO_TEXT, config.OUTRO_DURATION_SECONDS, outroVideoPath);

        // Finalizza (concatena e aggiungi musica) per TikTok
        await finalizeVideo(introVideoPath, mainVideoPath, outroVideoPath, config.OUTPUT_TIKTOK_PATH, config.TIKTOK_RESOLUTION);
        await createMetadataFile('tiktok', config.OUTPUT_TIKTOK_PATH, tiktokMeta);

        // Finalizza (concatena e aggiungi musica) per Instagram
        await finalizeVideo(introVideoPath, mainVideoPath, outroVideoPath, config.OUTPUT_INSTAGRAM_PATH, config.INSTAGRAM_RESOLUTION);
        await createMetadataFile('instagram', config.OUTPUT_INSTAGRAM_PATH, instagramMeta);

        if (config.CLEANUP_TEMP) {
            console.log(`   -> Pulizia file temporanei in ${tempDir}...`);
            await fsPromises.rm(tempDir, { recursive: true, force: true });
        }

        console.log(`\n✅ Video TikTok: ${config.OUTPUT_TIKTOK_PATH}`);
        console.log(`✅ Video Instagram: ${config.OUTPUT_INSTAGRAM_PATH}\n`);
    } catch (error) {
        console.error('\n❌ Si è verificato un errore durante la generazione del video:');
        console.error(error.stderr || error.message);
    } finally {
        if (config.CLEANUP_TEMP) {
            await fsPromises.rm(config.TEMP_FOLDER, { recursive: true, force: true }).catch(() => {});
        }
    }
})();
