const path = require('path');

// Percorso del modello vocale di Piper TTS
// Assicurati che questo percorso sia corretto per la tua macchina.
// Esempio con variabile d'ambiente: process.env.PIPER_VOICE_MODEL_PATH
const PIPER_VOICE_MODEL_PATH = process.env.PIPER_VOICE_MODEL_PATH; 

// Percorso dell'eseguibile di Piper TTS
// Modifica questo percorso se hai installato Piper in una directory diversa.
const PIPER_EXECUTABLE_PATH = process.env.PIPER_EXECUTABLE_PATH;

// --- Configurazione Stable Diffusion ---
// Percorso dell'eseguibile di Stable Diffusion (es. stable-diffusion.cpp)
const STABLE_DIFFUSION_EXECUTABLE_PATH = process.env.STABLE_DIFFUSION_EXECUTABLE_PATH;

// Percorso del modello di Stable Diffusion (es. .safetensors)
const STABLE_DIFFUSION_MODEL_PATH = process.env.STABLE_DIFFUSION_MODEL_PATH;

// Percorso di un font per i sottotitoli (es. Arial, Helvetica, o un font scaricato)
// NOTA: FFmpeg su alcuni sistemi vuole il nome del font, non il percorso.
// Su macOS, puoi usare "Arial Unicode MS". Su Linux, potresti dover usare il percorso.
// Esegui `fc-list` su Linux per vedere i nomi dei font disponibili.
const FFMPEG_FONT_PATH = process.env.FFMPEG_FONT_PATH;
const FFMPEG_FONT_NAME = 'Arial'; // Nome del font come visto da FFmpeg/fontconfig

// --- Configurazione Audio ---
// Percorso del file musicale di sottofondo (es. .mp3, .wav)
const BACKGROUND_MUSIC_PATH = './assets/music/background_music.mp3'; // Assicurati che questo file esista

// Volumi per il mixaggio (0.0 a 1.0)
const VOICE_VOLUME = 1.0; // Volume della voce
const MUSIC_VOLUME = 0.3; // Volume della musica di sottofondo (solitamente più basso)

// --- Configurazione Intro/Outro ---
const INTRO_IMAGE_PATH = './assets/intro_image.png'; // Assicurati che questo file esista
const OUTRO_IMAGE_PATH = './assets/outro_image.png'; // Assicurati che questo file esista
const INTRO_DURATION_SECONDS = 5; // Durata dell'introduzione in secondi
const OUTRO_DURATION_SECONDS = 5; // Durata della conclusione in secondi
const INTRO_TEXT = 'Il tuo video inizia qui!'; // Testo per l'introduzione
const OUTRO_TEXT = 'Grazie per aver guardato!'; // Testo per la conclusione
const INTRO_OUTRO_FONT_SIZE = 48; // Dimensione del font per intro/outro

// Dimensione font dei sottotitoli (accessibilità)
const SUBTITLE_FONT_SIZE = 36;

// Argomenti e trend di marketing per il prossimo anno
const NEXT_YEAR = new Date().getFullYear() + 1;
const MARKETING_TRENDS = [
    'AI-driven personalization',
    'augmented reality',
    'voice search',
    'sustainability marketing'
];

// --- Configurazione Logo ---
const LOGO_PATH = './assets/logo.png'; // Assicurati che questo file esista e sia trasparente (es. PNG con canale alpha)
const LOGO_WIDTH = 200; // Larghezza del logo in pixel
const LOGO_HEIGHT = 200; // Altezza del logo in pixel
// Distanza del watermark dai bordi destro e inferiore. Viene usata per
// calcolare la posizione in basso a destra.
const LOGO_MARGIN = 50;
const LOGO_X = `main_w-overlay_w-${LOGO_MARGIN}`; // Posizione X (bordo destro)
const LOGO_Y = `main_h-overlay_h-${LOGO_MARGIN}`; // Posizione Y (bordo inferiore)

// Informazioni predefinite per la condivisione sui social
const SOCIAL_METADATA = {
    tiktok: {
        title: 'Amazing crypto fact!',
        tags: ['crypto', 'fyp', 'viral'],
        hashtags: ['#crypto', '#foryou', '#viral']
    },
    instagram: {
        title: 'Curiosit\u00e0 sulle crypto',
        tags: ['crypto', 'reels', 'viral'],
        hashtags: ['#crypto', '#reels', '#viral']
    }
};
module.exports = {
    // Prompts per Ollama
    OLLAMA_MODEL: 'llama3',
    OLLAMA_PROMPT_INITIAL: `Genera uno script originale e accattivante per un video di 60 secondi da pubblicare su TikTok o Instagram. Cita almeno una delle tendenze di marketing digitale previste per il ${NEXT_YEAR}, come ${MARKETING_TRENDS.join(', ')}. L'argomento principale è legato alle criptovalute. Usa frasi brevi e termina con una call to action coinvolgente.`,
    OLLAMA_PROMPT_VALIDATE: (script) => `Rivedi questo testo per grammatica, chiarezza e veridicità. Riscrivi solo se serve mantenendo lo stile narrativo:\n\n${script}`,
    OLLAMA_PROMPT_IMAGE_GEN: (text) => `Trasforma la seguente frase narrativa in un prompt conciso e descrittivo per un generatore di immagini AI come Stable Diffusion. Il prompt deve essere in inglese, focalizzato sugli elementi visivi, e includere uno stile artistico (es. "digital art", "photorealistic", "cinematic"). Rispondi solo con il prompt. Frase: "${text}"`,
    OLLAMA_PROMPT_METADATA: (script, platform) => `Sei un esperto di marketing digitale. Crea un JSON con titolo, cinque tag e cinque hashtag ottimizzati SEO per ${platform}. Usa come ispirazione il seguente script: \n\n${script}`,

    // Percorsi dei file e delle directory
    // SCRIPT_PATH e AUDIO_PATH non sono più usati globalmente, ma per chunk
    OUTPUT_PATH: './output/video_finale.mp4',
    OUTPUT_TIKTOK_PATH: './output/video_tiktok.mp4',
    OUTPUT_INSTAGRAM_PATH: './output/video_instagram.mp4',
    TIKTOK_RESOLUTION: { width: 1080, height: 1920 },
    INSTAGRAM_RESOLUTION: { width: 1080, height: 1350 },
    IMAGE_FOLDER: './assets/images',
    TEMP_FOLDER: './temp', // Nuova cartella per file temporanei

    // Eseguibili e modelli
    PIPER_VOICE_MODEL: PIPER_VOICE_MODEL_PATH,
    PIPER_EXECUTABLE: PIPER_EXECUTABLE_PATH,

    // Configurazione Stable Diffusion
    STABLE_DIFFUSION_EXECUTABLE: STABLE_DIFFUSION_EXECUTABLE_PATH,
    STABLE_DIFFUSION_MODEL: STABLE_DIFFUSION_MODEL_PATH,

    // Configurazione FFmpeg
    FFMPEG_FONT_PATH: FFMPEG_FONT_PATH, // Potrebbe non essere necessario se FFMPEG_FONT_NAME funziona
    FFMPEG_FONT_NAME: FFMPEG_FONT_NAME,

    // Configurazione Audio
    BACKGROUND_MUSIC_PATH: BACKGROUND_MUSIC_PATH,
    VOICE_VOLUME: VOICE_VOLUME,
    MUSIC_VOLUME: MUSIC_VOLUME,

    // Configurazione Video
    VIDEO_CHUNKS: 3, // Numero di blocchi in cui dividere il video

    // Configurazione Intro/Outro
    INTRO_IMAGE_PATH: INTRO_IMAGE_PATH,
    OUTRO_IMAGE_PATH: OUTRO_IMAGE_PATH,
    INTRO_DURATION_SECONDS: INTRO_DURATION_SECONDS,
    OUTRO_DURATION_SECONDS: OUTRO_DURATION_SECONDS,
    INTRO_TEXT: INTRO_TEXT,
    OUTRO_TEXT: OUTRO_TEXT,
    INTRO_OUTRO_FONT_SIZE: INTRO_OUTRO_FONT_SIZE,
    SUBTITLE_FONT_SIZE: SUBTITLE_FONT_SIZE,

    // Configurazione Logo
    LOGO_PATH: LOGO_PATH,
    LOGO_WIDTH: LOGO_WIDTH,
    LOGO_HEIGHT: LOGO_HEIGHT,
    LOGO_MARGIN: LOGO_MARGIN,
    LOGO_X: LOGO_X,
    LOGO_Y: LOGO_Y,
    SOCIAL_METADATA: SOCIAL_METADATA,
    MARKETING_TRENDS: MARKETING_TRENDS,
};
