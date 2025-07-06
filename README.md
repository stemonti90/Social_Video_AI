# TikTok AI Autogen

Generatore di video ottimizzati per TikTok e Instagram eseguito interamente in locale.

## Prerequisiti
- Node.js 18+
- FFmpeg
- Piper TTS e modello vocale
- Stable Diffusion e relativo modello
- Ollama con un modello compatibile (es. `llama3`)

## Installazione
1. Copiare `.env.example` in `.env` e modificare i percorsi dei modelli e degli eseguibili.
2. Installare le dipendenze npm:
   ```bash
   npm install
   ```
3. Posizionare nella cartella `assets` le immagini per intro e outro e un file audio di sottofondo.

## Utilizzo
Eseguire:
```bash
npm start
```
Il programma genererà due video (`output/video_tiktok.mp4` e `output/video_instagram.mp4`) e i rispettivi file JSON con metadati SEO.

Se la variabile `CLEANUP_TEMP` nel file `.env` è impostata a `true` verrà eliminata la cartella temporanea al termine dell'esecuzione.

## Configurazione
Le opzioni principali sono definite in `config.js`. È possibile personalizzare:
- Percorsi dei modelli
- Testi di intro/outro
- Dimensioni del watermark e dei sottotitoli
- Trend di marketing utilizzati per la generazione degli script

## Avvertenze
Tutto il processo viene eseguito localmente, assicurarsi di avere sufficiente spazio su disco e risorse hardware adeguate per l'esecuzione di Stable Diffusion.
