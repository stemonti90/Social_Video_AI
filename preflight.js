const fs = require('fs');
const which = require('which');
const config = require('./config');

function checkFile(path, name) {
  if (!path) {
    console.warn(`[WARN] Percorso per ${name} non specificato`);
    return false;
  }
  if (!fs.existsSync(path)) {
    console.error(`[ERROR] ${name} non trovato a ${path}`);
    return false;
  }
  return true;
}

function checkExecutable(exePath, fallbackName) {
  if (exePath) {
    return checkFile(exePath, fallbackName);
  }
  try {
    which.sync(fallbackName);
    return true;
  } catch {
    console.error(`[ERROR] ${fallbackName} non trovato nel PATH`);
    return false;
  }
}

function runChecks() {
  let ok = true;
  ok &= checkExecutable(config.PIPER_EXECUTABLE, 'piper');
  ok &= checkFile(config.PIPER_VOICE_MODEL, 'modello Piper');
  ok &= checkExecutable(config.STABLE_DIFFUSION_EXECUTABLE, 'stable-diffusion');
  ok &= checkFile(config.STABLE_DIFFUSION_MODEL, 'modello Stable Diffusion');
  ok &= checkExecutable('ffmpeg', 'ffmpeg');
  return Boolean(ok);
}

module.exports = { runChecks };

