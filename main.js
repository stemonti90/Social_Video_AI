const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
require("dotenv").config();

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.loadFile(path.join(__dirname, "renderer/index.html"));
}

app.whenReady().then(createWindow);

ipcMain.on("generate-video", (event, args) => {
  event.sender.send("log", "🔧 Inizio generazione video...");
  // Chiama qui app/index.js per gestire la pipeline reale
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
