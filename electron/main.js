// Social AstroStacker — Electron main process. Wraps the local `avp` CLI.
const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

// In dev, the project root is one level up from electron/. When packaged, the bundled app lives
// elsewhere, so the local Python engine (.venv) + data (projects/, config.yaml, src/) are located
// at a fixed path — overridable via AVP_PROJECT_ROOT. The heavy local AI engine/models can't be
// bundled into a clickable app, so the packaged UI still drives the engine in this project folder.
const ROOT = process.env.AVP_PROJECT_ROOT
  || (app.isPackaged ? "/Users/ste/Desktop/Progetti/AUT_VIDEO_PIPELINE" : path.resolve(__dirname, ".."));
const AVP = path.join(ROOT, ".venv", "bin", "avp");
const PROJECTS = path.join(ROOT, "projects");
const ENV = { ...process.env, PYTHONPATH: path.join(ROOT, "src") };

let win;
function createWindow() {
  win = new BrowserWindow({
    width: 1200, height: 820, minWidth: 940, minHeight: 640,
    backgroundColor: "#070a13",
    title: "Social AstroStacker",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(createWindow);
app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });

// ---------- helpers ----------
function slugify(s) {
  return (s || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "video";
}
function readManifest(slug) {
  try { return JSON.parse(fs.readFileSync(path.join(PROJECTS, slug, "manifest.json"), "utf8")); }
  catch { return null; }
}
function runAvp(args) {
  return new Promise((resolve, reject) => {
    const p = spawn(AVP, args, { cwd: ROOT, env: ENV });
    let out = "", err = "";
    p.stdout.on("data", (d) => (out += d));
    p.stderr.on("data", (d) => (err += d));
    p.on("close", (code) => (code === 0 ? resolve(out) : reject(new Error(err || out || `exit ${code}`))));
    p.on("error", reject);
  });
}

// ---------- IPC ----------
ipcMain.handle("avp:list", () => {
  if (!fs.existsSync(PROJECTS)) return [];
  return fs.readdirSync(PROJECTS)
    .filter((d) => fs.existsSync(path.join(PROJECTS, d, "manifest.json")))
    .map((slug) => {
      const m = readManifest(slug) || {};
      const stages = {};
      for (const [k, v] of Object.entries(m.stages || {})) stages[k] = (v && v.state) || "pending";
      return { slug, title: m.title || slug, stages };
    });
});

ipcMain.handle("avp:delete", async (_e, slug) => {
  // Defense in depth: validate here too (the CLI re-validates and is the source of truth).
  if (typeof slug !== "string" || !/^[a-z0-9_][a-z0-9_-]*$/.test(slug)) {
    throw new Error("slug non valido");
  }
  await runAvp(["delete", slug]);   // CLI safely removes projects/<slug> (folder + all files)
  return true;
});

ipcMain.handle("avp:config-get", async () => {
  try { return JSON.parse(await runAvp(["config-get"])); } catch { return {}; }
});
ipcMain.handle("avp:config-set", async (_e, patch) => {
  await runAvp(["config-set", JSON.stringify(patch)]);
  return true;
});

ipcMain.handle("avp:new", (e, topic) => {
  // Stream the script-gen logs so the renderer can show real progress (it's a multi-minute LLM job).
  return new Promise((resolve, reject) => {
    const slug = slugify(topic);
    const send = (ev) => { if (!e.sender.isDestroyed()) e.sender.send("avp:new-event", ev); };
    const p = spawn(AVP, ["new", slug, "--topic", topic, "-v"], { cwd: ROOT, env: ENV });
    let err = "";
    const emit = (buf) => buf.toString().split(/\r?\n/).forEach((line) => {
      if (line.trim()) send({ type: "log", line });
    });
    p.stdout.on("data", emit);
    p.stderr.on("data", (d) => { err += d; emit(d); });   // avp logs go to stderr
    p.on("close", (code) => {
      if (code === 0) {
        const m = readManifest(slug) || {};
        send({ type: "done" });
        resolve({ slug, title: m.title || topic });
      } else {
        send({ type: "error", line: `exit ${code}` });
        reject(new Error(err || `exit ${code}`));
      }
    });
    p.on("error", (e2) => { send({ type: "error", line: String(e2) }); reject(e2); });
  });
});

ipcMain.handle("avp:readScript", (_e, slug) => {
  try { return fs.readFileSync(path.join(PROJECTS, slug, "script.md"), "utf8"); } catch { return ""; }
});

ipcMain.handle("avp:saveScript", (_e, { slug, text }) => {
  fs.writeFileSync(path.join(PROJECTS, slug, "script.md"), text);
  return true;
});

ipcMain.handle("avp:readMetadata", (_e, slug) => {
  try { return JSON.parse(fs.readFileSync(path.join(PROJECTS, slug, "metadata.json"), "utf8")); }
  catch { return {}; }
});

ipcMain.handle("avp:videoUrl", (_e, { slug, engine }) => {
  const f = path.join(PROJECTS, slug, `${slug}.${engine}.mp4`);
  return fs.existsSync(f) ? "file://" + f : "";
});

ipcMain.handle("avp:publish", async (_e, { slug, platforms, go }) => {
  const args = ["publish", slug];
  if (go) args.push("--go");
  if (platforms && platforms.length) args.push("--platforms", ...platforms);
  await runAvp(args);
  let plan = [];
  try { plan = JSON.parse(fs.readFileSync(path.join(PROJECTS, slug, "publish_plan.json"), "utf8")); } catch {}
  return { plan };
});

ipcMain.handle("avp:build", (e, slug) => {
  return new Promise((resolve) => {
    const send = (ev) => { if (!e.sender.isDestroyed()) e.sender.send("avp:build-event", ev); };
    const p = spawn(AVP, ["build", slug, "-v"], { cwd: ROOT, env: ENV });
    const handle = (buf) => {
      for (const line of String(buf).split(/\r?\n/)) {
        if (!line.trim()) continue;
        send({ type: "log", line });
        const m = line.match(/▶\s*(\w+)/);   // "▶ stage"
        if (m) send({ type: "stage", stage: m[1] });
      }
    };
    p.stdout.on("data", handle);
    p.stderr.on("data", handle);
    p.on("close", (code) => { send({ type: code === 0 ? "done" : "error", line: `exit ${code}` }); resolve(); });
    p.on("error", (err) => { send({ type: "error", line: String(err) }); resolve(); });
  });
});
