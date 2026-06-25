/* AUT · Video Pipeline — renderer logic (vanilla, accessible).
   Talks to window.avp (Electron preload) or falls back to mock data in a browser. */
"use strict";

const STAGES = [
  ["script", "Copione"], ["voice", "Voce"], ["footage", "Footage"],
  ["captions", "Sottotitoli"], ["metadata", "Metadati"], ["assemble", "Montaggio"],
];
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

let current = null;          // open project slug
let unsubscribeBuild = null;
let currentStages = {};      // stage states of the open project (gates Anteprima/Pubblica)
let busy = false;            // a long action is running — inhibit duplicate triggers

/* ---------------- backend bridge (real or mock) ---------------- */
const REAL = typeof window !== "undefined" && window.avp;
const API = REAL || makeMock();   // NOTE: not named `avp` — the preload exposes a global `avp`

function announce(msg) {
  $("#announce").textContent = "";
  // toggle to force SR re-announce
  requestAnimationFrame(() => { $("#announce").textContent = msg; });
}
function setConn(text, lamp = "idle") {
  $("#conn-text").textContent = text;
  $("#conn-status .lamp").className = "lamp lamp--" + lamp;
}
function lampFor(state) {
  if (state === "done") return "done";
  if (state === "failed") return "fail";
  if (state === "partial" || state === "active" || state === "running") return "active";
  return "idle";
}
// status as SHAPE + colour + text (WCAG 1.4.1 — never colour alone)
function stateGlyph(state) {
  if (state === "done") return { cls: "done", g: "✓", label: "fatto" };
  if (state === "failed") return { cls: "fail", g: "✕", label: "errore" };
  if (state === "partial" || state === "active" || state === "running") return { cls: "active", g: "◐", label: "in corso" };
  return { cls: "idle", g: "○", label: "in attesa" };
}

/* ---------------- view routing ---------------- */
function showView(name) {
  $$(".view").forEach((v) => { v.classList.remove("is-active"); v.hidden = true; });
  const el = $("#view-" + name);
  el.hidden = false; el.classList.add("is-active");
  $$(".rail__btn").forEach((b) => {
    const on = b.dataset.view === name;
    b.classList.toggle("is-active", on);
    if (on) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  });
  // move focus to the new section's heading so screen-reader users follow the change (WCAG 2.4.3)
  const heading = el.querySelector(".view__title");
  if (heading) { heading.setAttribute("tabindex", "-1"); heading.focus(); }
}

$$(".rail__btn").forEach((b) =>
  b.addEventListener("click", () => {
    showView(b.dataset.view);
    if (b.dataset.view === "projects") loadProjects();
    if (b.dataset.view === "settings") loadSettings();
  })
);

/* ---------------- projects ---------------- */
async function loadProjects() {
  const list = $("#projects-list");
  list.innerHTML = "";
  let projects = [];
  try { projects = await API.listProjects(); } catch (e) { console.error(e); }
  $("#projects-empty").hidden = projects.length > 0;
  for (const p of projects) {
    const li = document.createElement("li");
    li.className = "card";
    const stages = STAGES.map(([k, label]) => {
      const s = stateGlyph((p.stages && p.stages[k]) || "pending");
      return `<span class="stage-dot"><span class="stat stat--${s.cls}" aria-hidden="true">${s.g}</span> ${label}<span class="sr-only">: ${s.label}</span></span>`;
    }).join("");
    li.innerHTML = `
      <div class="card__head">
        <div class="card__heading">
          <h2 class="card__title">${escapeHtml(p.title || p.slug)}</h2>
          <p class="card__slug">${escapeHtml(p.slug)}</p>
        </div>
        <div class="card__actions">
          <button type="button" class="card__menu-btn" aria-haspopup="true" aria-expanded="false"
                  aria-label="Azioni per ${escapeHtml(p.title || p.slug)}">⋯</button>
          <div class="card__menu" role="menu" hidden>
            <button type="button" role="menuitem" class="card__delete">Elimina progetto</button>
          </div>
        </div>
      </div>
      <div class="card__stages" role="group" aria-label="Stato stadi di ${escapeHtml(p.slug)}">${stages}</div>
      <button type="button" class="btn btn--ghost card__open">Apri progetto</button>`;
    li.querySelector(".card__open").addEventListener("click", () => openProject(p.slug, p.title));
    wireCardMenu(li, p);
    list.appendChild(li);
  }
}

/* ---------------- card ⋯ menu + delete ---------------- */
function closeAllMenus(except) {
  $$(".card__menu").forEach((m) => {
    if (m === except || m.hidden) return;
    m.hidden = true;
    const b = m.parentElement.querySelector(".card__menu-btn");
    if (b) b.setAttribute("aria-expanded", "false");
  });
}

function wireCardMenu(li, p) {
  const btn = li.querySelector(".card__menu-btn");
  const menu = li.querySelector(".card__menu");
  const del = li.querySelector(".card__delete");
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const willOpen = menu.hidden;
    closeAllMenus(menu);
    menu.hidden = !willOpen;
    btn.setAttribute("aria-expanded", String(willOpen));
    if (willOpen) del.focus();
  });
  menu.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { menu.hidden = true; btn.setAttribute("aria-expanded", "false"); btn.focus(); }
  });
  del.addEventListener("click", () => {
    menu.hidden = true; btn.setAttribute("aria-expanded", "false");
    confirmDelete(p.slug, p.title || p.slug);
  });
}
// any click outside an open menu closes it (clicks on a card's own ⋯/menu manage themselves)
document.addEventListener("click", (e) => {
  if (e.target.closest && e.target.closest(".card__actions")) return;
  closeAllMenus();
});

async function doDelete(slug, label) {
  try {
    await API.deleteProject(slug);
    if (current === slug) { current = null; showView("projects"); }
    announce(`Progetto ${label} eliminato.`);
    await loadProjects();
    const r = $("#refresh-projects"); if (r) r.focus();   // move focus off the now-gone card
  } catch (e) {
    console.error(e);
    announce(`Errore nell'eliminare ${label}: ${(e && e.message) || e}`);
  }
}

function confirmDelete(slug, label) {
  const dlg = $("#confirm-delete");
  if (!dlg || typeof dlg.showModal !== "function") {   // fallback when <dialog> is unsupported
    if (window.confirm(`Eliminare «${label}»? L'operazione non è reversibile.`)) doDelete(slug, label);
    return;
  }
  $("#confirm-delete-name").textContent = label;
  const go = $("#confirm-delete-go");
  const cancel = $("#confirm-delete-cancel");
  // wire fresh each time so the handlers close over THIS slug; clean up to avoid stacking
  const cleanup = () => {
    go.removeEventListener("click", onGo);
    cancel.removeEventListener("click", onCancel);
    dlg.removeEventListener("close", onCancel);
  };
  const onGo = () => { cleanup(); dlg.close(); doDelete(slug, label); };
  const onCancel = () => { cleanup(); if (dlg.open) dlg.close(); };
  go.addEventListener("click", onGo);
  cancel.addEventListener("click", onCancel);
  dlg.addEventListener("close", onCancel);   // Escape / backdrop also cancels cleanly
  dlg.showModal();
}

/* ---------------- new project ---------------- */
// Inline status helper: green success (auto-clears after 3s), red error (persists), plain otherwise.
function setStatus(sel, text, kind) {
  const el = $(sel); if (!el) return;
  el.textContent = text;
  el.classList.remove("inline-status--ok", "inline-status--err");
  if (kind) el.classList.add("inline-status--" + kind);
  if (kind === "ok") setTimeout(() => {
    if (el.textContent === text) { el.textContent = ""; el.classList.remove("inline-status--ok"); }
  }, 3000);
}

// Script-gen progress: time-based creep (always moving → shows it's NOT stuck) lifted to real floors
// when the streamed log hits a milestone. The model job is opaque/multi-minute, so the percentage
// (2 decimals, as requested) reassures the user while the draft+critique phases have no log line.
// Per-phase progress bands. Within a band the bar approaches `ceil` ASYMPTOTICALLY (never reaches it
// from the clock alone), so the 6-decimal % keeps ticking every frame and never sits frozen, even if a
// phase runs long. A real log milestone jumps to the next band. tau ≈ that phase's typical seconds.
const NEW_BANDS = [
  { anchor: 0,  ceil: 55, tau: 170, label: "Scrivo la prima bozza…" },
  { anchor: 55, ceil: 80, tau: 110, label: "Rifinitura 1 di 2…" },
  { anchor: 80, ceil: 97, tau: 90,  label: "Rifinitura 2 di 2…" },
];
let _newTimer = null, _newAnchorT = 0, _newBand = 0;
function _setNewBar(pct) {
  pct = Math.max(0, Math.min(100, pct));
  const fill = $("#new-progress-fill"), track = $("#new-progress-track"), lbl = $("#new-progress-pct");
  if (fill) fill.style.width = pct.toFixed(2) + "%";          // CSS width: 2 dp is plenty (sub-pixel)
  if (track) track.setAttribute("aria-valuenow", pct.toFixed(2));
  // 6 decimals on the visible % so it ticks every frame on a multi-minute job → clearly not stuck
  if (lbl) lbl.textContent = pct.toFixed(6) + "%";
}
function _newStage(t) { const el = $("#new-progress-stage"); if (el) el.textContent = t; }
function _stopNewTimer() { if (_newTimer) { clearInterval(_newTimer); _newTimer = null; } }
function _newSetBand(i) { _newBand = i; _newAnchorT = Date.now(); _newStage(NEW_BANDS[i].label); }
function startNewProgress() {
  _stopNewTimer();
  const box = $("#new-progress"); if (box) box.hidden = false;
  _newSetBand(0); _setNewBar(0);
  _newTimer = setInterval(() => {
    const b = NEW_BANDS[_newBand], t = (Date.now() - _newAnchorT) / 1000;
    _setNewBar(b.ceil - (b.ceil - b.anchor) * Math.exp(-t / b.tau));   // asymptotic → always ticking
  }, 150);
}
function onNewLog(line) {
  if (/refine pass\s*1\s*\//i.test(line)) _newSetBand(1);
  else if (/refine pass\s*2\s*\//i.test(line)) _newSetBand(2);
}
function finishNewProgress(ok) {
  _stopNewTimer();
  if (ok) { _setNewBar(100); _newStage("Pronto ✓"); setTimeout(() => { const b = $("#new-progress"); if (b) b.hidden = true; }, 1500); }
  else { _newStage("Errore"); }
}

$("#new-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const topic = $("#topic").value.trim();
  if (!topic) return;
  const btn = $("#generate-btn");
  btn.disabled = true; btn.setAttribute("aria-busy", "true"); btn.textContent = "Genero…";
  setStatus("#new-status", "", null);
  announce("Generazione del copione in corso");
  const unsub = API.onNewEvent ? API.onNewEvent((ev) => { if (ev.type === "log") onNewLog(ev.line); }) : null;
  startNewProgress();
  try {
    const res = await API.newProject(topic);
    finishNewProgress(true);
    await openProject(res.slug, res.title);
    selectTab("tab-review");
    announce("Copione pronto per la revisione");
  } catch (err) {
    finishNewProgress(false);
    setStatus("#new-status", "Errore nella generazione: " + ((err && err.message) || err), "err");
    announce("Errore nella generazione");
  } finally {
    if (unsub) unsub();
    btn.disabled = false; btn.removeAttribute("aria-busy"); btn.textContent = "Genera copione";
  }
});

/* ---------------- open / workspace ---------------- */
async function openProject(slug, title) {
  current = slug;
  currentStages = {};
  $("#h-project").textContent = title || slug;
  showView("project");
  renderStageRail({});
  selectTab("tab-review");
  const blog = $("#build-log");                       // re-seed the placeholder (don't leave it blank)
  blog.textContent = "Il log apparirà qui all'avvio della build.";
  blog.dataset.seeded = "1";
  setStatus("#script-status", "", null);
  await loadScript();
  refreshStatus();
}
$("#back-btn").addEventListener("click", () => { showView("projects"); loadProjects(); });

function renderStageRail(stages) {
  const ol = $("#stage-rail");
  ol.innerHTML = STAGES.map(([k, label]) => {
    const s = stateGlyph(stages[k] || "pending");
    return `<li><span class="stat stat--${s.cls}" aria-hidden="true">${s.g}</span> ${label}<span class="sr-only">: ${s.label}</span></li>`;
  }).join("");
}
async function refreshStatus() {
  try {
    const projects = await API.listProjects();
    const p = projects.find((x) => x.slug === current);
    currentStages = (p && p.stages) || {};
  } catch (e) { currentStages = {}; }
  renderStageRail(currentStages);
  updateWizardFoot(currentStepId());
}

/* ---------------- tabs (accessible) ---------------- */
const tabs = $$(".tab");
function selectTab(id) {
  tabs.forEach((t) => {
    const on = t.id === id;
    t.classList.toggle("is-active", on);
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    const panel = $("#" + t.getAttribute("aria-controls"));
    panel.hidden = !on; panel.classList.toggle("is-active", on);
  });
  updateWizardFoot(id);
}
$(".tablist").addEventListener("click", (e) => { const t = e.target.closest(".tab"); if (t) selectTab(t.id); });
$(".tablist").addEventListener("keydown", (e) => {
  const i = tabs.findIndex((t) => t.id === document.activeElement.id);
  if (i < 0) return;
  let j = null;
  if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
  else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
  else if (e.key === "Home") j = 0;
  else if (e.key === "End") j = tabs.length - 1;
  if (j === null) return;
  e.preventDefault(); tabs[j].focus(); selectTab(tabs[j].id);
});

/* ---------------- step wizard footer (back / next / primary action) ---------------- */
const STEPS = ["tab-review", "tab-build", "tab-preview", "tab-publish"];
const STEP_LABELS = ["Revisione", "Build", "Anteprima", "Pubblica"];
const STEP_PRIMARY = {
  "tab-review":  { label: "Salva e monta",      run: () => saveAndBuild() },
  "tab-build":   { label: "Avvia build",        run: () => startBuild() },
  "tab-preview": { label: "Aggiorna anteprima", run: () => loadPreview() },
  "tab-publish": { label: "Genera piano (dry run)", run: () => doPublish(false) },
};
function currentStepId() {
  const t = tabs.find((x) => x.classList.contains("is-active"));
  return t ? t.id : "tab-review";
}
function stepReady(id) {
  // Anteprima & Pubblica require a finished build (output exists)
  if (id === "tab-preview" || id === "tab-publish") return currentStages.assemble === "done";
  return true;
}
function applyReadiness(id) {
  const ready = stepReady(id);
  const onPrev = id === "tab-preview";
  const onPub = id === "tab-publish";
  // Show the guard only on its own not-ready step…
  const pg = $("#preview-guard"); if (pg) pg.hidden = !(onPrev && !ready);
  const ug = $("#publish-guard"); if (ug) ug.hidden = !(onPub && !ready);
  // …and hide the actual content, so we never show empty players or dead controls.
  const pm = $("#preview-media"); if (pm) pm.hidden = onPrev && !ready;
  const pmeta = $("#preview-meta"); if (pmeta) pmeta.hidden = onPrev && !ready;
  const pc = $("#publish-controls"); if (pc) pc.hidden = onPub && !ready;
  const pb = $("#publish-btn"); if (pb) pb.disabled = busy || (onPub && !ready);
  return ready;
}
function setBusy(on) { busy = on; updateWizardFoot(currentStepId()); }
function updateWizardFoot(id) {
  const i = Math.max(0, STEPS.indexOf(id));
  $("#wizard-step").textContent = `Passo ${i + 1} di ${STEPS.length} · ${STEP_LABELS[i]}`;
  $("#nav-back").textContent = i === 0 ? "← Progetti" : "← Indietro";
  $("#nav-back").disabled = busy;
  $("#nav-next").disabled = busy || i === STEPS.length - 1;
  $("#nav-next").title = i === STEPS.length - 1 ? "Ultimo passo" : (busy ? "Attendi: operazione in corso" : "");
  const prim = $("#nav-primary");
  prim.textContent = STEP_PRIMARY[id].label;
  prim.onclick = STEP_PRIMARY[id].run;
  const ready = applyReadiness(id);
  prim.disabled = busy || !ready;          // action inhibited if prerequisites unmet (or busy)
  prim.title = ready ? "" : "Completa prima il Build per questo passo";
}
$("#nav-back").addEventListener("click", () => {
  const i = STEPS.indexOf(currentStepId());
  if (i <= 0) { showView("projects"); loadProjects(); }
  else selectTab(STEPS[i - 1]);
});
$("#nav-next").addEventListener("click", () => {
  const i = STEPS.indexOf(currentStepId());
  if (i < STEPS.length - 1) selectTab(STEPS[i + 1]);
});

/* ---------------- review ---------------- */
async function loadScript() {
  try { $("#script-text").value = await API.readScript(current); }
  catch (e) { $("#script-text").value = ""; }
}
$("#reload-script").addEventListener("click", () => { loadScript(); setStatus("#script-status", "Ricaricato.", "ok"); });
async function saveAndBuild() {
  if (busy) return;
  setStatus("#script-status", "Salvataggio…", null);
  try {
    await API.saveScript(current, $("#script-text").value);
  } catch (e) {                                   // do NOT build stale text on a failed save
    setStatus("#script-status", "Errore nel salvataggio — build non avviata: " + ((e && e.message) || e), "err");
    return;
  }
  setStatus("#script-status", "Salvato. Avvio build.", null);
  selectTab("tab-build");
  startBuild();
}

/* ---------------- build ---------------- */
function appendLog(line, cls) {
  const pre = $("#build-log");
  if (pre.dataset.seeded === "1") { pre.textContent = ""; delete pre.dataset.seeded; }  // clear placeholder
  const atBottom = pre.scrollHeight - pre.clientHeight - pre.scrollTop < 40;   // measure BEFORE append
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = line + "\n";
  pre.appendChild(span);
  if (atBottom) pre.scrollTop = pre.scrollHeight;   // only auto-follow if the user was already at the end
}
// Weighted build progress. `build` runs these 5 stages (script is already done); start/end are
// cumulative %, `expect` is the typical seconds (measured) so the bar creeps smoothly inside a long
// stage and the elapsed counter keeps moving — a 4-min assemble never looks frozen.
const BUILD_PLAN = [
  { key: "voice", label: "Voce", start: 0, end: 3, expect: 12 },
  { key: "footage", label: "Footage", start: 3, end: 26, expect: 88 },
  { key: "captions", label: "Sottotitoli", start: 26, end: 28, expect: 4 },
  { key: "metadata", label: "Metadati", start: 28, end: 42, expect: 60 },
  { key: "assemble", label: "Montaggio", start: 42, end: 100, expect: 240 },
];
let _progTimer = null, _progStageStart = 0, _progPlan = null;

function _setBar(pct, stageLabel, elapsedS) {
  const fill = $("#build-progress-fill"), track = $("#build-progress-track");
  if (fill) fill.style.width = Math.max(0, Math.min(100, pct)).toFixed(1) + "%";
  if (track) track.setAttribute("aria-valuenow", Math.round(pct));
  if (stageLabel != null) $("#build-progress-stage").textContent = stageLabel;
  if (elapsedS != null) $("#build-progress-elapsed").textContent = elapsedS + "s";
}
function _stopProgTimer() { if (_progTimer) { clearInterval(_progTimer); _progTimer = null; } }
function showBuildProgress(show) {
  const el = $("#build-progress"); if (el) el.hidden = !show;
  if (!show) { _stopProgTimer(); _setBar(0, "—", 0); }
}
function onBuildStage(stage) {
  _stopProgTimer();
  _progPlan = BUILD_PLAN.find((p) => p.key === stage) || null;
  _progStageStart = Date.now();
  if (!_progPlan) return;
  _setBar(_progPlan.start, _progPlan.label, 0);
  _progTimer = setInterval(() => {
    const elapsed = (Date.now() - _progStageStart) / 1000;
    const frac = Math.min(elapsed / _progPlan.expect, 1);
    // creep toward (but never reach) the next stage's start, so completion comes from the next event
    const pct = _progPlan.start + (_progPlan.end - _progPlan.start) * frac * 0.97;
    _setBar(pct, _progPlan.label, Math.round(elapsed));
  }, 500);
}

async function startBuild() {
  if (busy) return;                  // inhibit duplicate build triggers
  setBusy(true);
  $("#build-state").textContent = "in corso…";
  setConn("build in corso", "active");
  announce("Build avviata");
  showBuildProgress(true); _setBar(0, "avvio…", 0);
  if (unsubscribeBuild) unsubscribeBuild();
  unsubscribeBuild = API.onBuildEvent((ev) => {
    if (ev.type === "log") appendLog(ev.line, classify(ev.line));
    else if (ev.type === "stage") { onBuildStage(ev.stage); announce("Stadio: " + ev.stage); refreshStatus(); }
    else if (ev.type === "done") { _stopProgTimer(); _setBar(100, "completato", null); $("#build-state").textContent = "completato ✓"; setConn("pronto", "done"); announce("Build completata"); }
    else if (ev.type === "error") { _stopProgTimer(); $("#build-state").textContent = "errore"; appendLog(ev.line || "errore", "err"); setConn("errore build", "fail"); announce("Build fallita"); }
  });
  try { await API.build(current); }
  catch (e) { appendLog(String(e), "err"); $("#build-state").textContent = "errore"; setConn("errore build", "fail"); _stopProgTimer(); }
  finally { setBusy(false); refreshStatus(); loadPreview(); }
}
function classify(line) {
  if (/error|fail|traceback/i.test(line)) return "err";
  if (/warning|warn/i.test(line)) return "warn";
  if (/done|ready|✓|completat/i.test(line)) return "ok";
  return null;
}

/* ---------------- preview ---------------- */
async function loadPreview() {
  if (currentStages.assemble !== "done") return;   // nothing to preview before a build
  for (const eng of ["kokoro"]) {   // Kokoro is the only engine (Chatterbox removed)
    try {
      const url = await API.videoUrl(current, eng);
      const v = $("#vid-" + eng);
      if (url && v) { v.src = url; v.load(); }
    } catch (e) {}
  }
  try {
    const m = await API.readMetadata(current);
    const dl = $("#meta-list");
    const yt = m.youtube || {}, tk = m.tiktok || {}, ig = m.instagram || {};
    dl.innerHTML = `
      <dt>YouTube</dt><dd>${escapeHtml(yt.title || "")}<br><span style="color:var(--muted)">${escapeHtml(yt.description || "")}</span></dd>
      <dt>TikTok</dt><dd>${escapeHtml(tk.caption || "")}</dd>
      <dt>Instagram</dt><dd>${escapeHtml(ig.caption || "")}</dd>
      <dt>Disclosure AI</dt><dd>${m.disclosure_ai ? "richiesta" : "non richiesta"}</dd>`;
  } catch (e) {}
}
$("#tab-preview").addEventListener("click", loadPreview);

/* ---------------- publish ---------------- */
async function doPublish(go) {
  if (busy) return;
  if (currentStages.assemble !== "done") { $("#publish-state").textContent = "completa prima il Build"; return; }
  const platforms = $$('input[name="platform"]:checked').map((c) => c.value);
  if (!platforms.length) { $("#publish-state").textContent = "seleziona almeno una piattaforma"; return; }
  if (go && !window.confirm("Pubblicare ORA su " + platforms.join(", ") + " via Postiz?\nVerrà postato pubblicamente.")) return;
  setBusy(true);
  $("#publish-state").textContent = go ? "pubblico…" : "dry run…";
  try {
    const res = await API.publish(current, platforms, go);
    $("#publish-plan").textContent = JSON.stringify(res.plan || res, null, 2);
    setStatus("#publish-state", go ? "inviato ✓" : "piano pronto (dry run)", "ok");
    announce(go ? "Pubblicazione inviata" : "Piano di pubblicazione pronto");
  } catch (e) {
    $("#publish-plan").textContent = String(e);
    setStatus("#publish-state", "errore", "err");
  } finally { setBusy(false); }
}
$("#publish-btn").addEventListener("click", () => doPublish(true));

/* ---------------- settings ---------------- */
async function loadSettings() {
  let c = {};
  try { c = await API.getConfig(); } catch (e) {}
  const set = (id, v) => { const el = $("#" + id); if (el && v != null) el.value = v; };
  set("set-language", (c.script || {}).language);
  set("set-model", (c.llm || {}).model);
  set("set-engine", (c.tts || {}).engine);
  set("set-primary", (c.tts || {}).primary);
  set("set-seconds", (c.script || {}).target_seconds);
  set("set-stt", (c.stt || {}).engine);
  const v = c.video || {}, pub = c.publish || {}, fn = c.funnel || {};
  $("#set-video").checked = !!v.prefer_video;
  $("#set-trans").checked = (v.transition || 0) > 0;
  $("#set-credits").checked = !!v.show_credits;
  const plats = pub.platforms || [];
  ["youtube", "tiktok", "instagram"].forEach((p) => { $("#set-pf-" + p).checked = plats.includes(p); });
  set("set-app-name", fn.app_name);
  set("set-app-url", fn.url);
  set("set-app-tag", fn.tagline);
  set("set-subtitle-language", (c.script || {}).subtitle_language || "");
  set("set-refine", String((c.script || {}).refine_passes ?? 1));
  set("set-music-source", v.music_source || "generate");
  set("set-music-mood", v.music_mood || "ethereal");
  $("#set-trim").checked = v.trim_silence !== false;
}
$("#settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const plats = ["youtube", "tiktok", "instagram"].filter((p) => $("#set-pf-" + p).checked);
  const patch = {
    llm: { model: $("#set-model").value },
    script: { language: $("#set-language").value, target_seconds: Number($("#set-seconds").value) || 75,
              subtitle_language: $("#set-subtitle-language").value || null,
              refine_passes: Number($("#set-refine").value) || 0 },
    tts: { engine: $("#set-engine").value, primary: $("#set-primary").value },
    stt: { engine: $("#set-stt").value },
    video: { prefer_video: $("#set-video").checked, transition: $("#set-trans").checked ? 0.4 : 0,
             show_credits: $("#set-credits").checked, trim_silence: $("#set-trim").checked,
             music_source: $("#set-music-source").value, music_mood: $("#set-music-mood").value },
    publish: { platforms: plats },
    funnel: { app_name: $("#set-app-name").value, url: $("#set-app-url").value, tagline: $("#set-app-tag").value },
  };
  $("#settings-status").textContent = "Salvataggio…";
  try {
    await API.setConfig(patch);
    setStatus("#settings-status", "Salvato ✓", "ok");
    announce("Impostazioni salvate");
  } catch (err) {
    setStatus("#settings-status", "Errore: " + ((err && err.message) || err), "err");
  }
});

/* ---------------- misc ---------------- */
$("#refresh-projects").addEventListener("click", loadProjects);
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

/* ---------------- init ---------------- */
setConn(REAL ? "locale · pronto" : "anteprima demo (mock)", REAL ? "done" : "active");
loadProjects();

/* ============================================================
   MOCK backend — lets the UI be previewed in a plain browser.
   ============================================================ */
function makeMock() {
  const projects = [
    { slug: "saturn-rings-pro", title: "Saturn's Rings Are Vanishing", stages: { script: "done", voice: "done", footage: "done", captions: "done", assemble: "done", metadata: "done" } },
    { slug: "andromeda-collision", title: "When Andromeda Hits Us", stages: { script: "done", voice: "done", footage: "partial", captions: "pending", assemble: "pending", metadata: "pending" } },
  ];
  let buildCb = null, newCb = null;
  return {
    listProjects: async () => projects,
    onNewEvent: (cb) => { newCb = cb; return () => { newCb = null; }; },
    newProject: async (topic) => {
      const slug = topic.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 28).replace(/^-|-$/g, "");
      const log = (line) => newCb && newCb({ type: "log", line });
      await new Promise((r) => setTimeout(r, 1500)); log("Generating Italian script…");
      await new Promise((r) => setTimeout(r, 1800)); log("Script refine pass 1/2 applied.");
      await new Promise((r) => setTimeout(r, 1800)); log("Script refine pass 2/2 applied.");
      await new Promise((r) => setTimeout(r, 900));  log("Script ready: (8 segments)");
      projects.unshift({ slug, title: topic, stages: { script: "done" } });
      return { slug, title: topic };
    },
    deleteProject: async (slug) => { const i = projects.findIndex((x) => x.slug === slug); if (i >= 0) projects.splice(i, 1); return true; },
    readScript: async () => "# Saturn's Rings Are Vanishing — Here's Why\n\n## 1\nNARRATION: Saturn's iconic rings aren't eternal — they're slowly disappearing.\nVISUAL: Cassini wide view of Saturn\nKEYWORDS: Saturn rings, Cassini\n\n## 2\nNARRATION: NASA calls it \"ring rain\".\nVISUAL: ice falling into Saturn\nKEYWORDS: ring rain\n\n## 9\nNARRATION: Want to capture the cosmos yourself? Get AstroStackerPro — link in the bio.\nVISUAL: App endcard\nKEYWORDS:\n",
    saveScript: async () => {},
    onBuildEvent: (cb) => { buildCb = cb; return () => { buildCb = null; }; },
    build: async () => {
      const lines = [
        ["▶ voice", "stage"], ["[kokoro] segment 1/9", "log"], ["[kokoro] segment 9/9", "log"],
        ["▶ footage", "stage"], ["Segment 1 ← NASA video 'Cassini's Infrared Saturn'", "log"],
        ["▶ captions", "stage"], ["captions[kokoro]: 164 words", "log"],
        ["▶ assemble", "stage"], ["[kokoro] → saturn-rings-pro.kokoro.mp4", "log"], ["Outputs ready", "log"],
        ["▶ metadata", "stage"], ["Metadata ready", "log"],
      ];
      for (const [text, type] of lines) {
        await new Promise((r) => setTimeout(r, 420));
        if (!buildCb) return;
        if (type === "stage") buildCb({ type: "stage", stage: text.replace("▶ ", "") });
        buildCb({ type: "log", line: text });
      }
      buildCb && buildCb({ type: "done" });
    },
    readMetadata: async () => ({ youtube: { title: "Saturn's Rings Vanishing: Why? 🌌", description: "Ring rain is draining Saturn's rings. Get AstroStackerPro: https://…" }, tiktok: { caption: "Saturn's rings are vanishing 🌌 #astronomy #space #saturn" }, instagram: { caption: "Saturn's rings are vanishing 🌌 #astronomy #astrostackerpro" }, disclosure_ai: false }),
    videoUrl: async () => "",
    publish: async (slug, platforms, go) => ({ plan: platforms.map((p) => ({ platform: p, caption: "…", dry_run: !go })) }),
    getConfig: async () => ({
      llm: { model: "gemma4:26b-mlx" },
      script: { language: "en", target_seconds: 75 },
      tts: { engine: "kokoro", primary: "kokoro" },
      stt: { engine: "parakeet" },
      video: { prefer_video: true, transition: 0.4, show_credits: true },
      publish: { platforms: ["youtube", "tiktok", "instagram"] },
      funnel: { app_name: "AstroStackerPro", url: "https://apps.apple.com/app/astrostackerpro", tagline: "Turn your phone into an astrophotography studio." },
    }),
    setConfig: async () => true,
  };
}
