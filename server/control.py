#!/usr/bin/env python3
"""Social Video AI — control plane (server-side "brain").

Holds the topic queue, creates the daily jobs, hands them to the Mac GPU worker, and — once a worker
returns a finished video — uploads it to Postiz and schedules it at the day's best time. Dependency-free
(stdlib http.server + sqlite3 + urllib), so the container is tiny and can't drift on dependencies.

Config via env (all optional except the token):
  CONTROL_TOKEN     shared secret the worker/admin must send (Authorization header). REQUIRED.
  POSTIZ_URL        Postiz public API base, e.g. http://postiz:5000/api  (default that)
  POSTIZ_TOKEN      Postiz API key (from Postiz → Settings → Developers)
  SVAI_COUNT        videos/day (3)          SVAI_PLATFORMS   tiktok,instagram
  SVAI_POST_TIMES   12:00,18:00,21:00       SVAI_TIMEZONE    Europe/Rome
  SVAI_GENERATE_HOUR  hour (0-23) the daily jobs are created (6)
  SVAI_PRIVACY      public|unlisted|private SVAI_DISCLOSE_AI 0|1
  SVAI_DB           /data/control.db        SVAI_VIDEO_DIR   /data/videos
  SVAI_HOST 0.0.0.0  SVAI_PORT 8770
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import postiz_client as pz

try:
    from zoneinfo import ZoneInfo
except Exception:  # noqa: BLE001
    ZoneInfo = None


def env(name, default=""):
    return os.environ.get(name, default)


CFG = {
    "token": env("CONTROL_TOKEN"),
    "postiz_url": env("POSTIZ_URL", "http://postiz:5000/api"),
    "postiz_token": env("POSTIZ_TOKEN"),
    "count": int(env("SVAI_COUNT", "3")),
    "platforms": [p.strip() for p in env("SVAI_PLATFORMS", "tiktok,instagram").split(",") if p.strip()],
    "post_times": [t.strip() for t in env("SVAI_POST_TIMES", "12:00,18:00,21:00").split(",") if t.strip()],
    "timezone": env("SVAI_TIMEZONE", "Europe/Rome"),
    "generate_hour": int(env("SVAI_GENERATE_HOUR", "6")),
    "privacy": env("SVAI_PRIVACY", "public"),
    "disclose_ai": env("SVAI_DISCLOSE_AI", "0") in ("1", "true", "yes"),
    "db": env("SVAI_DB", "/data/control.db"),
    "video_dir": env("SVAI_VIDEO_DIR", "/data/videos"),
    "host": env("SVAI_HOST", "0.0.0.0"),
    "port": int(env("SVAI_PORT", "8770")),
}


# --------------------------------------------------------------------------- time
def zone(tz):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz)
        except Exception:  # noqa: BLE001
            pass
    return timezone.utc


def iso_utc(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def post_slots(now, times, tz, count):
    """The next `count` posting datetimes (tz-aware, future-only), rolling into following days."""
    z = zone(tz)
    now = now.astimezone(z)
    parsed = []
    for t in times:
        try:
            hh, mm = (int(x) for x in str(t).split(":")[:2])
            if 0 <= hh < 24 and 0 <= mm < 60:
                parsed.append((hh, mm))
        except Exception:  # noqa: BLE001
            continue
    parsed = parsed or [(12, 0), (18, 0), (21, 0)]
    slots = []
    for day in range(0, 15):
        base = now + timedelta(days=day)
        for hh, mm in parsed:
            cand = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if cand > now:
                slots.append(cand)
                if len(slots) >= count:
                    return slots
    return slots


# --------------------------------------------------------------------------- store
SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT UNIQUE, added_at TEXT);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, topic TEXT, slot_utc TEXT, status TEXT, created_at TEXT,
  claimed_at TEXT, worker TEXT, error TEXT, posted_to TEXT, meta_json TEXT);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


class Store:
    def __init__(self, path, client_factory=None):
        self.path = path
        self.lock = threading.Lock()
        self.client_factory = client_factory or (lambda: pz.PostizClient(CFG["postiz_url"], CFG["postiz_token"]))
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    # --- topics ---
    def add_topics(self, topics):
        added = 0
        with self.lock:
            for t in topics:
                t = (t or "").strip()
                if not t:
                    continue
                try:
                    self._conn.execute("INSERT INTO topics(topic, added_at) VALUES(?,?)", (t, self._now()))
                    added += 1
                except sqlite3.IntegrityError:
                    pass
            self._conn.commit()
        return added

    def topics(self):
        return [r["topic"] for r in self._conn.execute("SELECT topic FROM topics ORDER BY id")]

    def pop_topics(self, n):
        with self.lock:
            rows = self._conn.execute("SELECT id, topic FROM topics ORDER BY id LIMIT ?", (n,)).fetchall()
            for r in rows:
                self._conn.execute("DELETE FROM topics WHERE id=?", (r["id"],))
            self._conn.commit()
        return [r["topic"] for r in rows]

    # --- jobs ---
    def create_job(self, topic, slot_utc):
        jid = uuid.uuid4().hex[:12]
        with self.lock:
            self._conn.execute(
                "INSERT INTO jobs(id, topic, slot_utc, status, created_at) VALUES(?,?,?,?,?)",
                (jid, topic, slot_utc, "pending", self._now()))
            self._conn.commit()
        return jid

    def plan_day(self, count=None, now=None):
        """Pop `count` topics and create a job each, at the next post slots. Returns created jobs."""
        count = count or CFG["count"]
        topics = self.pop_topics(count)
        if not topics:
            return []
        slots = post_slots(now or datetime.now(timezone.utc), CFG["post_times"], CFG["timezone"], len(topics))
        return [{"id": self.create_job(t, iso_utc(s)), "topic": t, "slot_utc": iso_utc(s)}
                for t, s in zip(topics, slots)]

    def claim(self, worker):
        with self.lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE status='pending' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                return None
            self._conn.execute("UPDATE jobs SET status='in_progress', worker=?, claimed_at=? WHERE id=?",
                               (worker, self._now(), row["id"]))
            self._conn.commit()
            return {"id": row["id"], "topic": row["topic"], "slot_utc": row["slot_utc"]}

    def save_meta(self, jid, meta_json):
        with self.lock:
            self._conn.execute("UPDATE jobs SET meta_json=? WHERE id=?", (meta_json, jid))
            self._conn.commit()

    def fail(self, jid, error):
        with self.lock:
            self._conn.execute("UPDATE jobs SET status='failed', error=? WHERE id=?", (str(error)[:500], jid))
            self._conn.commit()

    def get(self, jid):
        r = self._conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return dict(r) if r else None

    def finalize(self, jid, video_path):
        """A worker returned the finished video: upload to Postiz + schedule at the slot, for every
        connected platform. If no channel is connected, the video is kept and marked done (posted_to=[])."""
        job = self.get(jid)
        if not job:
            return {"error": "unknown job"}
        meta = json.loads(job.get("meta_json") or "{}")
        posted = []
        try:
            client = self.client_factory()
            discovered = pz.discover(client)
            targets = [p for p in CFG["platforms"] if pz.canon(p) in discovered]
            if targets:
                media = client.upload(video_path)
                for p in targets:
                    settings = pz.settings_for(p, meta, disclose_ai=CFG["disclose_ai"], privacy=CFG["privacy"])
                    client.create_post(discovered[pz.canon(p)], pz.caption_for(p, meta), media,
                                       settings, job["slot_utc"])
                    posted.append(pz.canon(p))
        except Exception as e:  # noqa: BLE001 — keep the video; surface the error, don't crash the server
            with self.lock:
                self._conn.execute("UPDATE jobs SET status='uploaded', error=?, posted_to=? WHERE id=?",
                                   (f"publish failed: {e}"[:500], json.dumps(posted), jid))
                self._conn.commit()
            return {"status": "uploaded", "posted_to": posted, "error": str(e)}
        with self.lock:
            self._conn.execute("UPDATE jobs SET status='done', posted_to=?, error=NULL WHERE id=?",
                               (json.dumps(posted), jid))
            self._conn.commit()
        return {"status": "done", "posted_to": posted}

    def status(self):
        counts = {s: 0 for s in ("pending", "in_progress", "uploaded", "done", "failed")}
        for r in self._conn.execute("SELECT status, COUNT(*) c FROM jobs GROUP BY status"):
            counts[r["status"]] = r["c"]
        recent = [dict(r) for r in self._conn.execute(
            "SELECT id, topic, slot_utc, status, posted_to, error, worker FROM jobs ORDER BY created_at DESC LIMIT 30")]
        topics = self.topics()
        return {"queue": len(topics), "topics": topics, "jobs": counts, "recent": recent,
                "config": {k: CFG[k] for k in ("count", "platforms", "post_times", "timezone", "generate_hour")}}

    def meta_get(self, k):
        r = self._conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r["v"] if r else None

    def meta_set(self, k, v):
        with self.lock:
            self._conn.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=?", (k, v, v))
            self._conn.commit()


# --------------------------------------------------------------------------- http
STORE: Store | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quieter logs
        pass

    def _send(self, code, obj=None, ctype="application/json", raw=None):
        body = raw if raw is not None else (json.dumps(obj).encode() if obj is not None else b"")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _authed(self):
        return bool(CFG["token"]) and self.headers.get("Authorization", "") == CFG["token"]

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n) if n else b""

    def _need_auth(self):
        if not self._authed():
            self._send(401, {"error": "unauthorized"})
            return True
        return False

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p == "/healthz":
            return self._send(200, {"ok": True})
        if p == "/api/status":
            return self._send(200, STORE.status())
        if p == "/api/topics":
            if self._need_auth():
                return
            return self._send(200, {"topics": STORE.topics()})
        if p == "/" or p == "/index.html":
            return self._send(200, raw=_ui_html().encode(), ctype="text/html; charset=utf-8")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?", 1)[0]
        if p == "/api/topics":
            if self._need_auth():
                return
            data = json.loads(self._body() or b"{}")
            n = STORE.add_topics(data.get("topics", []))
            return self._send(200, {"added": n, "queue": len(STORE.topics())})
        if p == "/api/jobs/plan":
            if self._need_auth():
                return
            data = json.loads(self._body() or b"{}")
            return self._send(200, {"created": STORE.plan_day(count=data.get("count"))})
        if p == "/api/jobs/claim":
            if self._need_auth():
                return
            data = json.loads(self._body() or b"{}")
            job = STORE.claim(data.get("worker", "worker"))
            return self._send(200, job) if job else self._send(204)
        if p.startswith("/api/jobs/") and p.endswith("/metadata"):
            if self._need_auth():
                return
            jid = p.split("/")[3]
            STORE.save_meta(jid, self._body().decode() or "{}")
            return self._send(200, {"ok": True})
        if p.startswith("/api/jobs/") and p.endswith("/fail"):
            if self._need_auth():
                return
            jid = p.split("/")[3]
            STORE.fail(jid, json.loads(self._body() or b"{}").get("error", "worker reported failure"))
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})

    def do_PUT(self):
        p = self.path.split("?", 1)[0]
        if p.startswith("/api/jobs/") and p.endswith("/video"):
            if self._need_auth():
                return
            jid = p.split("/")[3]
            if not STORE.get(jid):
                return self._send(404, {"error": "unknown job"})
            os.makedirs(CFG["video_dir"], exist_ok=True)
            path = os.path.join(CFG["video_dir"], f"{jid}.mp4")
            with open(path, "wb") as f:
                f.write(self._body())
            return self._send(200, STORE.finalize(jid, path))
        return self._send(404, {"error": "not found"})


def _ui_html():
    return """<!doctype html><html lang=it><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Social Video AI — Control</title>
<style>
:root{--bg:#fff;--fg:#1a1a2e;--mut:#667;--line:#e3e3ee;--card:#f7f7fb;--accent:#5b5bd6}
@media(prefers-color-scheme:dark){:root{--bg:#14141c;--fg:#e8e8f0;--mut:#9a9ab0;--line:#2a2a3a;--card:#1d1d28;--accent:#8f8ff0}}
*{box-sizing:border-box}body{font:14px/1.5 system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
.wrap{max-width:980px;margin:0 auto;padding:1.5rem}h1{font-size:1.25rem;margin:.2rem 0 1rem}
.row{display:flex;gap:1rem;flex-wrap:wrap}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1rem;flex:1;min-width:220px}
.mut{color:var(--mut)}.big{font-size:1.6rem;font-weight:700}
table{border-collapse:collapse;width:100%;margin-top:.5rem}td,th{border-bottom:1px solid var(--line);padding:.45rem .5rem;text-align:left;font-size:13px}
th{color:var(--mut);font-weight:600}
.pill{display:inline-block;padding:.05rem .5rem;border-radius:99px;font-size:12px;font-weight:600}
.s-pending{background:#8883}.s-in_progress{background:#e9a23b33;color:#c8791a}.s-done{background:#2ecc7133;color:#1a9e54}
.s-uploaded{background:#5b5bd633;color:var(--accent)}.s-failed{background:#e5484d33;color:#e5484d}
input,textarea,button{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:.5rem}
button{background:var(--accent);color:#fff;border:0;cursor:pointer;font-weight:600}button.ghost{background:transparent;color:var(--fg);border:1px solid var(--line)}
textarea{width:100%;min-height:70px;resize:vertical}.tok{width:100%;font-family:monospace}#msg{min-height:1.2em}.warn{color:#e5484d}
</style><div class=wrap>
<h1>🛰️ Social Video AI — Control</h1>
<div class=row id=stats></div>
<p class=mut id=cfg></p>
<div class=row style="align-items:flex-start">
  <div class=card style="flex:1.3">
    <b>Coda topic</b> <span class=mut id=qn></span>
    <ol id=queue class=mut style="margin:.4rem 0 .8rem;padding-left:1.2rem"></ol>
    <textarea id=add placeholder="Aggiungi topic (uno per riga)…"></textarea>
    <div style="display:flex;gap:.5rem;margin-top:.5rem"><button onclick=addTopics()>Aggiungi alla coda</button>
    <button class=ghost onclick=planNow()>Genera i job di oggi ora</button></div>
  </div>
  <div class=card style="flex:.7">
    <b>Accesso</b><p class=mut style=margin:.3rem>Incolla il CONTROL_TOKEN per gestire coda e job (resta solo nel browser).</p>
    <input class=tok id=tok placeholder="CONTROL_TOKEN" oninput="localStorage.svai_tok=this.value">
    <p id=msg class=mut></p>
  </div>
</div>
<div class=card style=margin-top:1rem><b>Job recenti</b>
<table><thead><tr><th>topic</th><th>quando (UTC)</th><th>stato</th><th>postato</th><th>note</th></tr></thead>
<tbody id=jobs></tbody></table></div>
<p class=mut style=margin-top:1rem>aggiornamento automatico ogni 5s</p></div>
<script>
const $=s=>document.querySelector(s);
$('#tok').value=localStorage.svai_tok||'';
function tok(){return localStorage.svai_tok||''}
async function api(path,body){const r=await fetch(path,{method:'POST',headers:{'Authorization':tok(),'Content-Type':'application/json'},body:JSON.stringify(body||{})});
 if(r.status===401){$('#msg').innerHTML='<span class=warn>token mancante o errato</span>';return null}
 $('#msg').textContent='ok';return r.json()}
async function addTopics(){const t=$('#add').value.split('\\n').map(s=>s.trim()).filter(Boolean);if(!t.length)return;
 const r=await api('/api/topics',{topics:t});if(r){$('#add').value='';load()}}
async function planNow(){const r=await api('/api/jobs/plan',{});if(r){load()}}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function load(){const d=await(await fetch('/api/status')).json();
 $('#stats').innerHTML=[['coda',d.queue],['in coda job',d.jobs.pending],['in corso',d.jobs.in_progress],['fatti',d.jobs.done],['caricati',d.jobs.uploaded],['falliti',d.jobs.failed]]
   .map(([k,v])=>`<div class=card style=min-width:120px><div class=mut>${k}</div><div class=big>${v}</div></div>`).join('');
 $('#cfg').textContent=`${d.config.count} video/giorno → ${d.config.platforms.join(', ')} @ ${d.config.post_times.join(' / ')} (${d.config.timezone}) · genera alle ${d.config.generate_hour}:00`;
 $('#qn').textContent=`(${d.queue})`;
 $('#queue').innerHTML=d.topics.map(t=>`<li>${esc(t)}</li>`).join('')||'<li class=mut>vuota</li>';
 $('#jobs').innerHTML=d.recent.map(j=>`<tr><td>${esc(j.topic)}</td><td>${(j.slot_utc||'').replace('T',' ').replace('.000Z','')}</td>
   <td><span class="pill s-${j.status}">${j.status}</span></td><td>${esc(j.posted_to)||'—'}</td>
   <td class=warn>${esc(j.error)||''}</td></tr>`).join('')||'<tr><td colspan=5 class=mut>nessun job</td></tr>';
}
load();setInterval(load,5000);
</script></html>"""


# --------------------------------------------------------------------------- scheduler
def scheduler_loop(store, stop):
    """Create the day's jobs once, when the local hour hits generate_hour."""
    while not stop.is_set():
        try:
            now = datetime.now(zone(CFG["timezone"]))
            today = now.strftime("%Y-%m-%d")
            if now.hour == CFG["generate_hour"] and store.meta_get("last_plan") != today:
                created = store.plan_day(now=datetime.now(timezone.utc))
                store.meta_set("last_plan", today)
                print(f"[scheduler] {today}: created {len(created)} job(s)", flush=True)
        except Exception as e:  # noqa: BLE001 — never let the scheduler thread die
            print(f"[scheduler] error: {e}", flush=True)
        stop.wait(60)


def main():
    global STORE
    if not CFG["token"]:
        print("[warn] CONTROL_TOKEN is empty — all /api mutations will be rejected. Set it.", flush=True)
    STORE = Store(CFG["db"])
    stop = threading.Event()
    threading.Thread(target=scheduler_loop, args=(STORE, stop), daemon=True).start()
    srv = ThreadingHTTPServer((CFG["host"], CFG["port"]), Handler)
    print(f"[control] listening on {CFG['host']}:{CFG['port']} — Postiz {CFG['postiz_url']}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
