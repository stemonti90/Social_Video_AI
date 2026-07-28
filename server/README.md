# Social Video AI — control plane (server "brain")

Runs on the always-on server. Holds the topic queue, creates the daily jobs, hands them to the Mac GPU
worker, and once a worker returns a finished video, uploads it to Postiz and schedules it at the day's
best time. **Dependency-free** (stdlib `http.server` + `sqlite3` + `urllib`) so the container is tiny.

- `control.py` — the service (job store, scheduler, HTTP API, status UI).
- `postiz_client.py` — slim Postiz client (mirrors the verified contract in `../src/avp/publish.py`).
- `test_control.py` — `PYTHONPATH=server python -m unittest test_control`.

## Run
```bash
CONTROL_TOKEN=<shared-secret> POSTIZ_URL=http://192.168.1.184:4007/api POSTIZ_TOKEN=<postiz-key> \
  PYTHONPATH=server python server/control.py
```
Config via env (see the top of `control.py`): `SVAI_COUNT`, `SVAI_PLATFORMS`, `SVAI_POST_TIMES`,
`SVAI_TIMEZONE`, `SVAI_GENERATE_HOUR`, `SVAI_PRIVACY`, `SVAI_DISCLOSE_AI`, `SVAI_DB`, `SVAI_VIDEO_DIR`,
`SVAI_HOST`, `SVAI_PORT`.

## API (all `/api/*` mutations need `Authorization: <CONTROL_TOKEN>`)
| | |
|---|---|
| `GET /healthz` | liveness |
| `GET /` | status web UI |
| `GET /api/status` | queue + job counts + recent (public, read-only) |
| `POST /api/topics` `{topics:[…]}` | grow the queue |
| `POST /api/jobs/plan` `{count?}` | create today's jobs now (the scheduler also does this daily) |
| `POST /api/jobs/claim` `{worker}` | worker claims the next pending job → `{id,topic,slot_utc}` (204 if none) |
| `POST /api/jobs/{id}/metadata` | worker posts the video's metadata.json |
| `PUT  /api/jobs/{id}/video` | worker uploads the finished mp4 → server schedules it to Postiz |
| `POST /api/jobs/{id}/fail` `{error}` | worker reports a failure |

Publishing is honest: a job's video only posts to a platform whose channel is connected in Postiz;
otherwise the video is kept and the job is marked done with `posted_to: []` (ready for when channels
connect).

## Where this fits
Stage 1 (this) = the control plane. Stage 2 = `avp worker` on the Mac (claims jobs, renders on the M5,
uploads back). Stage 3 = Dockerfile + compose to run this on the server next to Postiz. Stage 4 = richer
UI. The heavy generation stays on Apple Silicon (MLX/Metal); this server only orchestrates + publishes.
