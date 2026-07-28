# Daily automation (`avp auto`)

Generate N videos a day from a topic queue and schedule them to Postiz at your best times — unattended.

## How it works
- **Topics** come from a queue file (`projects/topics.txt` by default). When it drops below
  `auto.refill_threshold`, the LLM proposes fresh, deduped topics (never repeating a past project).
- **`avp auto`** pops `auto.count` topics, generates + builds each video, then **schedules** each to
  Postiz at the next free slot from `auto.post_times` (rolling into the next day as slots fill).
- **Publishing is honest**: it only posts to a platform whose channel is actually connected in Postiz.
  Until TikTok/Instagram are connected, the videos are still built and left ready — nothing is posted.

## Commands
```bash
avp topics --add "Topic one" "Topic two"   # seed / grow the queue
avp topics --refill                        # top the queue up via the LLM now
avp topics                                 # show the queue
avp auto --dry-run                         # show today's plan (topics + slots), generate nothing
avp auto --no-publish                      # build the videos but don't schedule them
avp auto                                   # the real daily run
```
Tune in `config.yaml` under `auto:` — `count`, `platforms`, `post_times`, `timezone`, `theme`, queue.

## Run it every day (launchd)
```bash
deploy/auto/install.sh 8 0     # runs daily at 08:00 local (pick a time BEFORE your first post slot)
deploy/auto/run.sh             # run once now to test
```
`install.sh` writes `~/Library/LaunchAgents/com.astrostacker.auto.plist`; logs land in
`projects/_auto/`. Uninstall: `launchctl unload …/com.astrostacker.auto.plist && rm` it.

## Requirements (read these)
- **Apple Silicon** — the generation models are MLX/Metal. A Linux cloud VM will NOT run them; use a
  **Mac mini** for the always-on host.
- **Ollama running** (script/metadata/topics) and, for publishing, **Postiz up** (colima). `run.sh`
  best-effort starts colima; keep Ollama running (menu-bar app or a login service).
- **The Mac must be awake** at the run time. On a laptop that sleeps it fires late (launchd runs the
  missed job on wake); on an always-on Mac mini it's exact. Set Energy settings to prevent sleep, or
  wrap in `caffeinate`.
- **Cost**: `count` videos ≈ `count × ~6 min` of compute per day, plus disk for each project.

## Best times
`post_times` ships with sensible defaults (12:00 / 18:00 / 21:00) but **the best times depend on YOUR
audience** — they are not universal. Start here, then tune from each platform's analytics once you have
a few weeks of data.
