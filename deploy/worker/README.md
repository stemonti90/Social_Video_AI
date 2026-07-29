# Mac GPU worker (always-on)

The worker (`avp worker`) claims jobs from the control server, renders them on the M5, and uploads the
finished video back. It only needs **Ollama running** on the Mac.

## Run it now (manual)
```bash
deploy/worker/install.sh        # writes deploy/worker/.env (fetches the control token over SSH)
deploy/worker/run-worker.sh     # runs the polling loop in this terminal (Ctrl-C to stop)
```
`.env` holds `AVP_CONTROL_URL` + `AVP_CONTROL_TOKEN` (gitignored). You can also pass them as flags:
`avp worker --server http://192.168.1.184:8770 --token <CONTROL_TOKEN>`.

## Always-on across reboots — the macOS TCC caveat
`install.sh` also installs a launchd agent, **but on this machine it can't start**: the project lives
under `~/Desktop`, which macOS protects with TCC, and a LaunchAgent is denied read access to files
there (`Operation not permitted`, exit 126) — even with `LimitLoadToSessionType Aqua` or an explicit
`/bin/bash`. The worker itself is fine; only the unattended launchd context is blocked.

Two clean fixes (pick one when you want true 24/7):
1. **Grant Full Disk Access to `/bin/bash`** — System Settings → Privacy & Security → Full Disk Access →
   `+` → press ⌘⇧G, type `/bin/bash`, add it. Then re-run `install.sh`. (Broad, but a single manual step.)
2. **Move the project out of `~/Desktop`** (e.g. `~/Developer/AUT_VIDEO_PIPELINE`) — the cleanest fix.
   Note: the Electron app hardcodes the Desktop path, so it would need its `ROOT` updated + a rebuild.

Until then, run `run-worker.sh` by hand (e.g. in a `tmux`/terminal window) when you want the worker up.
It isn't urgent: nothing publishes until the social channels are connected anyway.
