#!/usr/bin/env bash
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
export PYTHONPATH="/Users/ste/Desktop/Progetti/AUT_VIDEO_PIPELINE/src"
cd "/Users/ste/Desktop/Progetti/AUT_VIDEO_PIPELINE"
if [ "$(date +%u)" = "7" ] && [ "$(date +%H)" -ge 20 ]; then
  "/Users/ste/Desktop/Progetti/AUT_VIDEO_PIPELINE/.venv/bin/avp" report --days 7 --config "/Users/ste/Desktop/Progetti/AUT_VIDEO_PIPELINE/config.yaml"
else
  "/Users/ste/Desktop/Progetti/AUT_VIDEO_PIPELINE/.venv/bin/avp" report --snapshot --config "/Users/ste/Desktop/Progetti/AUT_VIDEO_PIPELINE/config.yaml"
fi
