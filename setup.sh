#!/usr/bin/env bash
# AUT Video Pipeline — one-shot setup for macOS / Apple Silicon. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

echo "▶ AUT Video Pipeline setup"
VENV_PY=".venv/bin/python"

# 1) System tools via Homebrew (ffmpeg required; espeak-ng helps Kokoro's g2p)
for pkg in ffmpeg espeak-ng; do
  if brew list "$pkg" >/dev/null 2>&1; then
    echo "✓ $pkg present"
  else
    echo "Installing $pkg…"; brew install "$pkg"
  fi
done

# 2) Python 3.11 venv via uv
if ! command -v uv >/dev/null 2>&1; then
  echo "✗ uv not found. Install: brew install uv"; exit 1
fi
if [ ! -d ".venv" ]; then
  echo "Creating .venv (Python 3.11)…"; uv venv --python 3.11 .venv
else
  echo "✓ .venv present"
fi

# 3) Python deps (core + AI). Lazy imports mean a partial install still runs the rest.
echo "Installing core + AI deps (this pulls torch — a few GB)…"
for ex in ai parakeet whisper chatterbox; do
  echo "  .[$ex]"
  uv pip install --python "$VENV_PY" -e ".[$ex]" || echo "  ⚠ .[$ex] failed (continuing)"
done

# 4) LLM model for Ollama
if command -v ollama >/dev/null 2>&1; then
  if ollama list | grep -q "qwen3:14b"; then
    echo "✓ qwen3:14b present"
  else
    echo "Pulling qwen3:14b (~9GB)… (Ctrl-C to skip; you can use llama3 meanwhile)"
    ollama pull qwen3:14b || echo "  (skipped — set llm.model: llama3 in config.yaml to start now)"
  fi
else
  echo "⚠ ollama not found — install from https://ollama.com"
fi

# 5) Local config files
[ -f config.yaml ] || cp config.example.yaml config.yaml
[ -f .env ] || cp .env.example .env

cat <<'EOF'

✅ Setup complete.
   Activate venv:  source .venv/bin/activate
   First video:    avp new saturn-rings --topic "Why Saturn's rings are disappearing"
                   # edit projects/saturn-rings/script.md, then:
                   avp build saturn-rings
EOF
