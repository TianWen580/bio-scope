#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  echo "DASHSCOPE_API_KEY is required in .env or environment"
  exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/models/hf_cache}"
if ! mkdir -p "$HF_HOME" 2>/dev/null; then
  echo "HF_HOME '$HF_HOME' is not writable; falling back to project-local cache"
  export HF_HOME="$ROOT_DIR/models/hf_cache"
  mkdir -p "$HF_HOME"
fi

CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-benchmark}"
USE_CONDA="${USE_CONDA:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ "$USE_CONDA" != "0" ] && [ -x "$CONDA_BIN" ]; then
  if "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
    exec "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV_NAME" \
      python -m streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
  fi
fi

if [ "$USE_CONDA" = "1" ]; then
  echo "Conda mode requested but unavailable. Check CONDA_BIN and CONDA_ENV_NAME."
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable '$PYTHON_BIN' not found"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
  echo "Streamlit is not installed in '$PYTHON_BIN'. Run: python -m pip install -r requirements.txt"
  exit 1
fi

echo "Conda runtime unavailable, using local Python runtime ($PYTHON_BIN)."
exec "$PYTHON_BIN" -m streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
