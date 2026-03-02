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
mkdir -p "$HF_HOME"

CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch1}"

if [ ! -x "$CONDA_BIN" ]; then
  echo "Conda not found at $CONDA_BIN"
  exit 1
fi

if ! "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
  echo "Conda env '$CONDA_ENV_NAME' is not available"
  exit 1
fi

exec "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV_NAME" \
  python -m streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
