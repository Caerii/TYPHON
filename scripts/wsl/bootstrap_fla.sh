#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${1:-$ROOT_DIR/.venv-wsl-vllm}"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "Expected a Python environment at $VENV_PATH/bin/python"
  exit 1
fi

cd "$ROOT_DIR"
uv pip install --python "$VENV_PATH/bin/python" flash-linear-attention
