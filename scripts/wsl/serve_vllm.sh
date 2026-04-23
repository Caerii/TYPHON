#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model-id> [port]"
  exit 1
fi

MODEL_ID="$1"
PORT="${2:-8000}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-wsl-vllm"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "missing ${VENV_DIR}; run scripts/wsl/bootstrap_vllm.sh first"
  exit 1
fi

source "${VENV_DIR}/bin/activate"
exec vllm serve "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype auto \
  --generation-config vllm
