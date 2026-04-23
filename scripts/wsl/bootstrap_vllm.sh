#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-wsl-vllm"

cd "${ROOT_DIR}"
uv venv "${VENV_DIR}" --python 3.12
source "${VENV_DIR}/bin/activate"
uv pip install vllm

echo "vLLM environment ready at ${VENV_DIR}"
