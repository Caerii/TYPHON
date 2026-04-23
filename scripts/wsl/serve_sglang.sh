#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model-id> [port]"
  exit 1
fi

MODEL_ID="$1"
PORT="${2:-30000}"

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${PORT}"
