#!/usr/bin/env bash
set -euo pipefail

echo "cwd=$(pwd)"
python3 --version
uv --version

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

python3 - <<'PY'
import importlib.util

modules = ["torch", "transformers", "vllm", "sglang"]
for name in modules:
    print(f"{name}={bool(importlib.util.find_spec(name))}")
PY
