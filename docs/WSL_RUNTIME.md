# WSL Runtime Path

Last verified: 2026-04-22

This desktop has:

- Ubuntu on WSL2
- GPU visibility inside WSL2
- Python 3.12
- `uv`

Current package state inside WSL at verification time:

- `torch`: installed in `.venv-wsl-vllm`
- `transformers`: installed in `.venv-wsl-vllm`
- `vllm`: installed in `.venv-wsl-vllm`
- `flash-linear-attention`: installable into `.venv-wsl-vllm` via `scripts/wsl/bootstrap_fla.sh`
- `sglang`: not installed

## Why WSL Is The Right Path

- vLLM and SGLang are the strongest Linux-first serving targets for the next real model-backed step
- this repo now includes `openai_compatible_http`, so the benchmark and evaluation surface does not need to care whether the server is vLLM, SGLang, llama.cpp, or another compatible runtime

## Check The WSL Environment

From Windows PowerShell:

```powershell
wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/check_env.sh"
```

## Bootstrap vLLM In WSL

From Windows PowerShell:

```powershell
wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/bootstrap_vllm.sh"
```

This creates `.venv-wsl-vllm/` in the repo and installs `vllm` with `uv`.

## Bootstrap FLA In WSL

From Windows PowerShell:

```powershell
wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/bootstrap_fla.sh"
```

This installs `flash-linear-attention` into `.venv-wsl-vllm/`, which is the current runtime used by the `gated_deltanet_fla` baseline wrapper.

## Start A vLLM Server In WSL

Example:

```powershell
wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/serve_vllm.sh Qwen/Qwen3-4B 8000"
```

The repo-side probe is then:

```powershell
uv run typhon inspect-inference-backend --backend openai_compatible_http --base-url http://localhost:8000/v1
```

And the first evaluation pass is:

```powershell
uv run typhon evaluate-inference-backend --backend openai_compatible_http --model Qwen/Qwen3-4B --benchmark longbench_v2 --sample-source local --base-url http://localhost:8000/v1
```

## Start An SGLang Server In WSL

If `sglang` is installed in the active WSL Python environment:

```powershell
wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/serve_sglang.sh Qwen/Qwen3-4B 30000"
```

Probe it from Windows:

```powershell
uv run typhon inspect-inference-backend --backend openai_compatible_http --base-url http://localhost:30000/v1
```

## Notes

- `serve_vllm.sh` uses `--generation-config vllm` so server defaults are controlled by vLLM rather than silently inheriting a model repo generation config
- `openai_compatible_http` expects a `/v1` endpoint and normalizes the base URL automatically
- the `gated_deltanet_fla` baseline currently runs through WSL because the Windows uv environment does not have a usable Torch/FLA stack
