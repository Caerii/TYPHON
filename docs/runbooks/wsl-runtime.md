# WSL Runtime Path

Last verified: 2026-04-22

This workstation has:

- Ubuntu on WSL2
- GPU visibility inside WSL2
- Python 3.12
- `uv`

## Why WSL Exists In This Repo

Some runtimes and baselines are Linux-first in practice on this machine. WSL gives the repo a reproducible Linux GPU path without changing the Windows-side control plane.

Use WSL for:

- `flash-linear-attention`
- Gated DeltaNet baseline execution
- vLLM and SGLang experiments

Use the Windows process for:

- `uv run typhon ...`
- LM Studio-backed local evaluation
- orchestration and artifact writing

## WSL Checks

From PowerShell:

```powershell
wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/check_env.sh"
```

## Bootstrap Commands

vLLM environment:

```powershell
wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/bootstrap_vllm.sh"
```

FLA in the same WSL environment:

```powershell
wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/bootstrap_fla.sh"
```

## Serving Examples

Start vLLM:

```powershell
wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/serve_vllm.sh Qwen/Qwen3-4B 8000"
```

Probe from Windows:

```powershell
uv run typhon inspect-inference-backend --backend openai_compatible_http --base-url http://localhost:8000/v1
```

Start SGLang:

```powershell
wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/serve_sglang.sh Qwen/Qwen3-4B 30000"
```

## Gated DeltaNet Baseline

The `gated_deltanet_fla` baseline executes through WSL because the practical Torch and FLA runtime is there, not in the Windows Python environment.

Bootstrap:

```powershell
wsl.exe bash -lc "cd /mnt/f/Github/TYPHON && scripts/wsl/bootstrap_fla.sh"
```

Run from the normal repo CLI:

```powershell
uv run typhon run-baseline --baseline gated_deltanet_fla --benchmark longbench --sample-source local --sample-limit 1
```

## Notes

- Keep the control plane stable. The Windows CLI should not need to change when the Linux-side runtime changes.
- Prefer OpenAI-compatible HTTP backends for serving stacks and thin process wrappers only where direct model execution is required.
- Treat WSL runtime setup as operational infrastructure, not as a place to store ad hoc project logic.
