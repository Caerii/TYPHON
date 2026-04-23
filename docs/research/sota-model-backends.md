# SOTA Model Backend Note

Last reviewed: 2026-04-22

## Main Takeaways

### 1. vLLM and SGLang are the strongest general-purpose serving targets for later scale-up

- vLLM exposes an OpenAI-compatible server and is the cleanest path for heavier Linux-first serving.
- SGLang is also strong on NVIDIA GPUs and aligns with the same HTTP abstraction.

Practical implication:

- on this Windows RTX 3080 workstation, these are best treated as WSL-hosted targets

### 2. The repo should stabilize around one inference abstraction, not one serving stack

- LM Studio is already working locally and is the most useful live path today.
- OpenAI-compatible HTTP keeps the evaluation surface stable across LM Studio, vLLM, SGLang, llama.cpp, and similar servers.

Practical implication:

- backend abstraction is more important than committing early to one server implementation

### 3. A 10 GB RTX 3080 forces discipline in default model choice

Small and mid-sized instruction models remain the realistic local default. Multi-model hosting or very large checkpoints are the wrong default on this machine.

Practical implication:

- keep one hosted model active at a time for canonical live evaluation
- treat larger local models as explicit experiments, not the baseline workflow

## Recommendation For TYPHON

Short term:

- keep `lmstudio_local` as the primary live backend
- keep `openai_compatible_http` as the generic integration layer
- keep `extractive_heuristic` as a deterministic fallback and control condition

Medium term:

- expand WSL-backed `openai_compatible_http` use for vLLM or SGLang
- do not let backend experimentation fork the benchmark and evaluation interfaces

## Sources

- Ollama quickstart: https://docs.ollama.com/quickstart
- Ollama API: https://docs.ollama.com/api
- vLLM installation: https://docs.vllm.ai/en/latest/getting_started/installation.html
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- SGLang install: https://docs.sglang.ai/get_started/install.html
- llama.cpp repository: https://github.com/ggml-org/llama.cpp
