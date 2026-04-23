# SOTA Model Backend Note

Last reviewed: 2026-04-22

## What The Sources Say

### 1. For high-throughput serving, the strongest general-purpose research backends are vLLM and SGLang

- vLLM exposes an OpenAI-compatible server and supports multiple hardware platforms, but its official installation documentation emphasizes CUDA/Linux deployment in the common path.
- SGLang documents strong NVIDIA GPU support and recommends `uv` for installation, but its documented install path is still a heavyweight runtime stack.

Implication:

- these are the right targets for a later scale-up path
- they are best treated as WSL-hosted targets on this Windows RTX 3080 workstation

### 2. For this machine, Ollama and llama.cpp are the pragmatic first local model backends

- Ollama officially supports Windows and exposes a stable local API at `http://localhost:11434/api`
- the local Ollama API supports chat/generate, model listing, and local no-auth access
- llama.cpp explicitly supports CUDA, quantized inference, CPU+GPU hybrid inference, and an OpenAI-compatible `llama-server`

Implication:

- the repo should first target an OpenAI-compatible local inference abstraction
- the first concrete implementation should be `ollama_local` because Ollama is already installed on this machine
- `openai_compatible_http` should unify WSL-hosted vLLM/SGLang, llama.cpp, and Ollama-compatible servers behind one interface

### 2a. WSL changes the practical backend order on this specific desktop

- Ubuntu on WSL2 is available on this machine
- WSL sees the RTX 3080 and has both Python 3.12 and `uv`

Implication:

- Linux-first servers no longer need to be treated as hypothetical future work here
- the right execution plan is:
  - keep `ollama_local` as the zero-friction Windows path
  - use `openai_compatible_http` as the stable integration layer
  - run vLLM or SGLang inside WSL when moving beyond the Windows-native path

### 3. For a 10 GB RTX 3080, model choice should favor compact instruction models with long context

Primary candidates from official model sources:

- `gemma3:4b`
  - Ollama library lists it at about 3.3 GB with 128K context
  - Google’s official Gemma 3 4B model card describes Gemma 3 as a lightweight state-of-the-art open model with 128K context for the 4B size
- `qwen3:4b`
  - Ollama library lists it at about 2.5 GB and advertises a long context variant
  - Qwen’s official Hugging Face card states Qwen3-4B has 32,768 native context and 131,072 with YaRN, with strong reasoning, agent, and multilingual claims

Stretch candidate:

- `gpt-oss:20b`
  - Ollama lists it at about 14 GB and positions it as state-of-the-art open weight reasoning/agentic capability
  - the Ollama model page says the smaller model can run on systems with as little as 16 GB memory

Implication:

- for repeatable local evaluation on this box, start with `gemma3:4b` or `qwen3:4b`
- treat `gpt-oss:20b` as a later experiment, not the default local benchmark model

## Recommendation For TYPHON

### Short term

- add an inference backend abstraction now
- implement `ollama_local` first using the local HTTP API
- implement `openai_compatible_http` immediately after that for WSL-hosted servers
- preserve the extractive heuristic path as a fallback and control condition

### Medium term

- add a generic OpenAI-compatible HTTP backend
- use that to connect the same evaluation surface to:
  - Ollama
  - llama.cpp `llama-server`
  - vLLM
  - SGLang

### Why this is the right order

- it gives this repo a real model-backed path immediately
- it avoids committing early to a Linux-only or heavyweight runtime
- it keeps the benchmark/evaluation interfaces stable while the serving backend changes underneath

## Sources

- Ollama quickstart: https://docs.ollama.com/quickstart
- Ollama API intro: https://docs.ollama.com/api
- Ollama chat API: https://docs.ollama.com/api/chat
- Ollama list models API: https://docs.ollama.com/api/tags
- Ollama running models API: https://docs.ollama.com/api/ps
- Ollama model library, Gemma 3: https://ollama.com/library/gemma3
- Ollama model library, Qwen 3: https://ollama.com/library/qwen3
- Ollama model library, gpt-oss: https://ollama.com/library/gpt-oss
- Google Gemma 3 4B model card: https://huggingface.co/google/gemma-3-4b-it
- Qwen3-4B model card: https://huggingface.co/Qwen/Qwen3-4B
- vLLM installation: https://docs.vllm.ai/en/latest/getting_started/installation.html
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- vLLM quantization: https://docs.vllm.ai/en/latest/features/quantization/index.html
- SGLang install: https://docs.sglang.ai/get_started/install.html
- llama.cpp repository and server docs: https://github.com/ggml-org/llama.cpp
