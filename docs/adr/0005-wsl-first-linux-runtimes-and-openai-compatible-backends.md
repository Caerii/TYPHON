# ADR 0005: WSL-First Linux Runtimes And OpenAI-Compatible Backends

- Status: Accepted
- Date: 2026-04-22

## Context

The workstation is Windows-based but has a working WSL2 Ubuntu environment with GPU visibility. Several target runtimes and baseline implementations are Linux-first in practice.

## Decision

Use WSL for Linux-first runtime stacks and keep the repo-side inference abstraction centered on OpenAI-compatible HTTP where possible. Retain explicit aliases for practical local backends such as LM Studio.

## Consequences

- vLLM, SGLang, FLA, and similar stacks should run in WSL when needed.
- The Windows-side CLI remains the stable control plane.
- `openai_compatible_http` becomes the generic backend surface for multiple serving stacks.
- Thin process wrappers remain acceptable for direct model execution paths that do not fit the HTTP abstraction, such as the first Gated DeltaNet baseline.
