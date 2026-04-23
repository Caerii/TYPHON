# ADR 0001: PyTorch-First Research Substrate

- Status: Accepted
- Date: 2026-04-22

## Context

The repo needs one stable implementation control plane even though some target baselines and papers use other frameworks, especially JAX.

## Decision

The core TYPHON repo will be Python-first and PyTorch-oriented. External baselines in other frameworks will be wrapped behind adapters rather than shaping the internal architecture of the repo.

## Consequences

- `src/typhon/` remains framework-light at the orchestration layer but assumes a Python and PyTorch-centric workflow.
- PyTorch-native projects such as FLA, Gated DeltaNet, MemoryLLM, and PERK fit naturally into the repo.
- JAX-heavy baselines such as TTT-E2E should be integrated through explicit wrapper boundaries rather than introducing mixed-framework assumptions into the core codebase.
