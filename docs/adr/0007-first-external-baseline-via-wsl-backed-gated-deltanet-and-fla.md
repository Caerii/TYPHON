# ADR 0007: First External Baseline Via WSL-Backed Gated DeltaNet And FLA

- Status: Accepted
- Date: 2026-04-22

## Context

The project needed a stronger external baseline before moving further into heuristic tuning or learned-memory work. Official NVIDIA Gated DeltaNet weights were not available, and the practical runtime path on this workstation is Linux-first.

## Decision

Implement the first external baseline as a WSL-backed Gated DeltaNet wrapper using `flash-linear-attention`, with explicit checkpoint provenance and compatibility conversion for the selected third-party checkpoint family.

## Consequences

- The first external baseline is a real runnable systems baseline, not a full official paper-weight reproduction.
- Baseline execution bridges the Windows CLI and a Linux Torch runtime in WSL.
- Checkpoint assumptions and conversion behavior must stay explicit in configs, docs, and artifacts.
