# ADR 0003: Deterministic Harness And Heuristic v0 Before Learned Memory

- Status: Accepted
- Date: 2026-04-22

## Context

The project needed runnable value before full model-side memory mechanisms, training loops, or dataset reproductions existed.

## Decision

The first executable system should be a deterministic evaluation harness plus a heuristic `typhon_v0` memory-selection path, not a partially implemented learned memory model.

## Consequences

- Smoke tests and early runner outputs are planning or evaluation artifacts, not paper-style benchmark claims.
- `typhon_v0` currently uses ranked memory stores and heuristic write signals instead of learned updates.
- The repo can test whether memory allocation structure helps before investing in expensive fast-weight or consolidation mechanisms.
