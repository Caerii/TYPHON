# ADR 0002: JSON Configs And uv Tooling

- Status: Accepted
- Date: 2026-04-22

## Context

The repo needed a low-friction control surface for benchmarks, runtimes, and experiments before heavy dependencies and orchestration complexity accumulated.

## Decision

Use JSON for repo-native configuration and `uv` as the default command runner and environment manager.

## Consequences

- Config parsing stays on the Python standard library unless a later ADR changes it.
- Repo instructions and automation should use `uv run ...`.
- Dependency changes should be followed by `uv lock`.
- Config ergonomics are less flexible than richer config systems, but the project remains simple and portable.
