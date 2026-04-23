# ADR 0004: Manifest-Backed Benchmark Packs

- Status: Accepted
- Date: 2026-04-22

## Context

Ad hoc benchmark sample files were starting to sprawl and lacked provenance, split metadata, and stable grouping.

## Decision

Standardize local benchmark data on manifest-backed packs under `data/benchmarks/<benchmark_id>/`, with `pack.json` plus per-pack sample files under `packs/<pack_id>/samples.jsonl`.

## Consequences

- Local benchmark assets gain explicit provenance and grouping.
- Benchmark importers normalize into one local format rather than inventing per-benchmark layouts.
- Legacy root-level `samples.jsonl` and `samples.json` files remain readable for compatibility, but they are no longer the preferred local shape.
