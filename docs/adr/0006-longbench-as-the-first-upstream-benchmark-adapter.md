# ADR 0006: LongBench As The First Upstream Benchmark Adapter

- Status: Accepted
- Date: 2026-04-22

## Context

The repo needed a real upstream benchmark path that matched the current workstation constraints and the existing evaluation harness shape.

## Decision

LongBench English tasks are the first upstream benchmark adapter target. LongBench v2 is deferred to a later phase, and LoCoMo follows after the first stable long-context adapter path.

## Consequences

- The repo now includes a dedicated LongBench import path from the official Hugging Face dataset.
- The evaluation surface supports `reference_answers` because official LongBench rows may have multiple acceptable gold answers.
- The first real benchmark calibration point is now LongBench rather than curated local slices alone.
