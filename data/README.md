# Data

This directory stores normalized local benchmark assets and raw import sources.

## Subfolders

- `benchmarks/`: normalized benchmark packs used by the repo
- `imports/`: raw JSON or JSONL files used as import sources

## Rules

- Normalize benchmark data into manifest-backed packs under `data/benchmarks/`.
- Keep raw import payloads under `data/imports/`.
- Do not mix normalized data with scratch analysis outputs.
- Respect upstream dataset licensing and provenance requirements.

See [Benchmark Packs Runbook](../docs/runbooks/benchmark-packs.md) for the expected local format.
