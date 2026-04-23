# Configs

This directory is the declared control surface for experiments and runtime behavior.

## Subfolders

- `benchmarks/`: benchmark metadata, fixture declarations, and adapter configs
- `baselines/`: baseline-specific runtime configs
- `runtime/`: workstation or hardware profiles
- `live_eval/`: named live-evaluation presets
- `typhon/`: TYPHON architecture or runner presets

## Rules

- Prefer JSON for repo-native configs unless an ADR changes that decision.
- Keep configs declarative. Do not encode executable logic here.
- Add new benchmark or baseline configs before adding ad hoc command-line-only workflows.
- Name files by stable concept, not by one-off experiment notes.
