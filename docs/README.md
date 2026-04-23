# TYPHON Documentation

This repository keeps documentation in a small number of explicit categories. New docs should go into one of these sections rather than accumulating at the `docs/` root.

## Sections

- [`adr/`](adr/README.md): architecture decision records and documentation governance decisions
- [`architecture/`](architecture/system-overview.md): system shape, data flow, and major component boundaries
- [`project/`](project/technical-pm-brief.md): current status, handoff notes, experiment tracking, PM-facing material
- [`runbooks/`](runbooks/benchmark-packs.md): operational procedures for data, runtimes, and reproducible workflows
- [`research/`](research/sota-model-backends.md): research notes and source-backed implementation guidance
- [`archive/`](archive/decision-log-legacy.md): superseded or legacy documents retained for history

## Canonical Docs

- [System Overview](architecture/system-overview.md)
- [ADR Index](adr/README.md)
- [Technical PM Brief](project/technical-pm-brief.md)
- [Experiment Matrix](project/experiment-matrix.md)
- [Cursor Handoff](project/cursor-handoff.md)
- [Benchmark Packs Runbook](runbooks/benchmark-packs.md)
- [WSL Runtime Runbook](runbooks/wsl-runtime.md)

## Documentation Rules

- Use ADRs for architecture, tooling, data-shape, and workflow decisions that affect future work.
- Update the experiment matrix whenever a materially new benchmark path, baseline, or live result lands.
- Keep runbooks procedural. Keep research notes source-backed. Keep project docs status-oriented.
- Do not add scratch notes, meeting notes, or one-off experiment summaries directly under `docs/`.

## Operational Folder Guides

- [configs](../configs/README.md)
- [data](../data/README.md)
- [results](../results/README.md)
- [scripts](../scripts/README.md)
- [third_party](../third_party/README.md)
- [notebooks](../notebooks/README.md)
