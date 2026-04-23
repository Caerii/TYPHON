# ADR 0008: Structured Documentation Tree And Folder Governance

- Status: Accepted
- Date: 2026-04-22

## Context

The repo had accumulated a flat `docs/` directory and several operational folders without clear written rules. That makes drift, duplication, and storage junk more likely over time.

## Decision

Adopt a structured documentation tree under `docs/` with explicit categories:

- `adr/`
- `architecture/`
- `project/`
- `runbooks/`
- `research/`
- `archive/`

Also require top-level governance READMEs for operational folders such as `configs/`, `data/`, `results/`, `scripts/`, `notebooks/`, and `third_party/`.

## Consequences

- New documentation must be placed by purpose rather than dropped into the docs root.
- ADRs become the canonical place for stable decisions.
- Folder intent and storage rules are visible at the point of use.
- The repo becomes easier to maintain without accumulating ambiguous docs or mixed-purpose assets.
