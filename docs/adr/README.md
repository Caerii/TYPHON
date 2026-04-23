# Architecture Decision Records

ADRs are the canonical record for durable decisions about architecture, tooling, data shape, and workflow.

## Active ADRs

- [0001: PyTorch-first research substrate](0001-pytorch-first-research-substrate.md)
- [0002: JSON configs and uv tooling](0002-json-configs-and-uv-tooling.md)
- [0003: Deterministic harness and heuristic v0 before learned memory](0003-deterministic-harness-and-heuristic-v0-before-learned-memory.md)
- [0004: Manifest-backed benchmark packs](0004-manifest-backed-benchmark-packs.md)
- [0005: WSL-first Linux runtimes and OpenAI-compatible backends](0005-wsl-first-linux-runtimes-and-openai-compatible-backends.md)
- [0006: LongBench as the first upstream benchmark adapter](0006-longbench-as-the-first-upstream-benchmark-adapter.md)
- [0007: First external baseline via WSL-backed Gated DeltaNet and FLA](0007-first-external-baseline-via-wsl-backed-gated-deltanet-and-fla.md)
- [0008: Structured documentation tree and folder governance](0008-structured-documentation-tree-and-folder-governance.md)

## Usage

- Add a new ADR when a decision will shape future implementation or repository structure.
- Update or supersede an ADR rather than silently drifting away from it.
- Keep experiment outcomes in [project docs](../project/experiment-matrix.md), not in ADRs.
