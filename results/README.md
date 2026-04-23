# Results

This directory is for generated artifacts only.

## Expected Contents

- `smoke/`: smoke-test artifacts
- `baselines/`: baseline runner outputs
- `typhon/`: TYPHON runner outputs
- `comparisons/`: side-by-side baseline vs TYPHON artifacts
- `evaluations/`: aggregate metric summaries
- `inference/`: backend-specific sample artifacts
- `runtime/`: runtime detection or server logs

## Rules

- Do not hand-edit result artifacts.
- By default, generated artifacts stay out of git and only `.gitkeep` files are tracked.
- If a result materially changes project understanding, summarize it in `docs/project/experiment-matrix.md` rather than trying to preserve every raw output in version control.
