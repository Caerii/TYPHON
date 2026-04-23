# Scripts

This directory contains small operational wrappers that are easier to maintain as scripts than as inline shell commands.

## Subfolders

- `eval/`: evaluation entry wrappers and convenience scripts
- `wsl/`: Linux-first runtime helpers invoked from Windows

## Rules

- Keep scripts thin. Core project logic belongs in `src/typhon/`.
- Use scripts for reproducible environment setup, serving helpers, or process bridging.
- Document platform expectations inside the relevant runbook when a new script is added.
