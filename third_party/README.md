# Third-Party Code

This directory is reserved for isolated external repositories or vendored code.

## Rules

- Keep upstream code isolated here rather than mixing it into `src/`.
- Record source repository and commit when adding a new third-party dependency snapshot.
- Do not patch third-party code casually. Prefer thin wrappers in `src/typhon/`.
- If local modifications are unavoidable, document them clearly and keep them reviewable.
