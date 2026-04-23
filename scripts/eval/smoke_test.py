from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    src_value = str(src_dir)
    if src_value not in sys.path:
        sys.path.insert(0, src_value)


def main() -> int:
    _ensure_src_on_path()
    from typhon.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
