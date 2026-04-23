from __future__ import annotations

from pathlib import Path


def windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Expected a Windows drive path, got: {resolved}")
    posix_path = resolved.as_posix()
    _, remainder = posix_path.split(":", 1)
    return f"/mnt/{drive}{remainder}"
