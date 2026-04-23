from __future__ import annotations

import platform
import subprocess
import sys

from typhon.runtime.base import RuntimeInfo


def _detect_gpu_with_nvidia_smi() -> tuple[str | None, int | None, str | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None, None

    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    if not line:
        return None, None, None

    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None, None, None

    gpu_name, memory_total, driver_version = parts
    try:
        total_vram_mib = int(memory_total)
    except ValueError:
        total_vram_mib = None
    return gpu_name or None, total_vram_mib, driver_version or None


def detect_runtime() -> RuntimeInfo:
    gpu_name, total_vram_mib, driver_version = _detect_gpu_with_nvidia_smi()
    return RuntimeInfo(
        gpu_name=gpu_name,
        total_vram_mib=total_vram_mib,
        driver_version=driver_version,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )
