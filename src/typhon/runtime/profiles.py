from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typhon.runtime.base import RuntimeInfo, RuntimeProfile
from typhon.utils.paths import repo_root


def _profile_dir() -> Path:
    return repo_root() / "configs" / "runtime"


def _matches(runtime: RuntimeInfo, match: dict[str, Any]) -> bool:
    min_vram = int(match.get("min_vram_mib", 0))
    max_vram = int(match.get("max_vram_mib", 2**31 - 1))
    gpu_name_contains = match.get("gpu_name_contains")

    runtime_vram = runtime.total_vram_mib if runtime.total_vram_mib is not None else 0
    if runtime_vram < min_vram or runtime_vram > max_vram:
        return False
    if gpu_name_contains:
        if runtime.gpu_name is None or gpu_name_contains.lower() not in runtime.gpu_name.lower():
            return False
    return True


def select_runtime_profile(runtime: RuntimeInfo) -> RuntimeProfile:
    default_payload: dict[str, Any] | None = None
    for path in sorted(_profile_dir().glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["id"] == "default_workstation":
            default_payload = payload
            continue
        if _matches(runtime, payload.get("match", {})):
            return RuntimeProfile(
                profile_id=payload["id"],
                recommendations=dict(payload["recommendations"]),
                runtime=runtime,
            )

    if default_payload is None:
        raise RuntimeError("No default runtime profile found.")
    return RuntimeProfile(
        profile_id=default_payload["id"],
        recommendations=dict(default_payload["recommendations"]),
        runtime=runtime,
    )
