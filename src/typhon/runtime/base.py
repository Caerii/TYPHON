from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeInfo:
    gpu_name: str | None
    total_vram_mib: int | None
    driver_version: str | None
    python_version: str
    platform: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "total_vram_mib": self.total_vram_mib,
            "driver_version": self.driver_version,
            "python_version": self.python_version,
            "platform": self.platform,
        }


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    recommendations: dict[str, Any]
    runtime: RuntimeInfo

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "recommendations": self.recommendations,
            "runtime": self.runtime.to_dict(),
        }
