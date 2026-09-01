from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import FrozenSet


class GSIMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    FUNCTIONAL = "functional"
    KERNEL = "kernel"


_ACTIVE_PHASES = frozenset(
    {"decode", "target_verify", "draft_extend", "draft_extend_v2"}
)


def _parse_layers(raw: str) -> FrozenSet[int] | None:
    """Parse `GSI_EXPERT_LAYERS` as a comma list of indices and `a-b` ranges."""
    if not raw or raw.lower() == "all":
        return None
    if raw.lower() == "none":
        # Explicitly build no routed-expert images.  Distinct from unset, which
        # means every routed layer.
        return frozenset()
    layers: set[int] = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, _, stop = piece.partition("-")
            low, high = int(start), int(stop)
            if low > high:
                raise ValueError(f"invalid layer range in GSI_EXPERT_LAYERS: {piece}")
            layers.update(range(low, high + 1))
        else:
            layers.add(int(piece))
    if not layers:
        raise ValueError("GSI_EXPERT_LAYERS parsed to an empty set")
    return frozenset(layers)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class GSIConfig:
    mode: GSIMode = GSIMode.OFF
    profile: Path | None = None
    cache_dir: Path | None = None
    image_dtype: str = "bf16"
    strict_fallback: bool = True
    capture_dir: Path | None = None
    capture_rows: int = 4096
    telemetry_path: Path | None = None
    gate_override: str = "profile"
    active_phases: FrozenSet[str] = _ACTIVE_PHASES
    build_images: bool = False
    #: Layers whose routed-expert images are built.  None means every routed
    #: layer, which does not fit alongside the resident weights on an 80 GiB
    #: card at any useful rank -- see GSI_STATUS.md -- so a subset is the normal
    #: setting for measurement.
    expert_layers: FrozenSet[int] | None = None
    expert_row_budget: int = 1 << 16

    @classmethod
    def from_env(cls) -> "GSIConfig":
        try:
            mode = GSIMode(os.environ.get("GSI_MODE", "off").strip().lower())
        except ValueError as exc:
            raise ValueError(
                "GSI_MODE must be off, observe, functional, or kernel"
            ) from exc

        profile_raw = os.environ.get("GSI_PROFILE", "").strip()
        cache_raw = os.environ.get("GSI_CACHE_DIR", "").strip()
        capture_raw = os.environ.get("GSI_CAPTURE_DIR", "").strip()
        telemetry_raw = os.environ.get("GSI_TELEMETRY_PATH", "").strip()
        dtype = os.environ.get("GSI_IMAGE_DTYPE", "bf16").strip().lower()
        if dtype not in {"bf16", "fp8"}:
            raise ValueError("GSI_IMAGE_DTYPE must be bf16 or fp8")

        override = os.environ.get("GSI_GATE_OVERRIDE", "profile").strip().lower()
        if override not in {"profile", "all_slow", "always_fast"}:
            raise ValueError(
                "GSI_GATE_OVERRIDE must be profile, all_slow, or always_fast"
            )

        phases_raw = os.environ.get("GSI_ACTIVE_PHASES", "").strip()
        phases = (
            frozenset(x.strip() for x in phases_raw.split(",") if x.strip())
            if phases_raw
            else _ACTIVE_PHASES
        )
        rows = int(os.environ.get("GSI_CAPTURE_ROWS", "4096"))
        if rows <= 0:
            raise ValueError("GSI_CAPTURE_ROWS must be positive")

        expert_layers = _parse_layers(os.environ.get("GSI_EXPERT_LAYERS", "").strip())
        budget = int(os.environ.get("GSI_EXPERT_ROW_BUDGET", str(1 << 16)))
        if budget <= 0:
            raise ValueError("GSI_EXPERT_ROW_BUDGET must be positive")

        config = cls(
            mode=mode,
            profile=Path(profile_raw).resolve() if profile_raw else None,
            cache_dir=Path(cache_raw).resolve() if cache_raw else None,
            image_dtype=dtype,
            strict_fallback=_env_bool("GSI_STRICT_FALLBACK", True),
            capture_dir=Path(capture_raw).resolve() if capture_raw else None,
            capture_rows=rows,
            telemetry_path=(
                Path(telemetry_raw).resolve() if telemetry_raw else None
            ),
            gate_override=override,
            active_phases=phases,
            build_images=_env_bool("GSI_BUILD_IMAGES", False),
            expert_layers=expert_layers,
            expert_row_budget=budget,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode in {GSIMode.FUNCTIONAL, GSIMode.KERNEL} and self.profile is None:
            raise ValueError(f"GSI_PROFILE is required in {self.mode.value} mode")
        if self.profile is not None and not self.profile.is_file():
            raise FileNotFoundError(f"GSI profile not found: {self.profile}")
        if self.mode == GSIMode.OBSERVE and self.capture_dir is None:
            raise ValueError("GSI_CAPTURE_DIR is required in observe mode")
        if self.mode == GSIMode.KERNEL and not _env_bool(
            "GSI_EXPERIMENTAL_KERNEL", False
        ):
            raise RuntimeError(
                "kernel mode is capability-gated until the fused GLM/KT backend "
                "is installed; set GSI_EXPERIMENTAL_KERNEL=1 only for backend tests"
            )

