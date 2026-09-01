from __future__ import annotations

import atexit
import json
import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path


def _safe_key(value: str) -> str:
    return value.replace("/", "__")


@dataclass
class _Reservoir:
    rows: object
    seen: int = 0
    filled: int = 0


class ActivationRecorder:
    """Bounded, deterministic row reservoirs for eager calibration runs."""

    def __init__(self, output_dir: Path, capacity: int, rank: int = 0):
        import torch

        self.output_dir = output_dir
        self.capacity = capacity
        self.rank = rank
        self._torch = torch
        self._lock = threading.Lock()
        self._reservoirs: dict[str, _Reservoir] = {}
        self._rng = random.Random(0x475349 + rank)
        self._auto_flushed = False
        self._auto_flush = os.environ.get("GSI_CAPTURE_AUTO_FLUSH", "1") != "0"
        self._minimum_keys = int(os.environ.get("GSI_CAPTURE_MIN_KEYS", "100"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self.flush)

    def record(self, key: str, value) -> None:
        torch = self._torch
        if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
            return
        value = value.detach().reshape(-1, value.shape[-1]).to(
            device="cpu", dtype=torch.float16
        )
        should_flush = False
        with self._lock:
            reservoir = self._reservoirs.get(key)
            if reservoir is None:
                rows = torch.empty(
                    (self.capacity, value.shape[-1]), dtype=torch.float16
                )
                reservoir = _Reservoir(rows=rows)
                self._reservoirs[key] = reservoir
            if reservoir.rows.shape[-1] != value.shape[-1]:
                raise ValueError(
                    f"capture width changed for {key}: "
                    f"{reservoir.rows.shape[-1]} -> {value.shape[-1]}"
                )
            for row in value:
                reservoir.seen += 1
                if reservoir.filled < self.capacity:
                    reservoir.rows[reservoir.filled].copy_(row)
                    reservoir.filled += 1
                    continue
                candidate = self._rng.randrange(reservoir.seen)
                if candidate < self.capacity:
                    reservoir.rows[candidate].copy_(row)
            if (
                self._auto_flush
                and not self._auto_flushed
                and len(self._reservoirs) >= self._minimum_keys
                and all(
                    item.filled >= self.capacity
                    for item in self._reservoirs.values()
                )
            ):
                self._auto_flushed = True
                should_flush = True
        # Scheduler workers may be force-terminated by their parent before
        # Python atexit handlers finish.  Persist one complete bounded snapshot
        # as soon as every known reservoir fills, outside the record lock.
        if should_flush:
            self.flush()

    def flush(self) -> Path | None:
        if not self._reservoirs:
            return None
        from safetensors.torch import save_file

        with self._lock:
            tensors = {
                _safe_key(key): reservoir.rows[: reservoir.filled].contiguous()
                for key, reservoir in self._reservoirs.items()
                if reservoir.filled
            }
            metadata = {
                "format": "gsi-activation-reservoir-v1",
                "rank": str(self.rank),
                "pid": str(os.getpid()),
                "seen": json.dumps(
                    {key: value.seen for key, value in self._reservoirs.items()},
                    sort_keys=True,
                ),
            }
            if not tensors:
                return None
            path = self.output_dir / f"capture-r{self.rank}-p{os.getpid()}.safetensors"
            temporary = path.with_suffix(".tmp")
            save_file(tensors, str(temporary), metadata=metadata)
            temporary.replace(path)
            return path
