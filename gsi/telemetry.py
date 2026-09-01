from __future__ import annotations

import atexit
import json
import os
import threading
from collections import defaultdict
from pathlib import Path


class GSITelemetry:
    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        self._observations = 0
        self._flush_calls = int(
            os.environ.get("GSI_TELEMETRY_FLUSH_CALLS", "1000")
        )
        self._stats = defaultdict(
            lambda: {
                "calls": 0,
                "rows": 0,
                "fast": 0,
                "invalid": 0,
                "rho2_sum": 0.0,
                "rho2_max": 0.0,
            }
        )
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            atexit.register(self.flush)

    def observe(self, module: str, gate) -> None:
        rho2 = gate.rho2.detach().float()
        fast = gate.fast_mask.detach()
        valid = gate.valid_mask.detach()
        should_flush = False
        with self._lock:
            item = self._stats[module]
            item["calls"] += 1
            item["rows"] += rho2.numel()
            item["fast"] += int(fast.sum().item())
            item["invalid"] += int((~valid).sum().item())
            finite = rho2[valid]
            if finite.numel():
                item["rho2_sum"] += float(finite.sum().item())
                item["rho2_max"] = max(
                    item["rho2_max"], float(finite.max().item())
                )
            self._observations += 1
            should_flush = (
                self.path is not None
                and self._flush_calls > 0
                and self._observations % self._flush_calls == 0
            )
        if should_flush:
            self.flush()

    def snapshot(self) -> dict:
        with self._lock:
            result = {}
            for key, item in self._stats.items():
                copied = dict(item)
                copied["fast_fraction"] = (
                    copied["fast"] / copied["rows"] if copied["rows"] else 0.0
                )
                valid_rows = copied["rows"] - copied["invalid"]
                copied["mean_rho2"] = (
                    copied["rho2_sum"] / valid_rows if valid_rows else None
                )
                result[key] = copied
            return result

    def flush(self) -> None:
        if self.path is None:
            return
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
