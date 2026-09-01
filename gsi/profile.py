from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .distributed import tensor_parallel_rank


PROFILE_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GSIEntry:
    module: str
    family: str
    layer: int
    rank: int
    epsilon: float
    input_dim: int
    output_dim: int
    basis_file: str
    basis_sha256: str = ""
    image_file: str | None = None
    image_sha256: str = ""
    expert: int | None = None
    basis_dtype: str = "bf16"
    image_dtype: str = "bf16"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.module:
            raise ValueError("profile entry module cannot be empty")
        if self.rank <= 0 or self.rank > self.input_dim:
            raise ValueError(
                f"invalid rank {self.rank} for {self.module} input {self.input_dim}"
            )
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative for {self.module}")
        if self.input_dim <= 0 or self.output_dim < 0:
            raise ValueError(f"invalid dimensions for {self.module}")
        if not self.basis_file:
            raise ValueError(f"basis_file missing for {self.module}")


@dataclass
class GSIProfile:
    model_id: str
    model_hash: str
    quantization: str
    tp_size: int
    entries: list[GSIEntry]
    calibration: dict[str, Any]
    version: int = PROFILE_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    path: Path | None = field(default=None, repr=False, compare=False)

    def validate(self, verify_files: bool = False) -> None:
        if self.version != PROFILE_VERSION:
            raise ValueError(
                f"unsupported GSI profile version {self.version}; "
                f"expected {PROFILE_VERSION}"
            )
        if self.tp_size <= 0:
            raise ValueError("tp_size must be positive")
        seen: set[tuple[str, int | None]] = set()
        for entry in self.entries:
            entry.validate()
            key = (entry.module, entry.expert)
            if key in seen:
                raise ValueError(f"duplicate GSI entry: {key}")
            seen.add(key)
            if verify_files:
                basis_path = self.resolve(entry.basis_file)
                rank_key = str(tensor_parallel_rank())
                per_rank_basis = entry.metadata.get(
                    "basis_sha256_by_rank", {}
                )
                expected_basis = per_rank_basis.get(
                    rank_key, entry.basis_sha256
                )
                self._verify_artifact(
                    basis_path, expected_basis, f"basis for {entry.module}"
                )
                if entry.image_file:
                    image_path = self.resolve(entry.image_file)
                    per_rank = entry.metadata.get("image_sha256_by_rank", {})
                    expected_image = per_rank.get(
                        rank_key, entry.image_sha256
                    )
                    self._verify_artifact(
                        image_path, expected_image, f"image for {entry.module}"
                    )

    def resolve(self, artifact: str) -> Path:
        rank = tensor_parallel_rank()
        value = Path(artifact.format(tp_rank=rank))
        if value.is_absolute():
            return value
        if self.path is None:
            raise ValueError("relative artifact cannot be resolved before profile load")
        return self.path.parent / value

    @staticmethod
    def _verify_artifact(path: Path, expected: str, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
        if expected and sha256_file(path) != expected:
            raise ValueError(f"checksum mismatch for {label}: {path}")

    def by_module(self) -> dict[str, GSIEntry]:
        return {entry.module: entry for entry in self.entries if entry.expert is None}

    def expert_entries(self, module: str) -> list[GSIEntry]:
        return [entry for entry in self.entries if entry.module == module]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("path", None)
        return payload

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.path = path.resolve()

    @classmethod
    def load(cls, path: Path, verify_files: bool = True) -> "GSIProfile":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["entries"] = [GSIEntry(**entry) for entry in payload["entries"]]
        profile = cls(**payload)
        profile.path = path.resolve()
        profile.validate(verify_files=verify_files)
        return profile

    @classmethod
    def from_entries(
        cls,
        *,
        model_id: str,
        model_hash: str,
        quantization: str,
        tp_size: int,
        entries: Iterable[GSIEntry],
        calibration: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> "GSIProfile":
        return cls(
            model_id=model_id,
            model_hash=model_hash,
            quantization=quantization,
            tp_size=tp_size,
            entries=list(entries),
            calibration=calibration,
            metadata=metadata or {},
        )
