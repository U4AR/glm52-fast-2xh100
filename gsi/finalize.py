from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .profile import GSIProfile, sha256_file


def _resolve_for_rank(profile: GSIProfile, artifact: str, rank: int) -> Path:
    path = Path(artifact.format(tp_rank=rank))
    if path.is_absolute():
        return path
    if profile.path is None:
        raise ValueError("profile path is unavailable")
    return profile.path.parent / path


def finalize(profile_path: Path, output_path: Path) -> GSIProfile:
    from safetensors import safe_open

    profile = GSIProfile.load(profile_path, verify_files=False)
    updated = []
    for entry in profile.entries:
        basis_checksums = {}
        basis_sizes = {}
        basis_shapes = {}
        basis_dtypes = {}
        for rank in range(profile.tp_size):
            basis_path = _resolve_for_rank(profile, entry.basis_file, rank)
            if not basis_path.is_file():
                raise FileNotFoundError(
                    f"basis shard missing for rank {rank}: {basis_path}"
                )
            basis_checksums[str(rank)] = sha256_file(basis_path)
            basis_sizes[str(rank)] = basis_path.stat().st_size
            with safe_open(
                str(basis_path), framework="pt", device="cpu"
            ) as handle:
                tensor = handle.get_slice("basis")
                basis_shapes[str(rank)] = list(tensor.get_shape())
                basis_dtypes[str(rank)] = str(tensor.get_dtype())
        metadata = dict(entry.metadata)
        metadata.update(
            {
                "basis_sha256_by_rank": basis_checksums,
                "basis_bytes_by_rank": basis_sizes,
                "basis_shape_by_rank": basis_shapes,
                "basis_safetensors_dtype_by_rank": basis_dtypes,
            }
        )
        if not entry.image_file:
            updated.append(
                replace(
                    entry,
                    basis_sha256=basis_checksums["0"],
                    metadata=metadata,
                )
            )
            continue
        checksums = {}
        sizes = {}
        shapes = {}
        dtypes = {}
        for rank in range(profile.tp_size):
            image_path = _resolve_for_rank(profile, entry.image_file, rank)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"image shard missing for rank {rank}: {image_path}"
                )
            checksums[str(rank)] = sha256_file(image_path)
            sizes[str(rank)] = image_path.stat().st_size
            with safe_open(
                str(image_path), framework="pt", device="cpu"
            ) as handle:
                key = "image" if "image" in handle.keys() else handle.keys()[0]
                tensor = handle.get_slice(key)
                shapes[str(rank)] = list(tensor.get_shape())
                dtypes[str(rank)] = str(tensor.get_dtype())
        metadata.update(
            {
                "image_sha256_by_rank": checksums,
                "image_bytes_by_rank": sizes,
                "image_shape_by_rank": shapes,
                "image_safetensors_dtype_by_rank": dtypes,
            }
        )
        updated.append(
            replace(
                entry,
                basis_sha256=basis_checksums["0"],
                metadata=metadata,
            )
        )
    profile.entries = updated
    profile.metadata = {
        **profile.metadata,
        "cache_finalized": True,
        "cache_format": "gsi-image-v1",
    }
    profile.save(output_path)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate every TP image shard and finalize its manifest"
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    profile = finalize(args.profile, args.out)
    image_entries = sum(entry.image_file is not None for entry in profile.entries)
    print(
        f"finalized {image_entries} image entries x TP{profile.tp_size} "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
