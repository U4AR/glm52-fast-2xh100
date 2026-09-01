from __future__ import annotations

import argparse
import json
from pathlib import Path


def checkpoint_expert_bytes(model_dir: Path, layer: int = 3, expert: int = 0) -> int:
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    prefix = f"model.layers.{layer}.mlp.experts.{expert}."
    total = 0
    grouped: dict[str, list[str]] = {}
    for name, shard in index.items():
        if name.startswith(prefix):
            grouped.setdefault(shard, []).append(name)
    dtype_bytes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    for shard, names in grouped.items():
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_slice(name)
                count = 1
                for dim in tensor.get_shape():
                    count *= dim
                total += count * dtype_bytes[str(tensor.get_dtype())]
    if total == 0:
        raise ValueError(f"no expert tensors found under {prefix}")
    return total


def estimate(
    *,
    hidden: int,
    intermediate: int,
    experts: int,
    routed_layers: int,
    gate_rank: int,
    down_rank: int,
    image_bytes: int,
    tp_size: int,
    hot_full_experts: int,
    checkpoint_full_expert_bytes: int,
) -> dict:
    gate_up_elements = 2 * intermediate * gate_rank
    down_elements = hidden * down_rank
    per_expert_image = (gate_up_elements + down_elements) * image_bytes
    all_images = per_expert_image * experts * routed_layers
    all_host_weights = checkpoint_full_expert_bytes * experts * routed_layers
    hot_weights = checkpoint_full_expert_bytes * hot_full_experts * routed_layers
    return {
        "per_expert_image_bytes": per_expert_image,
        "all_images_model_bytes": all_images,
        "all_images_per_tp_rank_bytes": (all_images + tp_size - 1) // tp_size,
        "all_full_experts_host_bytes": all_host_weights,
        "hot_full_weights_model_bytes": hot_weights,
        "hot_full_weights_per_tp_rank_bytes": (hot_weights + tp_size - 1)
        // tp_size,
        "parameters": {
            "hidden": hidden,
            "intermediate": intermediate,
            "experts": experts,
            "routed_layers": routed_layers,
            "gate_rank": gate_rank,
            "down_rank": down_rank,
            "image_bytes": image_bytes,
            "tp_size": tp_size,
            "hot_full_experts": hot_full_experts,
            "checkpoint_full_expert_bytes": checkpoint_full_expert_bytes,
        },
    }


def _gib(value: int) -> float:
    return value / (1 << 30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate GLM GSI expert storage")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gate-rank", type=int, default=256)
    parser.add_argument("--down-rank", type=int, default=256)
    parser.add_argument("--image-dtype", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--hot-full-experts", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    first_sparse = int(config.get("first_k_dense_replace", 0))
    routed_layers = int(config["num_hidden_layers"]) - first_sparse
    expert_bytes = checkpoint_expert_bytes(args.model, first_sparse, 0)
    result = estimate(
        hidden=int(config["hidden_size"]),
        intermediate=int(config["moe_intermediate_size"]),
        experts=int(config["n_routed_experts"]),
        routed_layers=routed_layers,
        gate_rank=args.gate_rank,
        down_rank=args.down_rank,
        image_bytes=2 if args.image_dtype == "bf16" else 1,
        tp_size=args.tp_size,
        hot_full_experts=args.hot_full_experts,
        checkpoint_full_expert_bytes=expert_bytes,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    for key in (
        "per_expert_image_bytes",
        "all_images_model_bytes",
        "all_images_per_tp_rank_bytes",
        "all_full_experts_host_bytes",
        "hot_full_weights_per_tp_rank_bytes",
    ):
        print(f"{key:38s} {_gib(result[key]):8.2f} GiB")


if __name__ == "__main__":
    main()

