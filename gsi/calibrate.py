from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .families import classify_linear, layer_from_module, routed_family
from .profile import GSIEntry, GSIProfile, sha256_file
from .subspace import cascade_basis, subspace_overlap

#: Paper operating point (arXiv 2605.03109, Table 6): k=256, eps=0.10 gives 99.8%
#: fast path on GPT-J with a 0.991 perplexity ratio.  The earlier 0.025 default
#: in this repo was 4x tighter than anything the method was evaluated at.
DEFAULT_EPSILON = 0.10


def candidate_ranks(width: int) -> list[int]:
    if width <= 512:
        values = (32, 64, 128, 256)
    elif width <= 2048:
        values = (64, 128, 256, 512)
    else:
        values = (128, 256, 512, 1024)
    return [rank for rank in values if rank <= width]


def load_ranked_captures(paths: list[Path]) -> dict[int, dict[str, object]]:
    import torch
    from safetensors import safe_open

    grouped: dict[int, dict[str, list]] = {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if "rank" not in metadata:
                raise ValueError(f"capture is missing TP rank metadata: {path}")
            tp_rank = int(metadata["rank"])
            for key in handle.keys():
                if ".expert_ids|" in key:
                    continue
                grouped.setdefault(tp_rank, {}).setdefault(key, []).append(
                    handle.get_tensor(key)
                )
    return {
        rank: {key: torch.cat(rows, dim=0) for key, rows in captures.items()}
        for rank, captures in grouped.items()
    }


def load_captures(paths: list[Path]) -> dict[str, object]:
    """Compatibility helper that pools captures across TP ranks."""
    import torch

    ranked = load_ranked_captures(paths)
    grouped: dict[str, list] = {}
    for captures in ranked.values():
        for key, value in captures.items():
            grouped.setdefault(key, []).append(value)
    return {key: torch.cat(values, dim=0) for key, values in grouped.items()}


def usable_rows(matrix):
    """Drop rows an SVD must never see: non-finite, or identically zero.

    A zero row carries no direction and makes the residual ratio undefined at
    runtime; a non-finite row poisons the whole decomposition.  Both appear in
    fused-MoE captures, where the intermediate buffer is sized for the worst
    case and only partly written.
    """
    import torch

    finite = torch.isfinite(matrix).all(dim=1)
    nonzero = (matrix != 0).any(dim=1)
    return matrix[finite & nonzero]


def pool_module_captures(captures: dict[str, object]) -> dict[str, dict]:
    """Pool decode and verification samples into one deployable module basis."""
    import torch

    grouped: dict[str, dict[str, list]] = {}
    for capture_key, matrix in captures.items():
        module, separator, phase = capture_key.partition("|")
        if not separator or not module or getattr(matrix, "ndim", 0) != 2:
            continue
        matrix = usable_rows(matrix)
        if matrix.shape[0] == 0:
            continue
        grouped.setdefault(module, {}).setdefault(phase, []).append(matrix)
    result = {}
    for module, phases in grouped.items():
        phase_matrices = {
            phase: torch.cat(values, dim=0) for phase, values in phases.items()
        }
        widths = {matrix.shape[1] for matrix in phase_matrices.values()}
        if len(widths) != 1:
            raise ValueError(
                f"capture width differs across phases for {module}: {sorted(widths)}"
            )
        result[module] = {
            "matrix": torch.cat(list(phase_matrices.values()), dim=0),
            "phases": phase_matrices,
        }
    return result


def basis_group_key(module: str, width: int, scope: str) -> str:
    """Name the set of modules that share one activation basis.

    The paper fits a single ``V_k^(l)`` per layer and shares it across every
    linear map in that layer.  That is only well-posed for maps reading the same
    activation, so the group key carries the input width: in GLM-5.2 the residual
    stream (6144), the attention head concatenation (8192), the q-latent (2048)
    and the SwiGLU intermediates are distinct spaces and cannot share a basis.
    """
    if scope == "module":
        return module
    if scope != "layer":
        raise ValueError(f"unknown basis scope: {scope}")
    layer = layer_from_module(module)
    prefix = "nextn" if module.startswith("nextn.") else "model"
    return f"{prefix}.layer{layer}.w{width}"


def right_singular_basis(x):
    """Right singular vectors and singular values of ``x``, robustly.

    cuSOLVER's `gesvd` raises on inputs with a degenerate or highly repeated
    spectrum, which the post-SiLU SwiGLU intermediates reliably produce.  The
    eigendecomposition of the Gram matrix yields the same right singular vectors
    and tolerates those inputs; CPU LAPACK is the last resort.
    """
    import torch

    try:
        _, singular, vh = torch.linalg.svd(x, full_matrices=False)
        return vh.transpose(0, 1).contiguous(), singular
    except Exception:
        pass
    try:
        gram = (x.transpose(0, 1).double() @ x.double())
        values, vectors = torch.linalg.eigh(gram)
        order = torch.argsort(values, descending=True)
        return (
            vectors[:, order].to(dtype=x.dtype).contiguous(),
            values[order].clamp_min(0).sqrt().to(dtype=x.dtype),
        )
    except Exception:
        _, singular, vh = torch.linalg.svd(x.cpu(), full_matrices=False)
        return (
            vh.transpose(0, 1).contiguous().to(x.device),
            singular.to(x.device),
        )


def analyze_matrix(matrix, ranks: list[int], *, basis_full=None) -> tuple[object, list[dict]]:
    """Evaluate the residual/fast-fraction sweep, optionally on a supplied basis.

    When ``basis_full`` is None a thin SVD is taken (the exact reference path).
    Cascade calibration supplies its own principal-axis-ordered basis instead, so
    that truncation to each candidate rank still yields nested subspaces.
    """
    import torch

    x = matrix.float()
    if basis_full is None:
        basis_full, singular = right_singular_basis(x)
    else:
        basis_full = basis_full.float()
        projected = x @ basis_full
        singular = torch.linalg.svdvals(projected)
    analyses = []
    x_energy = (x * x).sum(dim=1).clamp_min(1e-12)
    total_singular = singular.sum().clamp_min(1e-12)
    probabilities = singular / total_singular
    effective_rank = float(
        torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
    )
    for rank in ranks:
        if rank > basis_full.shape[1]:
            continue
        basis = basis_full[:, :rank]
        g = x @ basis
        rho2 = ((x_energy - (g * g).sum(dim=1)).clamp_min(0) / x_energy)
        analyses.append(
            {
                "rank": rank,
                "effective_rank": effective_rank,
                "mean_rho": float(rho2.sqrt().mean()),
                "p95_rho": float(torch.quantile(rho2.sqrt(), 0.95)),
                "p99_rho": float(torch.quantile(rho2.sqrt(), 0.99)),
                "fast_fraction": {
                    str(epsilon): float((rho2 < epsilon**2).float().mean())
                    for epsilon in (0.01, 0.025, 0.05, 0.075, 0.10, 0.15)
                },
            }
        )
    return basis_full, analyses


def _profile_root(out: Path, requested: int, multiple: bool) -> Path:
    return out / f"rank{requested}" if multiple else out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GSI bases from captures")
    parser.add_argument("--captures", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-id", default="GLM5.2")
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--quantization", default="w4afp8")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--rank", type=int)
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        help=(
            "Build multiple requested-rank profiles from each SVD in one pass. "
            "Narrow families use the largest candidate not exceeding the cap."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"residual gate threshold (paper operating point {DEFAULT_EPSILON})",
    )
    parser.add_argument(
        "--basis-scope",
        choices=("layer", "module"),
        default="layer",
        help=(
            "'layer' shares one basis across every map at a layer that reads the "
            "same activation width, as the paper specifies; 'module' fits an "
            "independent basis per linear."
        ),
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help=(
            "Seed each layer's basis from the previous layer and refine by "
            "orthogonal iteration instead of taking an independent thin SVD."
        ),
    )
    parser.add_argument(
        "--cascade-iterations",
        type=int,
        default=2,
        help="orthogonal-iteration steps per layer when --cascade is set",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="SVD device (for example cpu, cuda, or cuda:0)",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        help=(
            "Run TP-rank SVDs concurrently on these devices, one per TP rank "
            "(for example --devices cuda:0 cuda:1)."
        ),
    )
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file

    if args.rank is not None and args.ranks:
        parser.error("--rank and --ranks are mutually exclusive")
    requested_ranks = sorted(
        set(args.ranks or ([args.rank] if args.rank is not None else [1024]))
    )
    if any(rank <= 0 for rank in requested_ranks):
        parser.error("requested ranks must be positive")
    if args.devices and len(args.devices) != args.tp_size:
        parser.error("--devices must provide exactly one device per TP rank")
    multiple = len(requested_ranks) > 1

    args.out.mkdir(parents=True, exist_ok=True)
    ranked_captures = load_ranked_captures(args.captures)
    expected_ranks = set(range(args.tp_size))
    if set(ranked_captures) != expected_ranks:
        raise ValueError(
            "capture TP ranks do not match profile TP size: "
            f"found {sorted(ranked_captures)}, expected {sorted(expected_ranks)}"
        )
    ranked_modules = {
        tp_rank: pool_module_captures(captures)
        for tp_rank, captures in ranked_captures.items()
    }
    module_sets = {rank: set(modules) for rank, modules in ranked_modules.items()}
    common_modules = set.intersection(*module_sets.values())
    if any(modules != common_modules for modules in module_sets.values()):
        details = {
            str(rank): sorted(modules - common_modules)
            for rank, modules in module_sets.items()
            if modules != common_modules
        }
        raise ValueError(f"capture modules differ across TP ranks: {details}")

    # Assign each module to the basis group it shares.  Widths must agree across
    # TP ranks, which the module-set check above already guarantees pairwise.
    widths: dict[str, int] = {}
    for module in sorted(common_modules):
        module_widths = {
            ranked_modules[tp_rank][module]["matrix"].shape[1]
            for tp_rank in range(args.tp_size)
        }
        if len(module_widths) != 1:
            raise ValueError(f"input width differs across TP ranks for {module}")
        widths[module] = module_widths.pop()

    groups: dict[str, list[str]] = defaultdict(list)
    for module in sorted(common_modules):
        groups[basis_group_key(module, widths[module], args.basis_scope)].append(module)

    def group_sort_key(key: str) -> tuple:
        members = groups[key]
        return (widths[members[0]], layer_from_module(members[0]), key)

    ordered_groups = sorted(groups, key=group_sort_key)

    entries_by_request: dict[int, list[GSIEntry]] = {
        rank: [] for rank in requested_ranks
    }
    report: dict[str, dict] = {}
    # Cascade state and the depth-coherence diagnostic are both keyed by the
    # space being spanned, not by the module: layer l+1 can only seed layer l+2
    # within the same activation width and TP shard.
    seeds: dict[tuple[int, int, int], object] = {}
    previous: dict[tuple[int, int, int], tuple[int, object]] = {}
    overlaps: list[dict] = []

    def calibrate_shard(group: str, tp_rank: int) -> dict:
        members = groups[group]
        pooled = [ranked_modules[tp_rank][module] for module in members]
        matrix_cpu = (
            pooled[0]["matrix"]
            if len(pooled) == 1
            else torch.cat([item["matrix"] for item in pooled], dim=0)
        )
        device = args.devices[tp_rank] if args.devices else args.device
        matrix = matrix_cpu.to(device)
        ranks = [
            rank
            for rank in candidate_ranks(matrix.shape[1])
            if rank <= min(matrix.shape)
        ]
        if not ranks:
            raise ValueError(
                "insufficient rows for minimum candidate rank: "
                f"{group} TP{tp_rank}, shape={tuple(matrix.shape)}"
            )

        cascade_report = None
        if args.cascade:
            sketch_rank = min(max(ranks), min(matrix.shape))
            seed = seeds.get((tp_rank, matrix.shape[1], sketch_rank))
            seed = seed.to(device) if seed is not None else None
            result = cascade_basis(
                matrix,
                sketch_rank,
                seed=seed,
                iterations=args.cascade_iterations,
            )
            basis_full, analysis = analyze_matrix(
                matrix, ranks, basis_full=result.basis
            )
            seeds[(tp_rank, matrix.shape[1], sketch_rank)] = result.basis.to("cpu")
            cascade_report = {
                "iterations": result.iterations,
                "seeded": result.seeded,
                "seed_overlap": result.seed_overlap,
                "mean_rho": result.mean_rho,
            }
        else:
            basis_full, analysis = analyze_matrix(matrix, ranks)

        phase_rows: dict[str, int] = defaultdict(int)
        for item in pooled:
            for phase, phase_matrix in item["phases"].items():
                phase_rows[phase] += phase_matrix.shape[0]
        shard_artifacts = {}
        for requested in requested_ranks:
            feasible = [rank for rank in ranks if rank <= requested]
            if not feasible:
                raise ValueError(
                    f"rank cap {requested} has no candidate for {group} "
                    f"TP{tp_rank}; candidates are {ranks}"
                )
            chosen_rank = feasible[-1]
            # Re-orthogonalize after an initial storage-dtype conversion.
            converted = basis_full[:, :chosen_rank].to(torch.bfloat16).float()
            basis = torch.linalg.qr(converted, mode="reduced").Q.to(
                device="cpu", dtype=torch.bfloat16
            )
            gram = basis.float().T @ basis.float()
            gram_error = float(
                (gram - torch.eye(chosen_rank)).abs().max()
            )
            # Depth coherence: compare this layer against the previous layer that
            # spans the same activation space at the same rank.
            coherence_key = (tp_rank, matrix.shape[1], chosen_rank)
            layer = layer_from_module(members[0])
            prior = previous.get(coherence_key)
            if prior is not None and prior[0] != layer:
                overlaps.append(
                    {
                        "group": group,
                        "tp_rank": tp_rank,
                        "width": matrix.shape[1],
                        "rank": chosen_rank,
                        "from_layer": prior[0],
                        "to_layer": layer,
                        "mean_cosine": subspace_overlap(prior[1], basis),
                    }
                )
            previous[coherence_key] = (layer, basis)

            root = _profile_root(args.out, requested, multiple)
            relative = (
                Path("bases")
                / f"{group.replace('.', '__')}.tp{{tp_rank}}.safetensors"
            )
            path = root / str(relative).format(tp_rank=tp_rank)
            path.parent.mkdir(parents=True, exist_ok=True)
            replay_rows = min(8, matrix_cpu.shape[0])
            replay = matrix_cpu[:replay_rows].to(torch.bfloat16).contiguous()
            save_file(
                {"basis": basis.contiguous(), "replay": replay},
                str(path),
                metadata={
                    "group": group,
                    "modules": ",".join(members),
                    "tp_rank": str(tp_rank),
                    "requested_rank": str(requested),
                    "rank": str(chosen_rank),
                },
            )
            shard_artifacts[requested] = {
                "relative": relative,
                "rank": chosen_rank,
                "checksum": sha256_file(path),
                "phase_rows": dict(phase_rows),
                "rows": matrix.shape[0],
                "gram_max_abs_error": gram_error,
                "replay_rows": replay_rows,
            }
        return {
            "tp_rank": tp_rank,
            "input_dim": matrix.shape[1],
            "report": {
                "rows": matrix.shape[0],
                "width": matrix.shape[1],
                "modules": members,
                "phase_rows": dict(phase_rows),
                "cascade": cascade_report,
                "calibration_replay": analysis,
            },
            "artifacts": shard_artifacts,
        }

    for group in ordered_groups:
        report[group] = {"tp": {}}
        artifacts: dict[int, dict[int, dict]] = {
            requested: {} for requested in requested_ranks
        }
        input_dim = None
        # TP shards are independent; cascade state is keyed per shard so the
        # concurrent path stays deterministic.
        if args.devices:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=args.tp_size) as executor:
                shard_results = list(
                    executor.map(
                        lambda rank: calibrate_shard(group, rank),
                        range(args.tp_size),
                    )
                )
        else:
            shard_results = [
                calibrate_shard(group, tp_rank)
                for tp_rank in range(args.tp_size)
            ]
        for shard in shard_results:
            tp_rank = shard["tp_rank"]
            report[group]["tp"][str(tp_rank)] = shard["report"]
            if input_dim is None:
                input_dim = shard["input_dim"]
            elif input_dim != shard["input_dim"]:
                raise ValueError(f"input width differs across TP ranks for {group}")
            for requested in requested_ranks:
                artifacts[requested][tp_rank] = shard["artifacts"][requested]

        for requested in requested_ranks:
            shards = artifacts[requested]
            chosen_ranks = {item["rank"] for item in shards.values()}
            if len(chosen_ranks) != 1:
                raise ValueError(
                    f"basis rank differs across TP ranks for {group}: "
                    f"{sorted(chosen_ranks)}"
                )
            chosen_rank = chosen_ranks.pop()
            for module in groups[group]:
                entries_by_request[requested].append(
                    GSIEntry(
                        module=module,
                        family=routed_family(module) or classify_linear(module)
                        or "unknown",
                        layer=layer_from_module(module),
                        rank=chosen_rank,
                        epsilon=args.epsilon,
                        input_dim=int(input_dim),
                        output_dim=0,
                        basis_file=str(shards[0]["relative"]),
                        basis_sha256=shards[0]["checksum"],
                        image_file=(
                            f"images/{module.replace('.', '__')}"
                            ".tp{tp_rank}.safetensors"
                        ),
                        metadata={
                            "requested_rank": requested,
                            "basis_group": group,
                            "basis_group_members": groups[group],
                            "basis_sha256_by_rank": {
                                str(rank): item["checksum"]
                                for rank, item in shards.items()
                            },
                            "phase_rows_by_rank": {
                                str(rank): item["phase_rows"]
                                for rank, item in shards.items()
                            },
                            "rows_by_rank": {
                                str(rank): item["rows"]
                                for rank, item in shards.items()
                            },
                            "basis_gram_max_abs_error_by_rank": {
                                str(rank): item["gram_max_abs_error"]
                                for rank, item in shards.items()
                            },
                            "replay_rows_by_rank": {
                                str(rank): item["replay_rows"]
                                for rank, item in shards.items()
                            },
                        },
                    )
                )

    report_path = args.out / "analysis.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    overlap_path = args.out / "depth-coherence.json"
    overlap_path.write_text(
        json.dumps(
            {
                "description": (
                    "Mean cosine of principal angles between consecutive layer "
                    "bases spanning the same activation width and rank."
                ),
                "pairs": overlaps,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for requested, entries in entries_by_request.items():
        root = _profile_root(args.out, requested, multiple)
        profile = GSIProfile.from_entries(
            model_id=args.model_id,
            model_hash=args.model_hash,
            quantization=args.quantization,
            tp_size=args.tp_size,
            entries=entries,
            calibration={
                "captures": [str(path) for path in args.captures],
                "analysis": str(report_path),
                "depth_coherence": str(overlap_path),
                "requested_rank": requested,
                "epsilon": args.epsilon,
                "basis_scope": args.basis_scope,
                "cascade": bool(args.cascade),
                "cascade_iterations": args.cascade_iterations,
                "exact_rows_embedded_per_basis": 8,
                "disjoint_evaluation_required": True,
            },
        )
        profile.save(root / "profile.json")


if __name__ == "__main__":
    main()
