"""Routed-expert image construction (paper coverage of the MLP up/down maps).

arXiv 2605.03109 covers "all linear layers" of the transformer.  In GLM-5.2 the
MLP up/down maps of every layer past `first_k_dense_replace` are the 256 routed
experts, which do not present a `LinearBase.quant_method.apply` interface: they
are consumed by a fused grouped GEMM.  Images for them therefore cannot be built
by the generic path in `runtime._build_images`.

Both routed stages get an image here:

* **gate/up** ``M13_e = W13_e V``, where ``V`` is the layer's shared basis over
  the MoE input (the residual stream).  One projection ``V'x`` per token is
  shared by every expert the token routes to.
* **down** ``M2_e = W2_e U``, where ``U`` is the layer's shared basis over the
  post-SiLU intermediate.  The paper fits one basis per layer, so ``U`` is shared
  across all experts rather than fitted per expert; `expert.functional_expert_forward`
  already accepts a rank-2 (shared) or rank-3 (per-expert) down basis.

Images are built through the deployed quantized arithmetic, not through a
dequantized copy of the weights, so the cached image carries exactly the same
rounding as the slow path it replaces.  The grouped-GEMM backend is injectable so
the index/chunking logic is testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExpertImageResult:
    #: expert index -> image tensor, ``[out_dim, rank]``
    images: dict[int, object] = field(default_factory=dict)
    #: per-expert relative L2 of image-vs-kernel on the embedded replay rows
    validation: dict[int, float] = field(default_factory=dict)


def plan_expert_chunks(experts: list[int], rank: int, budget_rows: int) -> list[list[int]]:
    """Split experts so no grouped-GEMM call exceeds ``budget_rows`` rows.

    Each expert contributes ``rank`` rows (one per basis column) to the batch.
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    if budget_rows < rank:
        raise ValueError(
            f"row budget {budget_rows} cannot fit a single expert of rank {rank}"
        )
    per_chunk = max(1, budget_rows // rank)
    return [
        list(experts[start : start + per_chunk])
        for start in range(0, len(experts), per_chunk)
    ]


def cutlass_gate_up_projection(layer, method, rows, expert_ids):
    """Run ``W13_e @ rows`` for each row's expert through the deployed kernel.

    ``rows`` is ``[N, hidden]`` bf16 and ``expert_ids`` is ``[N]``.  Returns
    ``[N, 2 * intermediate]`` bf16, in the same row order as the input.
    """
    import torch
    from sgl_kernel import cutlass_w4a8_moe_mm

    from sglang.srt.layers.moe.ep_moe.kernels import (
        cutlass_w4_run_moe_ep_preproess,
        pre_reorder_for_cutlass_moe,
    )

    w1_q = layer.w13_weight
    num_local_experts = w1_q.shape[0]
    hidden = w1_q.shape[2] * 2
    intermediate2 = w1_q.shape[1]
    if rows.shape[1] != hidden:
        raise ValueError(
            f"basis width {rows.shape[1]} does not match expert hidden {hidden}"
        )

    topk_ids = expert_ids.to(dtype=torch.int32).reshape(-1, 1)
    count = topk_ids.shape[0]
    src2dst = cutlass_w4_run_moe_ep_preproess(topk_ids)

    gateup_input = torch.empty(
        (count, hidden), device=rows.device, dtype=torch.float8_e4m3fn
    )
    pre_reorder_for_cutlass_moe(
        rows,
        gateup_input,
        src2dst,
        topk_ids,
        layer.w13_input_scale,
        num_local_experts,
        1,
        count,
        hidden,
    )

    # get_cutlass_w4a8_moe_mm_data fills the grouped-GEMM problem descriptors
    # from the (sorted) expert assignment; reuse it rather than recomputing the
    # offsets so the descriptors match the kernel's expectations exactly.
    from sgl_kernel import get_cutlass_w4a8_moe_mm_data

    expert_offsets = torch.empty(
        num_local_experts + 1, dtype=torch.int32, device=rows.device
    )
    problem_sizes1 = torch.empty(
        (num_local_experts, 3), dtype=torch.int32, device=rows.device
    )
    problem_sizes2 = torch.empty(
        (num_local_experts, 3), dtype=torch.int32, device=rows.device
    )
    a_map = torch.empty(count, dtype=torch.int32, device=rows.device)
    c_map = torch.empty(count, dtype=torch.int32, device=rows.device)
    get_cutlass_w4a8_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_local_experts,
        layer.w2_weight.shape[2] * 2,
        hidden,
    )

    out = torch.empty(
        (count, intermediate2), device=rows.device, dtype=torch.bfloat16
    )
    cutlass_w4a8_moe_mm(
        out,
        gateup_input,
        w1_q,
        layer.w13_input_scale.float(),
        layer.w13_weight_scale_inv,
        expert_offsets[:-1],
        problem_sizes1,
        method.a_strides1,
        method.b_strides1,
        method.c_strides1,
        method.s_strides13,
        128,
        1,
    )
    # `out` is in expert-sorted order; src2dst maps source row -> sorted row.
    return out.index_select(0, src2dst.to(torch.long))


def cutlass_down_projection(layer, method, rows, expert_ids):
    """Run ``W2_e @ rows`` for each row's expert through the deployed kernel.

    ``rows`` is ``[N, intermediate]`` bf16; returns ``[N, hidden]`` bf16.
    """
    import torch
    from sgl_kernel import cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data

    from sglang.srt.layers.moe.ep_moe.kernels import cutlass_w4_run_moe_ep_preproess
    from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8

    w2_q = layer.w2_weight
    num_local_experts = w2_q.shape[0]
    hidden = w2_q.shape[1]
    intermediate = w2_q.shape[2] * 2
    if rows.shape[1] != intermediate:
        raise ValueError(
            f"basis width {rows.shape[1]} does not match expert intermediate "
            f"{intermediate}"
        )

    topk_ids = expert_ids.to(dtype=torch.int32).reshape(-1, 1)
    count = topk_ids.shape[0]
    src2dst = cutlass_w4_run_moe_ep_preproess(topk_ids)
    order = torch.empty_like(src2dst, dtype=torch.long)
    order.scatter_(0, src2dst.to(torch.long), torch.arange(count, device=rows.device))

    sorted_rows = rows.index_select(0, order)
    quantized, _ = per_tensor_quant_fp8(sorted_rows, layer.w2_input_scale.float())

    expert_offsets = torch.empty(
        num_local_experts + 1, dtype=torch.int32, device=rows.device
    )
    problem_sizes1 = torch.empty(
        (num_local_experts, 3), dtype=torch.int32, device=rows.device
    )
    problem_sizes2 = torch.empty(
        (num_local_experts, 3), dtype=torch.int32, device=rows.device
    )
    a_map = torch.empty(count, dtype=torch.int32, device=rows.device)
    c_map = torch.empty(count, dtype=torch.int32, device=rows.device)
    get_cutlass_w4a8_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_local_experts,
        intermediate,
        hidden,
    )

    out = torch.empty((count, hidden), device=rows.device, dtype=torch.bfloat16)
    cutlass_w4a8_moe_mm(
        out,
        quantized,
        w2_q,
        layer.w2_input_scale.float(),
        layer.w2_weight_scale_inv,
        expert_offsets[:-1],
        problem_sizes2,
        method.a_strides2,
        method.b_strides2,
        method.c_strides2,
        method.s_strides2,
        128,
        1,
    )
    return out.index_select(0, src2dst.to(torch.long))


def build_stage_images(
    basis,
    experts,
    projection,
    *,
    budget_rows: int = 1 << 16,
) -> ExpertImageResult:
    """Build ``M_e = W_e @ basis`` for each expert via a grouped projection.

    ``basis`` is ``[width, rank]``; ``projection(rows, expert_ids)`` applies each
    row's expert weight.  The returned images are ``[out_dim, rank]`` so that a
    fast row is ``image @ coefficients``, matching `math.apply_gated_linear`.
    """
    import torch

    if basis.ndim != 2:
        raise ValueError("basis must be [width, rank]")
    width, rank = basis.shape
    experts = list(experts)
    result = ExpertImageResult()
    # Rows of the projection batch are basis *columns*: applying W to each basis
    # vector yields the image columns.
    vectors = basis.transpose(0, 1).contiguous()

    for chunk in plan_expert_chunks(experts, rank, budget_rows):
        rows = vectors.repeat(len(chunk), 1)
        expert_ids = torch.tensor(
            [expert for expert in chunk for _ in range(rank)],
            device=rows.device,
            dtype=torch.int32,
        )
        projected = projection(rows, expert_ids)
        if projected.shape[0] != rows.shape[0]:
            raise ValueError("projection returned the wrong number of rows")
        out_dim = projected.shape[1]
        for index, expert in enumerate(chunk):
            block = projected[index * rank : (index + 1) * rank]
            result.images[expert] = (
                block.transpose(0, 1).to(dtype=torch.bfloat16).contiguous()
            )
            if result.images[expert].shape != (out_dim, rank):
                raise ValueError(
                    f"expert {expert} image has shape "
                    f"{tuple(result.images[expert].shape)}, expected {(out_dim, rank)}"
                )
    return result


def validate_expert_image(image, basis, replay_rows, exact_projection) -> dict:
    """Compare ``image @ (V' x)`` against the exact kernel on calibration rows."""
    import torch

    with torch.no_grad():
        coefficients = replay_rows.float() @ basis.float()
        fast = coefficients @ image.float().transpose(0, 1)
        exact = exact_projection(replay_rows).float()
        projected = coefficients @ basis.float().transpose(0, 1)
        exact_projected = exact_projection(projected.to(replay_rows.dtype)).float()

        def relative_l2(reference, candidate) -> float:
            denominator = reference.square().sum().sqrt().clamp_min(1e-12)
            return float((reference - candidate).square().sum().sqrt() / denominator)

        energy = replay_rows.float().square().sum(dim=-1).clamp_min(1e-12)
        residual = (
            (replay_rows.float() - projected).square().sum(dim=-1) / energy
        ).clamp_min(0).sqrt()
        return {
            "rows": int(replay_rows.shape[0]),
            "rank": int(basis.shape[1]),
            "input_mean_rho": float(residual.mean()),
            "input_max_rho": float(residual.max()),
            "raw_exact_vs_image_relative_l2": relative_l2(exact, fast),
            "projected_exact_vs_image_relative_l2": relative_l2(exact_projected, fast),
        }
