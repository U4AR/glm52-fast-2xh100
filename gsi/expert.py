from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .math import project_and_gate


@dataclass
class ExpertGateStats:
    gate_up_fast: object
    down_fast: object
    gate_up_rho2: object
    down_rho2: object


def functional_expert_forward(
    x,
    topk_ids,
    topk_weights,
    *,
    gate_up_basis,
    gate_up_images,
    down_bases,
    down_images,
    gate_up_epsilon: float,
    down_epsilon: float,
    full_fallback: Callable,
    down_fallback: Callable,
    gate_override: str = "profile",
):
    """Reference two-stage GSI MoE implementation.

    This is intentionally eager and simple.  It defines the required semantics
    for the fused GLM/KT kernel: gate/up shares a token projection across routed
    experts; down is gated independently for each token/expert activation.
    """
    import torch
    import torch.nn.functional as F

    if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("x, topk_ids, and topk_weights must be rank-2 tensors")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != x.shape[0]:
        raise ValueError("routing tensors do not match the input batch")
    if gate_up_images.ndim != 3 or down_images.ndim != 3:
        raise ValueError("expert images must have [experts, output, rank] shape")

    gate = project_and_gate(
        x,
        gate_up_basis,
        gate_up_epsilon,
        gate_override=gate_override,
    )
    batch, routed = topk_ids.shape
    hidden = down_images.shape[1]
    output = torch.zeros((batch, hidden), dtype=x.dtype, device=x.device)
    down_fast = torch.zeros(
        (batch, routed), dtype=torch.bool, device=x.device
    )
    down_rho2 = torch.full(
        (batch, routed), float("inf"), dtype=torch.float32, device=x.device
    )

    for row in range(batch):
        for slot in range(routed):
            expert = int(topk_ids[row, slot].item())
            if expert < 0:
                continue
            route_weight = topk_weights[row, slot].to(dtype=x.dtype)
            if not bool(gate.fast_mask[row]):
                contribution = full_fallback(expert, x[row : row + 1])[0]
                output[row].add_(contribution, alpha=float(route_weight))
                continue

            coefficients = gate.coefficients[row]
            gate_up = gate_up_images[expert] @ coefficients
            if gate_up.numel() % 2:
                raise ValueError("gate/up image output must have even width")
            gate_part, up_part = gate_up.chunk(2, dim=-1)
            intermediate = F.silu(gate_part) * up_part

            basis = (
                down_bases[expert]
                if down_bases.ndim == 3
                else down_bases
            )
            down_gate = project_and_gate(
                intermediate.unsqueeze(0),
                basis,
                down_epsilon,
                gate_override=gate_override,
            )
            down_fast[row, slot] = down_gate.fast_mask[0]
            down_rho2[row, slot] = down_gate.rho2[0]
            if bool(down_gate.fast_mask[0]):
                contribution = (
                    down_images[expert] @ down_gate.coefficients[0]
                )
            else:
                contribution = down_fallback(
                    expert, intermediate.unsqueeze(0)
                )[0]
            output[row].add_(contribution, alpha=float(route_weight))

    return output, ExpertGateStats(
        gate_up_fast=gate.fast_mask,
        down_fast=down_fast,
        gate_up_rho2=gate.rho2,
        down_rho2=down_rho2,
    )

