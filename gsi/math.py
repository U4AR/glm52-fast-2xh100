from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class GateResult:
    coefficients: object
    rho2: object
    fast_mask: object
    valid_mask: object


def project_and_gate(
    x,
    basis,
    epsilon: float,
    *,
    delta: float = 1e-12,
    gate_override: str = "profile",
) -> GateResult:
    """Project x and evaluate a stable orthogonal residual-energy gate."""
    import torch

    if x.shape[-1] != basis.shape[0]:
        raise ValueError(
            f"input/basis mismatch: input={x.shape[-1]} basis={basis.shape[0]}"
        )
    original_shape = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    compute_x = x2.float()
    compute_basis = basis.float()
    coefficients = compute_x @ compute_basis
    x_energy = (compute_x * compute_x).sum(dim=-1)
    g_energy = (coefficients * coefficients).sum(dim=-1)
    residual_energy = (x_energy - g_energy).clamp_min(0.0)
    rho2 = residual_energy / x_energy.clamp_min(delta)
    valid = (
        torch.isfinite(rho2)
        & torch.isfinite(x_energy)
        & (x_energy > delta)
    )
    if gate_override == "all_slow":
        fast = torch.zeros_like(valid)
    elif gate_override == "always_fast":
        fast = valid
    elif gate_override == "profile":
        fast = valid & (rho2 < float(epsilon) ** 2)
    else:
        raise ValueError(f"unknown gate override: {gate_override}")
    return GateResult(
        coefficients=coefficients.to(dtype=x.dtype).reshape(*original_shape, -1),
        rho2=rho2.reshape(original_shape),
        fast_mask=fast.reshape(original_shape),
        valid_mask=valid.reshape(original_shape),
    )


def apply_gated_linear(
    x,
    basis,
    image,
    epsilon: float,
    baseline: Callable,
    *,
    bias=None,
    gate_override: str = "profile",
):
    """Apply Mg to fast rows and the supplied baseline to slow rows."""
    import torch

    gate = project_and_gate(
        x, basis, epsilon, gate_override=gate_override
    )
    x2 = x.reshape(-1, x.shape[-1])
    g2 = gate.coefficients.reshape(-1, gate.coefficients.shape[-1])
    mask = gate.fast_mask.reshape(-1)
    if image.shape[-1] != g2.shape[-1]:
        raise ValueError(
            f"image/rank mismatch: image={image.shape[-1]} rank={g2.shape[-1]}"
        )
    out_dim = image.shape[0]
    if not bool(mask.any()):
        return baseline(x), gate

    approx = g2 @ image.to(dtype=g2.dtype).transpose(0, 1)
    if bias is not None:
        approx = approx + bias
    if bool(mask.all()):
        return approx.reshape(*x.shape[:-1], out_dim), gate

    slow = baseline(x2[~mask])
    if isinstance(slow, tuple):
        slow = slow[0]
    output = approx
    output[~mask] = slow
    return output.reshape(*x.shape[:-1], out_dim), gate

