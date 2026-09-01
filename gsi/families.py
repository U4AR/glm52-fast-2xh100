from __future__ import annotations

import re

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def layer_from_module(module_name: str) -> int:
    match = _LAYER.search(module_name)
    return int(match.group(1)) if match else -1


def routed_family(module_name: str) -> str | None:
    """Name the two routed-expert stages captured outside the linear registry.

    The fused MoE backend owns these, so they never appear as `LinearBase`
    modules and `classify_linear` cannot see them.
    """
    for suffix in ("routed_gate_up", "routed_down"):
        if module_name.endswith(f".{suffix}"):
            return suffix
    return None


def classify_linear(module_name: str) -> str | None:
    name = module_name.lower()
    if "lm_head" in name or "embed_tokens" in name or "indexer" in name:
        return None
    if ".self_attn." in name:
        if "fused_qkv_a_proj_with_mqa" in name or "kv_a_proj_with_mqa" in name:
            return "attn_residual"
        if "q_b_proj" in name:
            return "q_latent"
        if "kv_b_proj" in name:
            return "kv_latent"
        if "o_proj" in name:
            return "attn_output"
    if ".mlp." in name or ".shared_experts." in name:
        if "gate_up_proj" in name or "gate_proj" in name or "up_proj" in name:
            return "mlp_residual"
        if "down_proj" in name:
            return "mlp_intermediate"
    return None

