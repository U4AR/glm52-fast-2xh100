from __future__ import annotations

import os


def tensor_parallel_rank() -> int:
    """Return the SGLang tensor-parallel rank without relying on worker env."""
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    for name in ("LOCAL_RANK", "RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0
