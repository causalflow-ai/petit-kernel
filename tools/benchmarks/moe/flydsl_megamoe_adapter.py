"""Adapter for benchmarking AITER's packaged MegaMoEV2 path."""

from __future__ import annotations

import inspect
import os
from contextlib import suppress
from typing import Any

import torch
import torch.distributed as dist

_MORI_SHMEM: Any | None = None


def initialize_flydsl_runtime() -> None:
    """Initialize MORI on the default torch process group."""
    global _MORI_SHMEM
    if _MORI_SHMEM is None:
        os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "8G")
        import flydsl
        import mori.shmem as ms
        import torch._C._distributed_c10d as c10d
        from flydsl._mlir.dialects import rocdl

        aux_parameter = inspect.signature(rocdl.RawPtrBufferLoadOp.__init__).parameters[
            "aux"
        ]
        if aux_parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise RuntimeError(
                f"The installed runtime {flydsl.__version__} has an incompatible "
                "ROCDL binding; install a compatible flydsl==0.3.1 package"
            )
        if not dist.is_initialized():
            raise RuntimeError("initialize the torch process group before MORI")
        c10d._register_process_group("default", dist.group.WORLD)
        ms.shmem_torch_process_group_init("default")
        _MORI_SHMEM = ms


def finalize_flydsl_runtime() -> None:
    global _MORI_SHMEM
    if _MORI_SHMEM is None:
        return
    with suppress(Exception):
        _MORI_SHMEM.shmem_finalize()
    _MORI_SHMEM = None


def create_flydsl_megamoe(
    *,
    rank: int,
    world_size: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    max_tokens_per_rank: int,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
):
    if _MORI_SHMEM is None:
        raise RuntimeError("initialize_flydsl_runtime must be called first")
    from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

    return MegaMoEV2(
        rank=rank,
        world_size=world_size,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant="a8w4",
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        max_tok_per_rank=max_tokens_per_rank,
    )
