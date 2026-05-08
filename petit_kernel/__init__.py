import enum

import torch

from . import ops
from .moe_mxfp4 import repack_moe_mxfp4_kernel_layout
from .ops import PetitSolutionHints


class DataType(enum.Enum):
    int4 = 0
    float8_e4m3fn = 1
    float4_e2m1 = 2
    float16 = 3
    bfloat16 = 4
    float8_e5m2fn = 5
    mxfloat4_e2m1 = 6


def repack_nvfp4(qw: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
    return ops.repack_nvfp4(qw, size_n, size_k)


def process_nvfp4_scales(
    scales: torch.Tensor, size_n: int, size_k: int
) -> torch.Tensor:
    return ops.process_nvfp4_scales(scales, size_n, size_k)


def repack_mxfp4(qw: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
    return ops.repack_nvfp4(qw, size_n, size_k)


def process_mxfp4_scales(
    scales: torch.Tensor, size_n: int, size_k: int
) -> torch.Tensor:
    return ops.process_mxfp4_scales(scales, size_n, size_k)


def mul_nvfp4_a16(
    a: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    global_scale: torch.Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
    solution_id: int,
) -> torch.Tensor:
    return ops.mul_nvfp4_a16(
        a, b, s, global_scale, size_m, size_n, size_k, solution_id
    )


def mul_mxfp4_a16(
    a: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    global_scale: torch.Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
    solution_id: int,
) -> torch.Tensor:
    return ops.mul_mxfp4_a16(
        a, b, s, global_scale, size_m, size_n, size_k, solution_id
    )


def fused_moe_fp8_blockscale_g1u1(
    input_q: torch.Tensor,
    w13_q: torch.Tensor,
    w2_q: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    input_scale: torch.Tensor,
    fc1_scale: torch.Tensor,
    fc2_scale: torch.Tensor,
    num_persistent_tgs: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return ops.fused_moe_fp8_blockscale_g1u1(
        input_q,
        w13_q,
        w2_q,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale,
        fc1_scale,
        fc2_scale,
        num_persistent_tgs,
        out,
    )


def fused_moe_fp8_blockscale_g1u1_mxfp4(
    input_q: torch.Tensor,
    w13_q: torch.Tensor,
    w2_q: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    input_scale: torch.Tensor,
    fc1_scale: torch.Tensor,
    fc2_scale: torch.Tensor,
    num_persistent_tgs: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    # The fused MXFP4 kernel consumes weights and scales in the MoE Petit byte
    # layout. Use `repack_moe_mxfp4_kernel_layout` on native checkpoint tensors.
    input_q_arg = input_q
    if input_q.dtype != torch.uint8:
        if input_q.element_size() != 1:
            raise TypeError("input_q must be uint8 or a 1-byte float8 tensor")
        input_q_arg = input_q.view(torch.uint8)

    return ops.fused_moe_fp8_blockscale_g1u1_mxfp4(
        input_q_arg,
        w13_q,
        w2_q,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale,
        fc1_scale,
        fc2_scale,
        num_persistent_tgs,
        out,
    )


def get_fp4_solutions(
    size_m: int, size_n: int, size_k: int, a_type: torch.dtype, c_type: torch.dtype
) -> list[int]:
    return ops.get_fp4_solutions(size_m, size_n, size_k, a_type, c_type)


__all__ = [
    "repack_nvfp4",
    "repack_mxfp4",
    "process_nvfp4_scales",
    "process_mxfp4_scales",
    "mul_nvfp4_a16",
    "mul_mxfp4_a16",
    "fused_moe_fp8_blockscale_g1u1",
    "fused_moe_fp8_blockscale_g1u1_mxfp4",
    "get_fp4_solutions",
    "repack_moe_mxfp4_kernel_layout",
    "DataType",
    "PetitSolutionHints",
]
