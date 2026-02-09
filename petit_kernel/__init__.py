import torch
import enum

from . import ops
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
    return ops.mul_nvfp4_a16(a, b, s, global_scale, size_m, size_n, size_k, solution_id)


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
    return ops.mul_mxfp4_a16(a, b, s, global_scale, size_m, size_n, size_k, solution_id)


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
    "get_fp4_solutions",
    "DataType",
    "PetitSolutionHints",
]
