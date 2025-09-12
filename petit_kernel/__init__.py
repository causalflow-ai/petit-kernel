import torch
import enum

from . import ops
from .ops import PetitSolutionHints


class DataType(enum.Enum):
    int4 = 0
    float8_e4m3 = 1
    float8_e5m2fn = 2
    float4_e2m1 = 3
    float16 = 4
    bfloat16 = 5


def repack_nvfp4(qw: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
    return ops.repack_nvfp4(qw, size_n, size_k)


def process_nvfp4_scales(
    scales: torch.Tensor, size_n: int, size_k: int
) -> torch.Tensor:
    return ops.process_nvfp4_scales(scales, size_n, size_k)


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


def get_fp4_solutions(
    size_m: int, size_n: int, size_k: int, a_type: torch.dtype, c_type: torch.dtype
) -> list[int]:
    return ops.get_fp4_solutions(size_m, size_n, size_k, a_type, c_type)


__all__ = [
    "repack_nvfp4",
    "process_nvfp4_scales",
    "mul_nvfp4_a16",
    "get_fp4_solutions",
    "DataType",
    "PetitSolutionHints",
]
