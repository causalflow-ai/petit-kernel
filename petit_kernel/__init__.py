import enum
import functools
import os
from dataclasses import dataclass, field

import torch

from . import ops
from .moe_mxfp4 import (
    MoeKernelLayout,
    repack_moe_kernel_layout,
)
from .ops import PetitSolutionHints


class DataType(enum.Enum):
    int4 = 0
    float8_e4m3fn = 1
    float4_e2m1 = 2
    float16 = 3
    bfloat16 = 4
    float8_e5m2fn = 5
    mxfloat4_e2m1 = 6


class _FusedMoeDataType(enum.IntEnum):
    none = 0
    mxfp4 = 1
    nvfp4 = 2
    channel_scale_fp8 = 3
    blockscale_fp8 = 4
    bf16 = 5


class _FusedMoeWeightOrdering(enum.IntEnum):
    native_mxfp4 = 0
    petit_mxfp4 = 1
    petit_fp8 = 2


class _FusedMoeStages(enum.IntEnum):
    one_stage = 0
    two_stage = 1


class _FusedMoeMfmaShape(enum.IntEnum):
    mfma_fp8_16x16x32 = 0
    mfma_bf16_mxfp4 = 1
    mfma_scale_fp4_mxfp4 = 2


class _FusedMoeActivationFunction(enum.IntEnum):
    silu_dot = 0
    openai_swiglu = 1


class _FusedMoeStage1Buffering(enum.IntEnum):
    single_buffer = 0
    double_buffer = 1


class _FusedMoeStage1TileShape(enum.IntEnum):
    m32_n256 = 0
    m64_n512 = 1


class _FusedMoeWeightLoadPolicy(enum.IntEnum):
    cached = 0
    non_temporal = 1


def _make_fused_moe_base_solution_id(
    activation_type: _FusedMoeDataType | int,
    weight_type: _FusedMoeDataType | int,
    bias_type: _FusedMoeDataType | int,
    weight_ordering: _FusedMoeWeightOrdering | int,
    mfma: _FusedMoeMfmaShape | int,
    stages: _FusedMoeStages | int,
    activation: _FusedMoeActivationFunction | int,
    stage1_buffering: _FusedMoeStage1Buffering | int,
    weight_load_policy: _FusedMoeWeightLoadPolicy | int = (
        _FusedMoeWeightLoadPolicy.cached
    ),
) -> int:
    return (
        (int(_FusedMoeDataType(activation_type)) & 0xF)
        | ((int(_FusedMoeDataType(weight_type)) & 0xF) << 4)
        | ((int(_FusedMoeDataType(bias_type)) & 0xF) << 8)
        | ((int(_FusedMoeWeightOrdering(weight_ordering)) & 0x3) << 12)
        | ((int(_FusedMoeMfmaShape(mfma)) & 0x3) << 14)
        | ((int(_FusedMoeStages(stages)) & 0xF) << 16)
        | ((int(_FusedMoeActivationFunction(activation)) & 0x7) << 20)
        | ((int(_FusedMoeStage1Buffering(stage1_buffering)) & 0x1) << 23)
        | ((int(_FusedMoeWeightLoadPolicy(weight_load_policy)) & 0x1) << 40)
    )


def _with_fused_moe_shape(solution_id: int, dim: int, inter_dim: int) -> int:
    dim = int(dim)
    inter_dim = int(inter_dim)
    if dim <= 0 or inter_dim <= 0:
        raise ValueError("MoE dimensions must be positive")
    if dim % 64 != 0 or inter_dim % 64 != 0:
        raise ValueError("MoE dimensions must be divisible by 64")
    if dim // 64 > 0xFF or inter_dim // 64 > 0xFF:
        raise ValueError("MoE dimensions exceed the solution-ID encoding")
    shape_mask = 0xFFFF << 24
    return (
        (int(solution_id) & ~shape_mask)
        | ((dim // 64) << 24)
        | ((inter_dim // 64) << 32)
    )


def _with_fused_moe_stage1_tile_shape(
    solution_id: int, shape: _FusedMoeStage1TileShape | int
) -> int:
    shape_mask = 1 << 41
    return (int(solution_id) & ~shape_mask) | (
        (int(_FusedMoeStage1TileShape(shape)) & 0x1) << 41
    )


def _with_fused_moe_weight_load_policy(
    solution_id: int, policy: _FusedMoeWeightLoadPolicy | int
) -> int:
    policy_mask = 1 << 40
    return (int(solution_id) & ~policy_mask) | (
        (int(_FusedMoeWeightLoadPolicy(policy)) & 0x1) << 40
    )


def _make_fused_moe_solution_id(
    activation_type: _FusedMoeDataType | int,
    weight_type: _FusedMoeDataType | int,
    bias_type: _FusedMoeDataType | int,
    weight_ordering: _FusedMoeWeightOrdering | int,
    mfma: _FusedMoeMfmaShape | int,
    stages: _FusedMoeStages | int,
    activation: _FusedMoeActivationFunction | int,
    stage1_buffering: _FusedMoeStage1Buffering | int,
    dim: int,
    inter_dim: int,
) -> int:
    base = _make_fused_moe_base_solution_id(
        activation_type,
        weight_type,
        bias_type,
        weight_ordering,
        mfma,
        stages,
        activation,
        stage1_buffering,
    )
    return _with_fused_moe_shape(base, dim, inter_dim)


_FUSED_MOE_FP8_BLOCKSCALE_SOLUTION_ID = _make_fused_moe_base_solution_id(
    _FusedMoeDataType.channel_scale_fp8,
    _FusedMoeDataType.blockscale_fp8,
    _FusedMoeDataType.none,
    _FusedMoeWeightOrdering.petit_fp8,
    _FusedMoeMfmaShape.mfma_fp8_16x16x32,
    _FusedMoeStages.one_stage,
    _FusedMoeActivationFunction.silu_dot,
    _FusedMoeStage1Buffering.double_buffer,
)
_FUSED_MOE_FP8_BLOCKSCALE_MXFP4_SOLUTION_ID = _make_fused_moe_base_solution_id(
    _FusedMoeDataType.channel_scale_fp8,
    _FusedMoeDataType.mxfp4,
    _FusedMoeDataType.none,
    _FusedMoeWeightOrdering.petit_mxfp4,
    _FusedMoeMfmaShape.mfma_fp8_16x16x32,
    _FusedMoeStages.one_stage,
    _FusedMoeActivationFunction.silu_dot,
    _FusedMoeStage1Buffering.single_buffer,
)
_FUSED_MOE_BF16_MXFP4_BIAS_SOLUTION_ID = _make_fused_moe_base_solution_id(
    _FusedMoeDataType.bf16,
    _FusedMoeDataType.mxfp4,
    _FusedMoeDataType.bf16,
    _FusedMoeWeightOrdering.native_mxfp4,
    _FusedMoeMfmaShape.mfma_bf16_mxfp4,
    _FusedMoeStages.one_stage,
    _FusedMoeActivationFunction.openai_swiglu,
    _FusedMoeStage1Buffering.double_buffer,
)
_FUSED_MOE_MXFP4_MXFP4_BIAS_SOLUTION_ID = _make_fused_moe_base_solution_id(
    _FusedMoeDataType.mxfp4,
    _FusedMoeDataType.mxfp4,
    _FusedMoeDataType.bf16,
    _FusedMoeWeightOrdering.native_mxfp4,
    _FusedMoeMfmaShape.mfma_scale_fp4_mxfp4,
    _FusedMoeStages.one_stage,
    _FusedMoeActivationFunction.openai_swiglu,
    _FusedMoeStage1Buffering.double_buffer,
)

_FUSED_MOE_TWO_STAGE_MXFP4_BIAS_SOLUTION_ID = _make_fused_moe_solution_id(
    _FusedMoeDataType.mxfp4,
    _FusedMoeDataType.mxfp4,
    _FusedMoeDataType.bf16,
    _FusedMoeWeightOrdering.native_mxfp4,
    _FusedMoeMfmaShape.mfma_scale_fp4_mxfp4,
    _FusedMoeStages.two_stage,
    _FusedMoeActivationFunction.openai_swiglu,
    _FusedMoeStage1Buffering.double_buffer,
    3072,
    3072,
)

_FUSED_MOE_TWO_STAGE_MXFP4_BIAS_M64_N512_SOLUTION_ID = (
    _with_fused_moe_stage1_tile_shape(
        _FUSED_MOE_TWO_STAGE_MXFP4_BIAS_SOLUTION_ID,
        _FusedMoeStage1TileShape.m64_n512,
    )
)

_FUSED_MOE_TWO_STAGE_MXFP4_SILU_7168X2048_SOLUTION_ID = (
    _make_fused_moe_solution_id(
        _FusedMoeDataType.mxfp4,
        _FusedMoeDataType.mxfp4,
        _FusedMoeDataType.none,
        _FusedMoeWeightOrdering.native_mxfp4,
        _FusedMoeMfmaShape.mfma_scale_fp4_mxfp4,
        _FusedMoeStages.two_stage,
        _FusedMoeActivationFunction.silu_dot,
        _FusedMoeStage1Buffering.double_buffer,
        7168,
        2048,
    )
)

_FUSED_MOE_TWO_STAGE_MXFP4_SILU_7168X3072_SOLUTION_ID = (
    _make_fused_moe_solution_id(
        _FusedMoeDataType.mxfp4,
        _FusedMoeDataType.mxfp4,
        _FusedMoeDataType.none,
        _FusedMoeWeightOrdering.native_mxfp4,
        _FusedMoeMfmaShape.mfma_scale_fp4_mxfp4,
        _FusedMoeStages.two_stage,
        _FusedMoeActivationFunction.silu_dot,
        _FusedMoeStage1Buffering.double_buffer,
        7168,
        3072,
    )
)

_FUSED_MOE_TWO_STAGE_PROFILES = {
    (3072, 3072, 4, "swiglu"): _FUSED_MOE_TWO_STAGE_MXFP4_BIAS_SOLUTION_ID,
    # Current vLLM runs the eight routed DeepSeek-V3 experts here and
    # evaluates the shared expert separately.  Keep the nine-route entry for
    # integrations that fuse the shared expert into this invocation.
    (7168, 2048, 8, "silu"): (
        _FUSED_MOE_TWO_STAGE_MXFP4_SILU_7168X2048_SOLUTION_ID
    ),
    (7168, 2048, 9, "silu"): (
        _FUSED_MOE_TWO_STAGE_MXFP4_SILU_7168X2048_SOLUTION_ID
    ),
    (7168, 3072, 7, "silu"): (
        _FUSED_MOE_TWO_STAGE_MXFP4_SILU_7168X3072_SOLUTION_ID
    ),
}


def _config_value_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).lower()


@dataclass(frozen=True)
class Moe1StageConfig:
    token: int
    model_dim: int
    inter_dim: int
    expert: int
    topk: int
    block_m: int = 32
    ksplit: int = 0
    run_1stage: bool = field(init=False, default=True)
    _solution_id: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_solution_id",
            _with_fused_moe_shape(
                _FUSED_MOE_MXFP4_MXFP4_BIAS_SOLUTION_ID,
                self.model_dim,
                self.inter_dim,
            ),
        )

    def stage1(
        self,
        out: torch.Tensor,
        input_q: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        sorted_weights: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        input_scale: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        *,
        num_persistent_tgs: int = 0,
        bias1: torch.Tensor | None = None,
        bias2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_q.size(0) != self.token:
            raise ValueError("input_q token dimension does not match config")
        if out.shape != (self.token, self.model_dim):
            raise ValueError("out shape does not match config")
        return ops.fmoe_matmul_1stage(
            out,
            input_q,
            w1_q,
            w2_q,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            self.topk,
            input_scale,
            w1_scale,
            w2_scale,
            self._solution_id,
            int(num_persistent_tgs),
            bias1,
            bias2,
        )


@dataclass(frozen=True)
class Moe2StageConfig:
    token: int
    model_dim: int
    inter_dim: int
    expert: int
    topk: int
    block_m: int = 32
    ksplit: int = 0
    run_1stage: bool = field(init=False, default=False)
    _solution_id: int = field(
        repr=False,
        compare=False,
        default=_FUSED_MOE_TWO_STAGE_MXFP4_BIAS_SOLUTION_ID,
    )

    def workspace_size(self, max_num_m_blocks: int) -> int:
        return int(
            ops.fmoe_matmul_2stage_workspace_size(
                int(max_num_m_blocks), self.inter_dim, self._solution_id
            )
        )

    def intermediate_views(
        self, intermediate: torch.Tensor, max_num_m_blocks: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_num_m_blocks = int(max_num_m_blocks)
        payload_capacity = max_num_m_blocks * self.block_m * self.inter_dim // 2
        payload_bytes = self.token * self.topk * self.inter_dim // 2
        scale_rows = ((max_num_m_blocks * self.block_m + 255) // 256) * 256
        scale_cols = ((self.inter_dim // 32 + 7) // 8) * 8
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)
        payload = intermediate[:payload_bytes].view(fp4_dtype).view(
            self.token, self.topk, self.inter_dim // 2
        )
        scales = intermediate[
            payload_capacity : payload_capacity + scale_rows * scale_cols
        ].view(e8m0_dtype).view(scale_rows, scale_cols)
        return payload, scales

    def stage1(
        self,
        intermediate: torch.Tensor,
        input_q: torch.Tensor,
        w1_q: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        input_scale: torch.Tensor,
        w1_scale: torch.Tensor,
        *,
        num_persistent_tgs: int = 0,
        bias1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_q.size(0) != self.token:
            raise ValueError("input_q token dimension does not match config")
        return ops.fmoe_matmul_2stage_stage1(
            intermediate,
            input_q,
            w1_q,
            sorted_token_ids,
            sorted_expert_ids,
            num_valid_ids,
            self.topk,
            input_scale,
            w1_scale,
            self.inter_dim,
            self.expert,
            self._solution_id,
            int(num_persistent_tgs),
            bias1,
        )

    def stage2(
        self,
        out: torch.Tensor,
        intermediate: torch.Tensor,
        w2_q: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        sorted_weights: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        w2_scale: torch.Tensor,
        *,
        num_persistent_tgs: int = 0,
        bias2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Accumulate stage 2 into a caller-provided, zeroed BF16 output."""
        if out.shape != (self.token, self.model_dim):
            raise ValueError("out shape does not match config")
        return ops.fmoe_matmul_2stage_stage2(
            out,
            intermediate,
            w2_q,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            self.topk,
            w2_scale,
            self.inter_dim,
            self.expert,
            self._solution_id,
            int(num_persistent_tgs),
            bias2,
        )


@functools.lru_cache(maxsize=2048)
def _get_2stage_cfgs_cached(
    token: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    dtype: object,
    q_dtype_a: object,
    q_dtype_w: object,
    q_type: object,
    use_g1u1: bool,
    activation: object,
    doweight_stage1: bool,
    hidden_pad: int,
    intermediate_pad: int,
    is_shuffled: bool = True,
    gate_mode: object = "separated",
    is_ep: bool = False,
    has_stage2_bias: bool = False,
    two_stage_threshold: int = 128,
    is_gfx950: bool = False,
) -> Moe1StageConfig | Moe2StageConfig:
    del has_stage2_bias, is_ep
    q_type_name = _config_value_name(q_type)
    activation_name = _config_value_name(activation)
    gate_mode_name = _config_value_name(gate_mode)
    q_dtype_a_name = _config_value_name(q_dtype_a)
    q_dtype_w_name = _config_value_name(q_dtype_w)
    activation_profile = (
        "swiglu"
        if "swiglu" in activation_name
        else "silu"
        if "silu" in activation_name
        else None
    )
    supported = (
        dtype == torch.bfloat16
        and (
            q_dtype_a == torch.uint8
            or "fp4" in q_dtype_a_name
            or "float4" in q_dtype_a_name
        )
        and (
            q_dtype_w == torch.uint8
            or "fp4" in q_dtype_w_name
            or "float4" in q_dtype_w_name
        )
        and "per_1x32" in q_type_name
        and bool(use_g1u1)
        and activation_profile is not None
        and not bool(doweight_stage1)
        and int(hidden_pad) == 0
        and int(intermediate_pad) == 0
        and bool(is_shuffled)
        and "separated" in gate_mode_name
    )
    if not supported:
        raise ValueError("unsupported MoE configuration")
    if token < 0 or expert <= 0 or topk <= 0 or topk > expert:
        raise ValueError("invalid MoE problem shape")
    one_stage = Moe1StageConfig(
        int(token),
        int(model_dim),
        int(inter_dim),
        int(expert),
        int(topk),
    )
    profile_solution_id = _FUSED_MOE_TWO_STAGE_PROFILES.get(
        (int(model_dim), int(inter_dim), int(topk), activation_profile)
    )
    if profile_solution_id is None:
        return one_stage
    is_gpt_oss_profile = (
        profile_solution_id == _FUSED_MOE_TWO_STAGE_MXFP4_BIAS_SOLUTION_ID
    )
    if token * topk // expert < 64:
        profile_solution_id = _with_fused_moe_weight_load_policy(
            profile_solution_id, _FusedMoeWeightLoadPolicy.non_temporal
        )

    if not is_gpt_oss_profile:
        return Moe2StageConfig(
            int(token),
            int(model_dim),
            int(inter_dim),
            int(expert),
            int(topk),
            _solution_id=profile_solution_id,
        )

    if (
        two_stage_threshold <= 0
        or token <= 0
        or token > two_stage_threshold
        or model_dim != 3072
        or inter_dim != 3072
        or not is_gfx950
    ):
        return one_stage
    return Moe2StageConfig(
        int(token),
        int(model_dim),
        int(inter_dim),
        int(expert),
        int(topk),
        _solution_id=profile_solution_id,
    )


def get_2stage_cfgs(
    token: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    dtype: object,
    q_dtype_a: object,
    q_dtype_w: object,
    q_type: object,
    use_g1u1: bool,
    activation: object,
    doweight_stage1: bool,
    hidden_pad: int,
    intermediate_pad: int,
    is_shuffled: bool = True,
    gate_mode: object = "separated",
    is_ep: bool = False,
    has_stage2_bias: bool = False,
) -> Moe1StageConfig | Moe2StageConfig:
    raw_threshold = os.environ.get("PETIT_KERNEL_GPT_OSS_2STAGE_M_THRESHOLD")
    try:
        threshold = 128 if raw_threshold is None else int(raw_threshold)
    except ValueError:
        threshold = 128
    return _get_2stage_cfgs_cached(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        activation,
        doweight_stage1,
        hidden_pad,
        intermediate_pad,
        is_shuffled,
        gate_mode,
        is_ep,
        has_stage2_bias,
        threshold,
        _gcn_arch_name().startswith("gfx950"),
    )


get_2stage_cfgs.cache_clear = _get_2stage_cfgs_cached.cache_clear


def _gcn_arch_name(device: torch.device | int | str | None = None) -> str:
    if device is None:
        if not torch.cuda.is_available():
            return ""
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return getattr(props, "gcnArchName", "")


def _native_fp8_e4m3_dtype(device: torch.device | int | str | None = None) -> torch.dtype:
    arch = _gcn_arch_name(device)
    if arch.startswith(("gfx950", "gfx1200", "gfx1201")) and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _is_one_byte_float8_or_uint8(tensor: torch.Tensor) -> bool:
    return tensor.dtype == torch.uint8 or tensor.element_size() == 1


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


def _check_device(name: str, tensor: torch.Tensor, device: torch.device) -> None:
    if tensor.device != device:
        raise RuntimeError(f"{name} device mismatch")


def _check_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


def _check_dtype(name: str, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    if tensor.dtype != dtype:
        raise RuntimeError(f"{name} must be {str(dtype).removeprefix('torch.')}")


def _check_common_fmoe_matmul_1stage_args(
    out: torch.Tensor,
    input_q: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    input_scale: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    num_persistent_tgs: int,
) -> tuple[int, int, int]:
    if input_q.device.type != "cuda":
        raise RuntimeError("input_q must be on GPU")

    device = input_q.device
    for name, tensor in (
        ("out", out),
        ("w1_q", w1_q),
        ("w2_q", w2_q),
        ("sorted_token_ids", sorted_token_ids),
        ("sorted_weights", sorted_weights),
        ("sorted_expert_ids", sorted_expert_ids),
        ("num_valid_ids", num_valid_ids),
        ("input_scale", input_scale),
        ("w1_scale", w1_scale),
        ("w2_scale", w2_scale),
    ):
        _check_device(name, tensor, device)

    if input_q.dim() != 2:
        raise RuntimeError("input_q must be [tokens, dim]")
    if w1_q.dim() != 3:
        raise RuntimeError("w1_q must be rank 3")
    if w2_q.dim() != 3:
        raise RuntimeError("w2_q must be rank 3")
    tokens, dim = input_q.shape
    experts = w2_q.size(0)

    for name, tensor in (
        ("out", out),
        ("input_q", input_q),
        ("w1_q", w1_q),
        ("w2_q", w2_q),
        ("sorted_token_ids", sorted_token_ids),
        ("sorted_weights", sorted_weights),
        ("sorted_expert_ids", sorted_expert_ids),
        ("num_valid_ids", num_valid_ids),
        ("input_scale", input_scale),
        ("w1_scale", w1_scale),
        ("w2_scale", w2_scale),
    ):
        _check_contiguous(name, tensor)

    _check_dtype("out", out, torch.bfloat16)
    _check_dtype("sorted_token_ids", sorted_token_ids, torch.int32)
    _check_dtype("sorted_weights", sorted_weights, torch.float32)
    _check_dtype("sorted_expert_ids", sorted_expert_ids, torch.int32)
    _check_dtype("num_valid_ids", num_valid_ids, torch.int32)
    _check_dtype("input_scale", input_scale, torch.float32)

    if out.dim() != 2 or out.size(0) != tokens or out.size(1) != dim:
        raise RuntimeError("out must be [tokens, dim]")
    if num_valid_ids.numel() != 2:
        raise RuntimeError("num_valid_ids must have 2 elements")
    if topk <= 0:
        raise RuntimeError("topk must be > 0")
    if num_persistent_tgs < 0:
        raise RuntimeError("num_persistent_tgs must be >= 0")
    if dim % 128 != 0:
        raise RuntimeError("dim must be divisible by 128")
    return tokens, dim, experts


def _check_fp8_blockscale_fmoe_args(
    input_q: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    input_scale: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    tokens: int,
    dim: int,
    experts: int,
) -> None:
    if not _is_one_byte_float8_or_uint8(input_q):
        raise RuntimeError("input_q must be uint8, float8_e4m3fn, or float8_e4m3fnuz")
    if not _is_one_byte_float8_or_uint8(w1_q):
        raise RuntimeError("w1_q must be uint8, float8_e4m3fn, or float8_e4m3fnuz")
    if not _is_one_byte_float8_or_uint8(w2_q):
        raise RuntimeError("w2_q must be uint8, float8_e4m3fn, or float8_e4m3fnuz")
    _check_dtype("w1_scale", w1_scale, torch.float32)
    _check_dtype("w2_scale", w2_scale, torch.float32)

    inter_dim = w2_q.size(2)
    if w2_q.size(1) != dim:
        raise RuntimeError("w2_q dim mismatch with input_q")
    if w1_q.size(0) != experts:
        raise RuntimeError("w1_q experts mismatch with w2_q")
    if w1_q.size(1) != 2 * inter_dim:
        raise RuntimeError("w1_q second dim must be 2*inter_dim")
    if w1_q.size(2) != dim:
        raise RuntimeError("w1_q dim mismatch with input_q")
    if inter_dim % 128 != 0:
        raise RuntimeError("inter_dim must be divisible by 128")
    if (
        input_scale.dim() != 2
        or input_scale.size(0) != tokens
        or input_scale.size(1) != dim // 128
    ):
        raise RuntimeError("input_scale must be contiguous with shape [tokens, dim/128]")


def _check_mxfp4_scale_shape(
    name: str,
    scale: torch.Tensor,
    experts: int,
    rows: int,
    scale_cols: int,
) -> None:
    if scale.dim() == 3:
        if scale.size(0) != experts:
            raise RuntimeError(f"{name} experts mismatch")
        if scale.size(1) != rows:
            raise RuntimeError(f"{name} second dim mismatch")
        if scale.size(2) != scale_cols:
            raise RuntimeError(f"{name} third dim mismatch")
        return
    if scale.dim() == 2:
        if scale.size(0) != experts:
            raise RuntimeError(f"{name} experts mismatch")
        if scale.size(1) != rows * scale_cols:
            raise RuntimeError(f"{name} second dim mismatch")
        return
    raise RuntimeError(f"{name} must be rank-2 or rank-3")


def _check_fp8_blockscale_mxfp4_fmoe_args(
    input_q: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    input_scale: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    tokens: int,
    dim: int,
    experts: int,
) -> None:
    if not _is_one_byte_float8_or_uint8(input_q):
        raise RuntimeError("input_q must be uint8, float8_e4m3fn, or float8_e4m3fnuz")
    _check_dtype("w1_q", w1_q, torch.uint8)
    _check_dtype("w2_q", w2_q, torch.uint8)
    _check_dtype("w1_scale", w1_scale, torch.uint8)
    _check_dtype("w2_scale", w2_scale, torch.uint8)

    inter_dim = w2_q.size(2) * 2
    if w2_q.size(1) != dim:
        raise RuntimeError("w2_q dim mismatch with input_q")
    if w1_q.size(0) != experts:
        raise RuntimeError("w1_q experts mismatch with w2_q")
    if w1_q.size(1) != 2 * inter_dim:
        raise RuntimeError("w1_q second dim must be 2*inter_dim")
    if w1_q.size(2) * 2 != dim:
        raise RuntimeError("w1_q third dim must be dim/2")
    if inter_dim % 128 != 0:
        raise RuntimeError("inter_dim must be divisible by 128")
    if (
        input_scale.dim() != 2
        or input_scale.size(0) != dim // 128
        or input_scale.size(1) != tokens
    ):
        raise RuntimeError("input_scale must be [dim/128, tokens]")

    _check_mxfp4_scale_shape("w1_scale", w1_scale, experts, 2 * inter_dim, dim // 32)
    _check_mxfp4_scale_shape("w2_scale", w2_scale, experts, dim, inter_dim // 32)


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
    if out is None:
        out = torch.zeros(
            (input_q.size(0), input_q.size(1)),
            dtype=torch.bfloat16,
            device=input_q.device,
        )
    tokens = input_q.size(0)
    dim = input_q.size(1)
    experts = w2_q.size(0)
    _check_fp8_blockscale_fmoe_args(
        input_q, w13_q, w2_q, input_scale, fc1_scale, fc2_scale, tokens, dim, experts
    )
    tokens, dim, experts = _check_common_fmoe_matmul_1stage_args(
        out,
        input_q,
        w13_q,
        w2_q,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        int(topk),
        input_scale,
        fc1_scale,
        fc2_scale,
        int(num_persistent_tgs),
    )
    out.zero_()
    return ops.fmoe_matmul_1stage(
        out,
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
        _with_fused_moe_shape(
            _FUSED_MOE_FP8_BLOCKSCALE_SOLUTION_ID, dim, w2_q.size(2)
        ),
        num_persistent_tgs,
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
    input_q_arg = input_q
    if input_q.dtype != torch.uint8:
        if input_q.element_size() != 1:
            raise TypeError("input_q must be uint8 or a 1-byte float8 tensor")
        input_q_arg = input_q.view(torch.uint8)

    if out is None:
        out = torch.zeros(
            (input_q_arg.size(0), input_q_arg.size(1)),
            dtype=torch.bfloat16,
            device=input_q_arg.device,
        )
    tokens = input_q_arg.size(0)
    dim = input_q_arg.size(1)
    experts = w2_q.size(0)
    _check_fp8_blockscale_mxfp4_fmoe_args(
        input_q_arg,
        w13_q,
        w2_q,
        input_scale,
        fc1_scale,
        fc2_scale,
        tokens,
        dim,
        experts,
    )
    tokens, dim, experts = _check_common_fmoe_matmul_1stage_args(
        out,
        input_q_arg,
        w13_q,
        w2_q,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        int(topk),
        input_scale,
        fc1_scale,
        fc2_scale,
        int(num_persistent_tgs),
    )
    out.zero_()
    return ops.fmoe_matmul_1stage(
        out,
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
        _with_fused_moe_shape(
            _FUSED_MOE_FP8_BLOCKSCALE_MXFP4_SOLUTION_ID,
            dim,
            w2_q.size(2) * 2,
        ),
        num_persistent_tgs,
    )


def _check_optional_bf16_moe_bias(
    bias: torch.Tensor | None,
    hidden_states: torch.Tensor,
    name: str,
) -> None:
    if bias is None:
        return
    if bias.device != hidden_states.device:
        raise RuntimeError(f"{name} device mismatch")
    if bias.dtype != torch.bfloat16:
        raise RuntimeError(f"{name} must be bfloat16")
    if not bias.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


def _check_bf16_mxfp4_fmoe_args(
    hidden_states: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    tokens: int,
    dim: int,
    experts: int,
) -> None:
    _check_dtype("hidden_states", hidden_states, torch.bfloat16)
    _check_dtype("w1_q", w1_q, torch.uint8)
    _check_dtype("w2_q", w2_q, torch.uint8)
    _check_dtype("w1_scale", w1_scale, torch.uint8)
    _check_dtype("w2_scale", w2_scale, torch.uint8)

    inter_dim = w2_q.size(2) * 2
    if w2_q.size(1) != dim:
        raise RuntimeError("w2_q dim mismatch with hidden_states")
    if w1_q.size(0) != experts:
        raise RuntimeError("w1_q experts mismatch with w2_q")
    if w1_q.size(1) != 2 * inter_dim:
        raise RuntimeError("w1_q second dim must be 2*inter_dim")
    if w1_q.size(2) * 2 != dim:
        raise RuntimeError("w1_q third dim must be dim/2")
    if inter_dim % 128 != 0:
        raise RuntimeError("inter_dim must be divisible by 128")

    _check_mxfp4_scale_shape("w1_scale", w1_scale, experts, 2 * inter_dim, dim // 32)
    _check_mxfp4_scale_shape("w2_scale", w2_scale, experts, dim, inter_dim // 32)


def fused_moe_bf16_mxfp4(
    hidden_states: torch.Tensor,
    w13_q: torch.Tensor,
    w2_q: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    fc1_scale: torch.Tensor,
    fc2_scale: torch.Tensor,
    num_persistent_tgs: int = 0,
    out: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.zeros(
            (hidden_states.size(0), hidden_states.size(1)),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
    tokens = hidden_states.size(0)
    dim = hidden_states.size(1)
    experts = w2_q.size(0)
    input_scale = torch.empty((0,), dtype=torch.float32, device=hidden_states.device)
    _check_bf16_mxfp4_fmoe_args(
        hidden_states, w13_q, w2_q, fc1_scale, fc2_scale, tokens, dim, experts
    )
    _check_optional_bf16_moe_bias(w13_bias, hidden_states, "w13_bias")
    _check_optional_bf16_moe_bias(w2_bias, hidden_states, "w2_bias")
    tokens, dim, experts = _check_common_fmoe_matmul_1stage_args(
        out,
        hidden_states,
        w13_q,
        w2_q,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        int(topk),
        input_scale,
        fc1_scale,
        fc2_scale,
        int(num_persistent_tgs),
    )

    out.zero_()
    return ops.fmoe_matmul_1stage(
        out,
        hidden_states,
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
        _with_fused_moe_shape(
            _FUSED_MOE_BF16_MXFP4_BIAS_SOLUTION_ID,
            dim,
            w2_q.size(2) * 2,
        ),
        num_persistent_tgs,
        w13_bias,
        w2_bias,
    )


def get_fp4_solutions(
    size_m: int, size_n: int, size_k: int, a_type: torch.dtype, c_type: torch.dtype
) -> list[int]:
    return ops.get_fp4_solutions(size_m, size_n, size_k, a_type, c_type)


__all__ = [
    "DataType",
    "Moe1StageConfig",
    "Moe2StageConfig",
    "MoeKernelLayout",
    "PetitSolutionHints",
    "fused_moe_bf16_mxfp4",
    "fused_moe_fp8_blockscale_g1u1",
    "fused_moe_fp8_blockscale_g1u1_mxfp4",
    "get_2stage_cfgs",
    "get_fp4_solutions",
    "mul_mxfp4_a16",
    "mul_nvfp4_a16",
    "process_mxfp4_scales",
    "process_nvfp4_scales",
    "repack_moe_kernel_layout",
    "repack_mxfp4",
    "repack_nvfp4",
]
