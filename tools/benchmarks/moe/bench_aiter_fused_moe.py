#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Callable, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

import petit_kernel
from petit_kernel.moe_mxfp4 import MoeKernelLayout

BLOCK_N = 128
BLOCK_K = 128
SORTED_TOKEN_PADDING = 32
AITER_STAGE1_TUNING_NAME = "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2"
AITER_STAGE1_KERNEL_SYMBOL = (
    "mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_swiglu_v32"
)
AITER_STAGE2_TUNING_NAME = (
    "flydsl_moe2_afp4_wfp4_bf16_t32x256x256_atomic_bnt2_persist"
)
AITER_STAGE2_KERNEL_SYMBOL = (
    "mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x256x256_vscale_fix3_"
    "fp4opt_v1_persist_cu256"
)


KERNEL_VARIANTS = (
    "fp8_blockscale_silu",
    "fp8_petit_mxfp4_silu",
    "fp8_petit_mxfp4_openai_bias",
    "bf16_native_mxfp4_openai_bias",
    "mxfp4_native_mxfp4_silu",
    "mxfp4_native_mxfp4_openai_bias",
    "mxfp4_native_mxfp4_openai_bias_2stage",
    "mxfp4_native_mxfp4_openai_bias_2stage_m64_n512",
    "mxfp4_native_mxfp4_silu_2stage",
    "aiter_fp8_blockscale_silu",
    "aiter_dynamic_fp8_blockscale_silu",
    "aiter_dynamic_mxfp4_silu",
    "aiter_mxfp4_silu",
    "aiter_mxfp4_openai_bias_2stage",
    "aiter_dynamic_mxfp4_openai_bias",
)

MODEL_PRESETS = {
    "gpt-oss-120b": (3072, 3072, 16, 4),
    "deepseekv3": (7168, 2048, 33, 9),
    "deepseekv4": (7168, 3072, 49, 7),
}
MODEL_KERNEL_VARIANTS = {
    "gpt-oss-120b": {
        "petit": "mxfp4_native_mxfp4_openai_bias_2stage",
        "aiter": "aiter_mxfp4_openai_bias_2stage",
    },
    "deepseekv3": {
        "petit": "mxfp4_native_mxfp4_silu_2stage",
        "aiter": "aiter_mxfp4_silu",
    },
    "deepseekv4": {
        "petit": "mxfp4_native_mxfp4_silu_2stage",
        "aiter": "aiter_mxfp4_silu",
    },
}


def make_solution_id(
    act_dtype: int,
    weight_dtype: int,
    bias_dtype: int,
    weight_ordering: int,
    mfma: int,
    stages: int,
    activation: int,
    stage1_buffering: int,
) -> int:
    return (
        (act_dtype & 0xF)
        | ((weight_dtype & 0xF) << 4)
        | ((bias_dtype & 0xF) << 8)
        | ((weight_ordering & 0x3) << 12)
        | ((mfma & 0x3) << 14)
        | ((stages & 0xF) << 16)
        | ((activation & 0x7) << 20)
        | ((stage1_buffering & 0x1) << 23)
    )


@dataclass(frozen=True)
class KernelVariant:
    name: str
    act_format: str
    weight_kind: str
    weight_layout: MoeKernelLayout | None
    solution_id: int
    has_bias: bool = False
    aiter_supported: bool = False
    activation_quantization: str | None = None
    two_stage: bool = False
    aiter_activation: str = "silu"
    sorted_token_padding: int = SORTED_TOKEN_PADDING


VARIANTS = {
    "fp8_blockscale_silu": KernelVariant(
        name="fp8_blockscale_silu",
        act_format="fp8",
        weight_kind="fp8",
        weight_layout=None,
        solution_id=make_solution_id(3, 4, 0, 2, 0, 0, 0, 1),
    ),
    "fp8_petit_mxfp4_silu": KernelVariant(
        name="fp8_petit_mxfp4_silu",
        act_format="fp8",
        weight_kind="mxfp4",
        weight_layout=MoeKernelLayout.petit_mxfp4,
        solution_id=make_solution_id(3, 1, 0, 1, 0, 0, 0, 0),
    ),
    "fp8_petit_mxfp4_openai_bias": KernelVariant(
        name="fp8_petit_mxfp4_openai_bias",
        act_format="fp8",
        weight_kind="mxfp4",
        weight_layout=MoeKernelLayout.petit_mxfp4,
        solution_id=make_solution_id(3, 1, 5, 1, 0, 0, 1, 1),
        has_bias=True,
    ),
    "bf16_native_mxfp4_openai_bias": KernelVariant(
        name="bf16_native_mxfp4_openai_bias",
        act_format="bf16",
        weight_kind="mxfp4",
        weight_layout=MoeKernelLayout.native_mxfp4,
        solution_id=make_solution_id(5, 1, 5, 0, 1, 0, 1, 1),
        has_bias=True,
    ),
    "mxfp4_native_mxfp4_silu": KernelVariant(
        name="mxfp4_native_mxfp4_silu",
        act_format="mxfp4",
        weight_kind="mxfp4",
        weight_layout=MoeKernelLayout.native_mxfp4,
        solution_id=make_solution_id(1, 1, 0, 0, 2, 0, 0, 1),
    ),
    "mxfp4_native_mxfp4_openai_bias": KernelVariant(
        name="mxfp4_native_mxfp4_openai_bias",
        act_format="mxfp4",
        weight_kind="mxfp4",
        weight_layout=MoeKernelLayout.native_mxfp4,
        solution_id=make_solution_id(1, 1, 5, 0, 2, 0, 1, 1),
        has_bias=True,
    ),
    "mxfp4_native_mxfp4_openai_bias_2stage": KernelVariant(
        name="mxfp4_native_mxfp4_openai_bias_2stage",
        act_format="mxfp4",
        weight_kind="mxfp4",
        weight_layout=MoeKernelLayout.native_mxfp4,
        solution_id=petit_kernel._FUSED_MOE_TWO_STAGE_MXFP4_BIAS_SOLUTION_ID,
        has_bias=True,
        two_stage=True,
        aiter_activation="swiglu",
    ),
    "mxfp4_native_mxfp4_openai_bias_2stage_m64_n512": KernelVariant(
        name="mxfp4_native_mxfp4_openai_bias_2stage_m64_n512",
        act_format="mxfp4",
        weight_kind="mxfp4",
        weight_layout=MoeKernelLayout.native_mxfp4,
        solution_id=(
            petit_kernel._FUSED_MOE_TWO_STAGE_MXFP4_BIAS_M64_N512_SOLUTION_ID
        ),
        has_bias=True,
        two_stage=True,
        aiter_activation="swiglu",
        sorted_token_padding=64,
    ),
    "mxfp4_native_mxfp4_silu_2stage": KernelVariant(
        name="mxfp4_native_mxfp4_silu_2stage",
        act_format="mxfp4",
        weight_kind="mxfp4",
        weight_layout=MoeKernelLayout.native_mxfp4,
        solution_id=0,
        two_stage=True,
    ),
    "aiter_fp8_blockscale_silu": KernelVariant(
        name="aiter_fp8_blockscale_silu",
        act_format="fp8",
        weight_kind="aiter_fp8",
        weight_layout=None,
        solution_id=0,
        aiter_supported=True,
        activation_quantization="static_fp8_blockscale",
    ),
    "aiter_dynamic_fp8_blockscale_silu": KernelVariant(
        name="aiter_dynamic_fp8_blockscale_silu",
        act_format="bf16",
        weight_kind="aiter_fp8",
        weight_layout=None,
        solution_id=0,
        aiter_supported=True,
        activation_quantization="dynamic_fp8_blockscale",
    ),
    "aiter_dynamic_mxfp4_silu": KernelVariant(
        name="aiter_dynamic_mxfp4_silu",
        act_format="bf16",
        weight_kind="aiter_mxfp4",
        weight_layout=None,
        solution_id=0,
        aiter_supported=True,
        activation_quantization="dynamic_mxfp4",
    ),
    "aiter_mxfp4_silu": KernelVariant(
        name="aiter_mxfp4_silu",
        act_format="mxfp4",
        weight_kind="aiter_mxfp4",
        weight_layout=None,
        solution_id=0,
        aiter_supported=True,
        activation_quantization="static_mxfp4",
    ),
    "aiter_mxfp4_openai_bias_2stage": KernelVariant(
        name="aiter_mxfp4_openai_bias_2stage",
        act_format="mxfp4",
        weight_kind="mxfp4",
        weight_layout=MoeKernelLayout.native_mxfp4,
        solution_id=0,
        has_bias=True,
        aiter_supported=True,
        two_stage=True,
        aiter_activation="swiglu",
    ),
    "aiter_dynamic_mxfp4_openai_bias": KernelVariant(
        name="aiter_dynamic_mxfp4_openai_bias",
        act_format="bf16",
        weight_kind="aiter_mxfp4",
        weight_layout=None,
        solution_id=0,
        has_bias=True,
        aiter_supported=True,
        activation_quantization="dynamic_mxfp4",
        aiter_activation="swiglu",
    ),
}


def native_fp8_e4m3_dtype(device: torch.device) -> torch.dtype:
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    if arch.startswith(("gfx950", "gfx1200", "gfx1201")) and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Fused MoE kernels.")
    parser.add_argument(
        "--model",
        choices=MODEL_PRESETS,
        help="Apply a predefined local MoE shape; explicit shape arguments override it.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="petit",
        choices=["petit", "aiter"],
        help="Backend implementation to benchmark.",
    )
    parser.add_argument(
        "--kernel-variant",
        type=str,
        choices=KERNEL_VARIANTS,
        help="Kernel variant; defaults to the model/backend preset when available.",
    )
    parser.add_argument("--tokens", type=int, default=256, help="Number of tokens.")
    parser.add_argument("--dim", type=int, help="Model dimension.")
    parser.add_argument("--inter-dim", "--inter_dim", dest="inter_dim", type=int, help="Intermediate dimension.")
    parser.add_argument("--experts", type=int, help="Number of local experts.")
    parser.add_argument("--topk", type=int, help="Top-k local routes per token.")
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Enable persistent scheduling for the petit backend.",
    )
    parser.add_argument(
        "--num-persistent-tgs",
        type=int,
        default=0,
        help="Total persistent workgroups. Overrides --persistent auto-sizing when > 0.",
    )
    parser.add_argument(
        "--persistent-tgs-per-cu",
        type=int,
        default=2,
        help="Auto-sized persistent workgroups per CU when --persistent is set.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--repeat", type=int, default=100, help="Number of benchmark iterations.")
    parser.add_argument(
        "--graph-iters",
        type=int,
        default=16,
        help="Iterations captured in each CUDA graph replay.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    defaults = MODEL_PRESETS.get(args.model, (4096, 1024, 8, 2))
    for field, value in zip(("dim", "inter_dim", "experts", "topk"), defaults):
        if getattr(args, field) is None:
            setattr(args, field, value)
    if args.kernel_variant is None:
        args.kernel_variant = MODEL_KERNEL_VARIANTS.get(args.model, {}).get(
            args.backend, "fp8_blockscale_silu"
        )
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.tokens <= 0 or args.dim <= 0 or args.inter_dim <= 0:
        raise ValueError("tokens, dim and inter_dim must be positive.")
    if args.experts <= 0 or args.topk <= 0:
        raise ValueError("experts and topk must be positive.")
    if args.topk > args.experts:
        raise ValueError("topk must be <= experts.")
    if args.topk > 255:
        raise ValueError("topk must be <= 255 (slot id is packed in 8 bits).")
    if args.tokens >= (1 << 24):
        raise ValueError("tokens must be < 2^24 (token id is packed in 24 bits).")
    if args.dim % 256 != 0 or args.inter_dim % 256 != 0:
        raise ValueError("dim and inter_dim must both be divisible by 256.")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive.")
    if args.graph_iters <= 0:
        raise ValueError("graph-iters must be positive.")
    if args.num_persistent_tgs < 0 or args.persistent_tgs_per_cu <= 0:
        raise ValueError("num-persistent-tgs must be >= 0 and persistent-tgs-per-cu must be > 0.")
    variant = VARIANTS[args.kernel_variant]
    if variant.two_stage:
        two_stage_shapes = {
            "mxfp4_native_mxfp4_openai_bias_2stage": {(3072, 3072, 4)},
            "mxfp4_native_mxfp4_openai_bias_2stage_m64_n512": {
                (3072, 3072, 4),
            },
            "mxfp4_native_mxfp4_silu_2stage": {
                (7168, 2048, 9),
                (7168, 3072, 7),
            },
            "aiter_mxfp4_openai_bias_2stage": {(3072, 3072, 4)},
        }
        if (args.dim, args.inter_dim, args.topk) not in two_stage_shapes[
            variant.name
        ]:
            raise ValueError(
                f"unsupported two-stage MXFP4 shape for {variant.name}: "
                f"dim={args.dim}, inter-dim={args.inter_dim}, topk={args.topk}."
            )
    if args.backend == "aiter" and not variant.aiter_supported:
        supported = ", ".join(v.name for v in VARIANTS.values() if v.aiter_supported)
        raise ValueError(
            f"backend=aiter is only supported for kernel-variant in {{{supported}}}."
        )
    if args.backend == "petit" and variant.aiter_supported:
        supported = ", ".join(v.name for v in VARIANTS.values() if not v.aiter_supported)
        raise ValueError(
            f"backend=petit is only supported for kernel-variant in {{{supported}}}."
        )


def resolve_num_persistent_tgs(args: argparse.Namespace) -> int:
    num_persistent_tgs = args.num_persistent_tgs
    if num_persistent_tgs == 0 and args.persistent:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        num_persistent_tgs = props.multi_processor_count * args.persistent_tgs_per_cu
    return num_persistent_tgs


class FusedMoEInputBuilder:
    WEIGHT_STD = 40.0
    WEIGHT_MEAN = 0.0
    SCALE_INV_STD = 2e-5
    SCALE_INV_MEAN = 1e-4
    INPUT_STD = 1.0
    INPUT_MEAN = 0.0
    INPUT_SPIKE_PROB = 0.002
    INPUT_SPIKE_STD = 16.0
    MXFP4_SCALE_MIN = 1
    MXFP4_SCALE_MAX = 237

    def __init__(
        self,
        *,
        device: torch.device,
        tokens: int,
        dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        variant: KernelVariant,
    ):
        self.device = device
        self.tokens = tokens
        self.dim = dim
        self.inter_dim = inter_dim
        self.experts = experts
        self.topk = topk
        self.variant = variant
        self.sorted_token_padding = variant.sorted_token_padding
        self.solution_id = variant.solution_id
        if variant.name == "mxfp4_native_mxfp4_silu_2stage":
            self.solution_id = {
                (7168, 2048): petit_kernel._FUSED_MOE_TWO_STAGE_MXFP4_SILU_7168X2048_SOLUTION_ID,
                (7168, 3072): petit_kernel._FUSED_MOE_TWO_STAGE_MXFP4_SILU_7168X3072_SOLUTION_ID,
            }[(dim, inter_dim)]
        if self.solution_id and not variant.two_stage:
            self.solution_id = petit_kernel._with_fused_moe_shape(
                self.solution_id, dim, inter_dim
            )

    def _rand_positive_normal(
        self, shape: Tuple[int, ...], mean: float, std: float
    ) -> torch.Tensor:
        x = torch.randn(shape, dtype=torch.float32, device=self.device) * std + mean
        return x.clamp_min(1e-8)

    def _sample_spiky_input(self, shape: Tuple[int, ...]) -> torch.Tensor:
        base = torch.randn(shape, dtype=torch.float32, device=self.device) * self.INPUT_STD + self.INPUT_MEAN
        spike_mask = torch.rand(shape, dtype=torch.float32, device=self.device) < self.INPUT_SPIKE_PROB
        spikes = torch.randn(shape, dtype=torch.float32, device=self.device) * self.INPUT_SPIKE_STD
        return base + spikes * spike_mask.to(torch.float32)

    @staticmethod
    def _pack_blocks(x: torch.Tensor, block_n: int, block_k: int) -> torch.Tensor:
        experts, rows, cols = x.shape
        pack_k = 16
        kk = block_k // pack_k
        return (
            x.view(experts, rows // block_n, block_n, cols // block_k, kk, pack_k)
            .permute(0, 1, 3, 4, 2, 5)
            .contiguous()
        )

    def _build_padded_sorted_routing(
        self, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        padding = self.sorted_token_padding
        max_num_tokens_padded = int(topk_ids.numel() + self.experts * padding - self.topk)
        max_num_m_blocks = (max_num_tokens_padded + padding - 1) // padding
        init_val = (self.topk << 24) | self.tokens

        sorted_token_ids = torch.full((max_num_tokens_padded,), init_val, dtype=torch.int32)
        sorted_weights = torch.zeros((max_num_tokens_padded,), dtype=torch.float32)
        sorted_expert_ids = torch.full((max_num_m_blocks,), -1, dtype=torch.int32)

        topk_ids_cpu = topk_ids.to("cpu", dtype=torch.int32)
        topk_weights_cpu = topk_weights.to("cpu", dtype=torch.float32)
        token_ids_cpu = (
            torch.arange(self.tokens, dtype=torch.int32).unsqueeze(1).expand(self.tokens, self.topk).reshape(-1)
        )
        slot_ids_cpu = (
            torch.arange(self.topk, dtype=torch.int32).unsqueeze(0).expand(self.tokens, self.topk).reshape(-1)
        )
        expert_ids_cpu = topk_ids_cpu.reshape(-1)
        route_weights_cpu = topk_weights_cpu.reshape(-1)

        order = torch.argsort(expert_ids_cpu, stable=True)
        token_ids_cpu = token_ids_cpu[order]
        slot_ids_cpu = slot_ids_cpu[order]
        expert_ids_cpu = expert_ids_cpu[order]
        route_weights_cpu = route_weights_cpu[order]

        expert_counts = torch.bincount(expert_ids_cpu, minlength=self.experts)
        route_begin = 0
        sorted_ids_begin = 0
        sorted_expert_ids_begin = 0
        for expert in range(self.experts):
            tokens_num = int(expert_counts[expert].item())
            if tokens_num == 0:
                continue
            route_end = route_begin + tokens_num
            sorted_token_ids[sorted_ids_begin:sorted_ids_begin + tokens_num] = (
                (slot_ids_cpu[route_begin:route_end] << 24) | token_ids_cpu[route_begin:route_end]
            )
            sorted_weights[sorted_ids_begin:sorted_ids_begin + tokens_num] = route_weights_cpu[
                route_begin:route_end
            ]

            sorted_expert_ids_num = (tokens_num + padding - 1) // padding
            tokens_num_pad = sorted_expert_ids_num * padding
            sorted_expert_ids[
                sorted_expert_ids_begin:sorted_expert_ids_begin + sorted_expert_ids_num
            ] = expert

            route_begin = route_end
            sorted_ids_begin += tokens_num_pad
            sorted_expert_ids_begin += sorted_expert_ids_num

        # Match the kernel ABI used by the C++ test harness:
        # [num_valid_route_ids_padded, token_count].
        num_valid_ids = torch.tensor([sorted_ids_begin, self.tokens], dtype=torch.int32)
        return (
            sorted_token_ids.to(self.device),
            sorted_weights.to(self.device),
            sorted_expert_ids.to(self.device),
            num_valid_ids.to(self.device),
            sorted_expert_ids_begin,
        )

    @staticmethod
    def _mask_negative_zero_native_fp4(words: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(words)
        for i in range(8):
            nibble = (words >> (i * 4)) & 0xF
            nibble = torch.where(nibble == 0x8, torch.zeros_like(nibble), nibble)
            out |= nibble << (i * 4)
        return out

    def _build_mxfp4_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w13_words = torch.randint(
            0,
            1 << 32,
            (self.experts, self.inter_dim * 2, self.dim // 8),
            dtype=torch.int64,
            device=self.device,
        ).to(torch.int32)
        w2_words = torch.randint(
            0,
            1 << 32,
            (self.experts, self.dim, self.inter_dim // 8),
            dtype=torch.int64,
            device=self.device,
        ).to(torch.int32)
        w13_q = self._mask_negative_zero_native_fp4(w13_words).view(torch.uint8).reshape(
            self.experts, self.inter_dim * 2, self.dim // 2
        )
        w2_q = self._mask_negative_zero_native_fp4(w2_words).view(torch.uint8).reshape(
            self.experts, self.dim, self.inter_dim // 2
        )
        fc1_scale = torch.randint(
            self.MXFP4_SCALE_MIN,
            self.MXFP4_SCALE_MAX + 1,
            (self.experts, self.inter_dim * 2, self.dim // 32),
            dtype=torch.uint8,
            device=self.device,
        )
        fc2_scale = torch.randint(
            self.MXFP4_SCALE_MIN,
            self.MXFP4_SCALE_MAX + 1,
            (self.experts, self.dim, self.inter_dim // 32),
            dtype=torch.uint8,
            device=self.device,
        )
        return w13_q.contiguous(), w2_q.contiguous(), fc1_scale.contiguous(), fc2_scale.contiguous()

    def _build_aiter_mxfp4_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            from aiter import dtypes  # type: ignore
            from aiter.ops.shuffle import shuffle_weight  # type: ignore
            from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp  # type: ignore
        except Exception as exc:
            raise RuntimeError("AITER MXFP4 weight generation requires aiter.") from exc

        w13 = (
            torch.randn(
                (self.experts, self.inter_dim * 2, self.dim),
                dtype=torch.float32,
                device=self.device,
            )
            * self.WEIGHT_STD
            + self.WEIGHT_MEAN
        ).to(torch.bfloat16)
        w2 = (
            torch.randn(
                (self.experts, self.dim, self.inter_dim),
                dtype=torch.float32,
                device=self.device,
            )
            * self.WEIGHT_STD
            + self.WEIGHT_MEAN
        ).to(torch.bfloat16)
        w13_q, fc1_scale = downcast_to_mxfp(w13, torch.uint8, axis=-1)
        w2_q, fc2_scale = downcast_to_mxfp(w2, torch.uint8, axis=-1)
        w13_q = shuffle_weight(w13_q.view(dtypes.fp4x2), (16, 16), use_int4=True)
        w2_q = shuffle_weight(w2_q.view(dtypes.fp4x2), (16, 16), use_int4=True)
        return (
            w13_q.contiguous(),
            w2_q.contiguous(),
            fc1_scale.contiguous().view(dtypes.fp8_e8m0),
            fc2_scale.contiguous().view(dtypes.fp8_e8m0),
        )

    def _build_aiter_openai_mxfp4_weights(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            from aiter import dtypes  # type: ignore
            from aiter.ops.shuffle import (  # type: ignore
                shuffle_scale_a16w4,
                shuffle_weight,
                shuffle_weight_a16w4,
            )
            from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp  # type: ignore
            from aiter.utility.fp4_utils import e8m0_shuffle  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "AITER OpenAI MXFP4 weight generation requires aiter."
            ) from exc

        w13 = (
            torch.randn(
                (self.experts, self.inter_dim * 2, self.dim),
                dtype=torch.float32,
                device=self.device,
            )
            * self.WEIGHT_STD
            + self.WEIGHT_MEAN
        ).to(torch.bfloat16)
        w2 = (
            torch.randn(
                (self.experts, self.dim, self.inter_dim),
                dtype=torch.float32,
                device=self.device,
            )
            * self.WEIGHT_STD
            + self.WEIGHT_MEAN
        ).to(torch.bfloat16)
        w13_q, fc1_scale = downcast_to_mxfp(w13, torch.uint8, axis=-1)
        w2_q, fc2_scale = downcast_to_mxfp(w2, torch.uint8, axis=-1)

        w13_q = shuffle_weight(w13_q.view(dtypes.fp4x2), (16, 16))
        w2_q = shuffle_weight_a16w4(
            w2_q.view(dtypes.fp4x2), 16, gate_up=False
        )
        fc1_scale = e8m0_shuffle(
            fc1_scale.view(self.experts * self.inter_dim * 2, self.dim // 32)
        )
        fc2_scale = shuffle_scale_a16w4(
            fc2_scale.view(self.experts * self.dim, self.inter_dim // 32),
            self.experts,
            gate_up=False,
        )
        return (
            w13_q.contiguous(),
            w2_q.contiguous(),
            fc1_scale.contiguous().view(dtypes.fp8_e8m0),
            fc2_scale.contiguous().view(dtypes.fp8_e8m0),
        )

    def _build_mxfp4_activations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        act_words = torch.randint(
            0,
            1 << 32,
            (self.tokens, self.dim // 8),
            dtype=torch.int64,
            device=self.device,
        ).to(torch.int32)
        act_q = self._mask_negative_zero_native_fp4(act_words).view(torch.uint8).reshape(
            self.tokens, self.dim // 2
        )
        act_scale = torch.randint(
            self.MXFP4_SCALE_MIN,
            self.MXFP4_SCALE_MAX + 1,
            (self.tokens, self.dim // 32),
            dtype=torch.uint8,
            device=self.device,
        )
        return act_q.contiguous(), act_scale.contiguous()

    def _pack_aiter_sorted_mxfp4_activation_scales(
        self,
        act_scale: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
    ) -> torch.Tensor:
        try:
            from aiter.utility.fp4_utils import moe_mxfp4_sort  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "MXFP4 activation scale packing requires AITER's moe_mxfp4_sort."
            ) from exc

        return moe_mxfp4_sort(
            act_scale.contiguous(),
            sorted_ids=sorted_token_ids,
            num_valid_ids=num_valid_ids,
            token_num=self.tokens,
            block_size=self.sorted_token_padding,
        ).contiguous()

    def _build_biases(self) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.variant.has_bias:
            return None, None
        w13_bias = torch.randn(
            (self.experts, 2, self.inter_dim),
            dtype=torch.float32,
            device=self.device,
        )
        w2_bias = torch.randn(
            (self.experts, self.dim), dtype=torch.float32, device=self.device
        )
        if self.variant.aiter_supported:
            return w13_bias.contiguous(), w2_bias.contiguous()
        if self.variant.weight_layout is None:
            raise RuntimeError("biased Petit kernels require an MXFP4 layout")
        return (
            petit_kernel.repack_moe_kernel_layout(
                w13_bias.to(torch.bfloat16),
                layout=self.variant.weight_layout,
            ),
            petit_kernel.repack_moe_kernel_layout(
                w2_bias.to(torch.bfloat16),
                layout=self.variant.weight_layout,
            ),
        )

    def build(self) -> Dict[str, object]:
        experts = self.experts
        dim = self.dim
        inter_dim = self.inter_dim
        model_blocks = dim // BLOCK_K
        w2_row_blocks = dim // BLOCK_N
        inter_blocks = inter_dim // BLOCK_K
        fp8_dtype = (
            native_fp8_e4m3_dtype(self.device)
            if self.variant.act_format == "fp8" or self.variant.weight_kind in ("fp8", "aiter_fp8")
            else None
        )

        if self.variant.act_format == "bf16":
            input_q = self._sample_spiky_input((self.tokens, dim)).to(torch.bfloat16)
            input_scale = torch.empty((0,), dtype=torch.float32, device=self.device)
        elif self.variant.act_format == "mxfp4":
            input_q, input_scale = self._build_mxfp4_activations()
            if self.variant.aiter_supported:
                try:
                    from aiter import dtypes  # type: ignore
                except Exception as exc:
                    raise RuntimeError(
                        "AITER MXFP4 activation generation requires aiter."
                    ) from exc
                input_q = input_q.view(dtypes.fp4x2)
                input_scale = input_scale.view(dtypes.fp8_e8m0)
        else:
            assert fp8_dtype is not None
            input_q = self._sample_spiky_input((self.tokens, dim)).to(fp8_dtype)
            input_scale = self._rand_positive_normal(
                (self.tokens, model_blocks), self.SCALE_INV_MEAN, self.SCALE_INV_STD
            )
        if self.variant.weight_kind in ("fp8", "aiter_fp8"):
            assert fp8_dtype is not None
            w1_q = (
                torch.randn((experts, inter_dim * 2, dim), dtype=torch.float32, device=self.device)
                * self.WEIGHT_STD
                + self.WEIGHT_MEAN
            ).to(fp8_dtype)
            w2_q = (
                torch.randn((experts, dim, inter_dim), dtype=torch.float32, device=self.device)
                * self.WEIGHT_STD
                + self.WEIGHT_MEAN
            ).to(fp8_dtype)
            fc1_scale = self._rand_positive_normal(
                (experts, ((inter_dim * 2) // BLOCK_N) * model_blocks), self.SCALE_INV_MEAN, self.SCALE_INV_STD
            )
            fc2_scale = self._rand_positive_normal(
                (experts, w2_row_blocks * inter_blocks), self.SCALE_INV_MEAN, self.SCALE_INV_STD
            )
        elif self.variant.weight_kind == "aiter_mxfp4":
            if self.variant.aiter_activation == "swiglu":
                w1_q, w2_q, fc1_scale, fc2_scale = (
                    self._build_aiter_openai_mxfp4_weights()
                )
            else:
                w1_q, w2_q, fc1_scale, fc2_scale = (
                    self._build_aiter_mxfp4_weights()
                )
        else:
            w1_q, w2_q, fc1_scale, fc2_scale = self._build_mxfp4_weights()

        routing_generator = torch.Generator(device=self.device)
        routing_generator.manual_seed(torch.initial_seed())
        topk_ids = (
            torch.rand(
                (self.tokens, experts),
                dtype=torch.float32,
                device=self.device,
                generator=routing_generator,
            )
            .topk(self.topk, dim=-1)
            .indices.to(torch.int32)
        )
        topk_weights = torch.rand(
            (self.tokens, self.topk),
            dtype=torch.float32,
            device=self.device,
            generator=routing_generator,
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        (
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            num_valid_m_blocks,
        ) = self._build_padded_sorted_routing(
            topk_ids=topk_ids, topk_weights=topk_weights
        )

        if self.variant.weight_kind in ("fp8", "aiter_fp8"):
            w1_q_kernel = self._pack_blocks(w1_q, block_n=16, block_k=32).reshape(experts, inter_dim * 2, dim)
            w2_q_kernel = self._pack_blocks(w2_q, block_n=16, block_k=32).reshape(experts, dim, inter_dim)
            if self.variant.weight_kind == "aiter_fp8":
                w1_q_kernel.is_shuffled = True
                w2_q_kernel.is_shuffled = True
            input_scale_kernel = input_scale.contiguous()
            fc1_scale_kernel = fc1_scale.contiguous()
            fc2_scale_kernel = (
                fc2_scale.contiguous()
                .view(experts, w2_row_blocks, inter_blocks)
                .transpose(1, 2)
                .reshape(experts, inter_blocks * w2_row_blocks)
                .contiguous()
            )
        elif self.variant.weight_kind == "aiter_mxfp4":
            w1_q_kernel = w1_q
            w2_q_kernel = w2_q
            input_scale_kernel = input_scale.contiguous()
            fc1_scale_kernel = fc1_scale.contiguous()
            fc2_scale_kernel = fc2_scale.contiguous()
        else:
            assert self.variant.weight_layout is not None
            (
                w1_q_kernel,
                fc1_scale_kernel,
            ) = petit_kernel.repack_moe_kernel_layout(
                w1_q,
                fc1_scale,
                layout=self.variant.weight_layout,
            )
            (
                w2_q_kernel,
                fc2_scale_kernel,
            ) = petit_kernel.repack_moe_kernel_layout(
                w2_q,
                fc2_scale,
                layout=self.variant.weight_layout,
            )
            input_scale_kernel = (
                input_scale.t().contiguous()
                if self.variant.act_format == "fp8"
                else input_scale.contiguous()
            )
        if self.variant.act_format == "mxfp4":
            input_scale_kernel = (
                input_scale.contiguous()
                if self.variant.aiter_supported
                else self._pack_aiter_sorted_mxfp4_activation_scales(
                    input_scale,
                    sorted_token_ids,
                    num_valid_ids,
                )
            )
            input_q = input_q.contiguous()
        w13_bias, w2_bias = self._build_biases()

        return {
            "input_q": input_q,
            "w1_q_kernel": w1_q_kernel,
            "w2_q_kernel": w2_q_kernel,
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "sorted_token_ids": sorted_token_ids,
            "sorted_weights": sorted_weights,
            "sorted_expert_ids": sorted_expert_ids,
            "num_valid_ids": num_valid_ids,
            "num_valid_m_blocks": num_valid_m_blocks,
            "sorted_token_padding": self.sorted_token_padding,
            "topk": self.topk,
            "input_scale": input_scale.contiguous(),
            "input_scale_kernel": input_scale_kernel,
            "fc1_scale_kernel": fc1_scale_kernel,
            "fc2_scale_kernel": fc2_scale_kernel,
            "w13_bias": w13_bias,
            "w2_bias": w2_bias,
            "kernel_variant": self.variant.name,
            "solution_id": self.solution_id,
            "activation_quantization": self.variant.activation_quantization,
            "two_stage": self.variant.two_stage,
            "aiter_activation": self.variant.aiter_activation,
            "tokens": self.tokens,
            "dim": self.dim,
            "inter_dim": self.inter_dim,
        }


def make_explicit_gptoss_run_fn(
    data: Dict[str, object],
) -> Tuple[Callable[[], torch.Tensor], str]:
    try:
        from aiter import ActivationType  # type: ignore
        from aiter._version import __version__ as aiter_version  # type: ignore
        from aiter.fused_moe import (  # type: ignore
            _flydsl_stage1_wrapper,
            _flydsl_stage2_wrapper,
        )
        from aiter.ops.quant import (  # type: ignore
            fused_dynamic_mxfp4_quant_moe_sort,
        )
    except Exception as exc:
        raise RuntimeError(
            "The explicit GPT-OSS benchmark requires AITER v0.1.19.post2 "
            "with FlyDSL."
        ) from exc
    if aiter_version != "0.1.19.post2":
        raise RuntimeError(
            "The explicit GPT-OSS kernel pair requires AITER v0.1.19.post2; "
            f"found {aiter_version}."
        )

    hidden_states = data["input_q"]
    w1 = data["w1_q_kernel"]
    w2 = data["w2_q_kernel"]
    sorted_ids = data["sorted_token_ids"]
    sorted_weights = data["sorted_weights"]
    sorted_expert_ids = data["sorted_expert_ids"]
    num_valid_ids = data["num_valid_ids"]
    topk_ids = data["topk_ids"]
    for tensor in (
        hidden_states,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk_ids,
    ):
        assert isinstance(tensor, torch.Tensor)

    tokens = int(data["tokens"])
    topk = int(data["topk"])
    dim = int(data["dim"])
    inter_dim = int(data["inter_dim"])
    stage1_out = torch.empty(
        (tokens, topk, inter_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    out = torch.empty(
        (tokens, dim), dtype=torch.bfloat16, device=hidden_states.device
    )

    def run() -> torch.Tensor:
        a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
            hidden_states,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=tokens,
            topk=topk,
            block_size=int(data["sorted_token_padding"]),
            sorted_weights=sorted_weights,
        )
        _flydsl_stage1_wrapper(
            a1,
            w1,
            w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            stage1_out,
            topk,
            kernelName=AITER_STAGE1_TUNING_NAME,
            activation=ActivationType.Swiglu,
            w1_scale=data["fc1_scale_kernel"],
            a1_scale=a1_scale,
            bias1=data["w13_bias"],
            topk_ids=topk_ids,
            swiglu_limit=7.0,
        )
        a2, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            stage1_out.view(-1, inter_dim),
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=tokens,
            topk=topk,
            block_size=int(data["sorted_token_padding"]),
            sorted_weights=sorted_weights,
        )
        a2 = a2.view(tokens, topk, -1)
        out.zero_()
        _flydsl_stage2_wrapper(
            a2,
            w1,
            w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out,
            topk,
            kernelName=AITER_STAGE2_TUNING_NAME,
            w2_scale=data["fc2_scale_kernel"],
            a2_scale=a2_scale,
            sorted_weights=sorted_weights,
            bias2=data["w2_bias"],
        )
        return out

    return run, f"{AITER_STAGE1_KERNEL_SYMBOL}+{AITER_STAGE2_KERNEL_SYMBOL}"


def make_run_fn(
    backend: str, data: Dict[str, object], num_persistent_tgs: int
) -> Tuple[Callable[[], torch.Tensor], str]:
    input_q = data["input_q"]
    assert isinstance(input_q, torch.Tensor)
    if backend == "petit":
        out = torch.empty(
            (int(data["tokens"]), int(data["dim"])),
            dtype=torch.bfloat16,
            device=input_q.device,
        )

        if bool(data["two_stage"]):
            config = petit_kernel.Moe2StageConfig(
                int(data["tokens"]),
                int(data["dim"]),
                int(data["inter_dim"]),
                int(data["w2_q_kernel"].size(0)),
                int(data["topk"]),
                block_m=int(data["sorted_token_padding"]),
                _solution_id=int(data["solution_id"]),
            )
            intermediate = torch.empty(
                config.workspace_size(data["sorted_expert_ids"].numel()),
                dtype=torch.uint8,
                device=input_q.device,
            )

            def run_two_stage() -> torch.Tensor:
                config.stage1(
                    intermediate,
                    data["input_q"],
                    data["w1_q_kernel"],
                    data["sorted_token_ids"],
                    data["sorted_expert_ids"],
                    data["num_valid_ids"],
                    data["input_scale_kernel"],
                    data["fc1_scale_kernel"],
                    num_persistent_tgs=num_persistent_tgs,
                    bias1=data["w13_bias"],
                )
                out.zero_()
                return config.stage2(
                    out,
                    intermediate,
                    data["w2_q_kernel"],
                    data["sorted_token_ids"],
                    data["sorted_weights"],
                    data["sorted_expert_ids"],
                    data["num_valid_ids"],
                    data["fc2_scale_kernel"],
                    num_persistent_tgs=num_persistent_tgs,
                    bias2=data["w2_bias"],
                )

            return run_two_stage, "matmul_2stage"

        def run() -> torch.Tensor:
            out.zero_()
            return petit_kernel.ops.fmoe_matmul_1stage(
                out,
                data["input_q"],
                data["w1_q_kernel"],
                data["w2_q_kernel"],
                data["sorted_token_ids"],
                data["sorted_weights"],
                data["sorted_expert_ids"],
                data["num_valid_ids"],
                int(data["topk"]),
                data["input_scale_kernel"],
                data["fc1_scale_kernel"],
                data["fc2_scale_kernel"],
                int(data["solution_id"]),
                num_persistent_tgs,
                data["w13_bias"],
                data["w2_bias"],
            )

        return run, "matmul_1stage"

    if num_persistent_tgs > 0:
        raise RuntimeError(
            "Persistent launch forcing is only supported for backend=petit; "
            "backend=aiter uses its own kernel heuristic."
        )

    if bool(data["two_stage"]):
        try:
            from aiter import dtypes  # type: ignore
            from aiter.ops.flydsl import moe_kernels  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "The AITER two-stage benchmark requires FlyDSL."
            ) from exc

        out = torch.empty(
            (int(data["tokens"]), int(data["dim"])),
            dtype=torch.bfloat16,
            device=input_q.device,
        )
        input_fp4 = input_q.view(dtypes.fp4x2)
        w1_fp4 = data["w1_q_kernel"].view(dtypes.fp4x2)
        w2_fp4 = data["w2_q_kernel"].view(dtypes.fp4x2)
        input_scale = data["input_scale_kernel"].view(dtypes.fp8_e8m0)
        w1_scale = data["fc1_scale_kernel"].view(dtypes.fp8_e8m0)
        w2_scale = data["fc2_scale_kernel"].view(dtypes.fp8_e8m0)

        def run_aiter_two_stage() -> torch.Tensor:
            intermediate, intermediate_scale = moe_kernels.flydsl_moe_stage1(
                a=input_fp4,
                w1=w1_fp4,
                sorted_token_ids=data["sorted_token_ids"],
                sorted_expert_ids=data["sorted_expert_ids"],
                num_valid_ids=data["num_valid_ids"],
                topk=int(data["topk"]),
                tile_m=32,
                tile_n=128,
                tile_k=256,
                a_dtype="fp4",
                b_dtype="fp4",
                out_dtype="fp4",
                act="swiglu",
                w1_scale=w1_scale,
                a1_scale=input_scale,
                use_async_copy=True,
                waves_per_eu=2,
                b_nt=2,
                gate_mode="separated",
                bias=data["w13_bias"],
                swiglu_limit=7.0,
            )
            out.zero_()
            return moe_kernels.flydsl_moe_stage2(
                inter_states=intermediate,
                w2=w2_fp4,
                sorted_token_ids=data["sorted_token_ids"],
                sorted_expert_ids=data["sorted_expert_ids"],
                num_valid_ids=data["num_valid_ids"],
                out=out,
                topk=int(data["topk"]),
                tile_m=32,
                tile_n=256,
                tile_k=256,
                a_dtype="fp4",
                b_dtype="fp4",
                out_dtype="bf16",
                mode="atomic",
                w2_scale=w2_scale,
                a2_scale=intermediate_scale,
                sorted_weights=data["sorted_weights"],
                b_nt=2,
                bias=data["w2_bias"],
                persist=True,
            )

        return run_aiter_two_stage, "flydsl_moe_stage1+flydsl_moe_stage2"

    if data["aiter_activation"] == "swiglu":
        return make_explicit_gptoss_run_fn(data)

    try:
        from aiter import ActivationType, QuantType  # type: ignore
        from aiter.fused_moe import fused_moe  # type: ignore
    except Exception as exc:
        raise RuntimeError("Failed to import aiter for backend=aiter.") from exc

    activation_quantization = data.get("activation_quantization")
    if activation_quantization == "static_fp8_blockscale":
        quant_type = QuantType.per_1x128
        impl = "fused_moe_fp8_blockscale"
        a1_scale = data["input_scale_kernel"]
        topk_ids = data["topk_ids"]
        assert isinstance(topk_ids, torch.Tensor)
        num_local_tokens = torch.tensor(
            [int(data["tokens"])], dtype=topk_ids.dtype, device=topk_ids.device
        )
    elif activation_quantization == "dynamic_fp8_blockscale":
        quant_type = QuantType.per_1x128
        impl = "fused_moe_dynamic_fp8"
        a1_scale = None
        num_local_tokens = None
    elif activation_quantization == "dynamic_mxfp4":
        quant_type = QuantType.per_1x32
        impl = "fused_moe_dynamic_mxfp4"
        a1_scale = None
        num_local_tokens = None
    elif activation_quantization == "static_mxfp4":
        quant_type = QuantType.per_1x32
        impl = "fused_moe_mxfp4"
        a1_scale = data["input_scale_kernel"]
        num_local_tokens = None
    else:
        raise RuntimeError(
            "backend=aiter requires an AITER kernel variant."
        )

    def run_aiter() -> torch.Tensor:
        return fused_moe(
            data["input_q"],
            data["w1_q_kernel"],
            data["w2_q_kernel"],
            data["topk_weights"],
            data["topk_ids"],
            activation=ActivationType.Silu,
            quant_type=quant_type,
            w1_scale=data["fc1_scale_kernel"],
            w2_scale=data["fc2_scale_kernel"],
            a1_scale=a1_scale,
            num_local_tokens=num_local_tokens,
            dtype=torch.bfloat16,
        )

    return run_aiter, impl


def benchmark_with_events(fn: Callable[[], torch.Tensor], repeat: int) -> Tuple[float, int]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat, repeat


def benchmark_with_cuda_graph(
    fn: Callable[[], torch.Tensor], warmup: int, repeat: int, graph_iters: int
) -> Tuple[float, int, bool]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    try:
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                fn()
            with torch.cuda.graph(graph):
                for _ in range(graph_iters):
                    fn()
        torch.cuda.current_stream().wait_stream(capture_stream)

        num_replays = math.ceil(repeat / graph_iters)
        total_iters = num_replays * graph_iters
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(num_replays):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / total_iters, total_iters, True
    except RuntimeError:
        ms, total_iters = benchmark_with_events(fn, repeat)
        return ms, total_iters, False


def compute_metrics(args: argparse.Namespace, data: Dict[str, object], avg_ms: float) -> Tuple[float, float]:
    routes = args.tokens * args.topk
    flops = 6.0 * routes * args.dim * args.inter_dim
    tflops = flops / (avg_ms * 1e-3) / 1e12

    byte_tensors: list[torch.Tensor] = [
        data["input_q"],
        data["w1_q_kernel"],
        data["w2_q_kernel"],
        data["sorted_token_ids"],
        data["sorted_weights"],
        data["sorted_expert_ids"],
        data["num_valid_ids"],
        data["input_scale_kernel"],
        data["fc1_scale_kernel"],
        data["fc2_scale_kernel"],
    ]  # type: ignore[list-item]
    for key in ("w13_bias", "w2_bias"):
        tensor = data[key]
        if isinstance(tensor, torch.Tensor):
            byte_tensors.append(tensor)
    total_bytes = sum(t.numel() * t.element_size() for t in byte_tensors)
    total_bytes += args.tokens * args.dim * torch.tensor([], dtype=torch.bfloat16).element_size()
    gbps = total_bytes / (avg_ms * 1e-3) / 1e9
    return tflops, gbps


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("CUDA/HIP device is not available.", file=sys.stderr)
        return 1
    variant = VARIANTS[args.kernel_variant]
    if variant.act_format == "fp8" and not (
        hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")
    ):
        print("torch float8_e4m3 dtype is required for this benchmark.", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    num_persistent_tgs = resolve_num_persistent_tgs(args)

    builder = FusedMoEInputBuilder(
        device=device,
        tokens=args.tokens,
        dim=args.dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
        variant=variant,
    )
    data = builder.build()

    try:
        run_fn, impl = make_run_fn(args.backend, data, num_persistent_tgs)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    avg_ms, total_iters, used_graph = benchmark_with_cuda_graph(
        run_fn, args.warmup, args.repeat, args.graph_iters
    )
    tflops, gbps = compute_metrics(args, data, avg_ms)

    num_valid_ids = int(data["num_valid_ids"][0].item())
    num_valid_m_blocks = int(data["num_valid_m_blocks"])
    launch = (
        "persistent"
        if args.backend == "petit" and num_persistent_tgs > 0
        else "standard"
        if args.backend == "petit"
        else "heuristic"
    )
    activation_quantization = data["activation_quantization"]
    activation_quantization_field = (
        f"activation_quantization={activation_quantization} "
        if activation_quantization is not None
        else ""
    )
    model_field = f"model={args.model} " if args.model is not None else ""
    print(
        f"backend={args.backend} impl={impl} graph={1 if used_graph else 0} "
        f"{model_field}"
        f"kernel_variant={args.kernel_variant} solution_id={int(data['solution_id'])} "
        f"{activation_quantization_field}"
        f"tokens={args.tokens} dim={args.dim} inter_dim={args.inter_dim} "
        f"experts={args.experts} topk={args.topk} "
        f"valid_ids={num_valid_ids} valid_m_blocks={num_valid_m_blocks} "
        f"persistent_tgs={num_persistent_tgs} launch={launch} "
        f"total_iters={total_iters} device_ms={avg_ms:.4f} "
        f"tflops={tflops:.3f} gbps={gbps:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
