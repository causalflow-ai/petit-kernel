"""Compare Petit, AITER, and FlyDSL two-stage MegaMoE full paths.

This benchmark is intentionally closer to the vLLM serving path than the local
fused-MoE microbenchmarks.  Each iteration starts from per-rank hidden states,
computes top-k, then runs the Petit or FlyDSL MegaMoE path, or vLLM's default
all-gather/reduce-scatter EP flow around AITER's public ``fused_moe`` entry
point. The timed region includes top-k, live routing metadata, activation
quantization, dispatch, both expert stages, return, and combine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist
from torch.cuda import nvtx

import petit_kernel

MXFP4_SCALE_MIN = 118
MXFP4_SCALE_MAX = 122


@dataclass(frozen=True)
class Topology:
    dp_size: int
    tp_size: int
    ep_size: int
    ranks_per_dp: int
    world_size: int
    rank: int
    local_rank: int
    dp_rank: int
    rank_in_dp: int
    local_experts: int
    local_expert_start: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MegaMoE full path.")
    parser.add_argument(
        "--backend",
        choices=("petit", "aiter", "flydsl"),
        default="petit",
    )
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--m", type=int, nargs="+", default=[256])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[256])
    parser.add_argument("--global-experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=2880)
    parser.add_argument("--padded-hidden-size", type=int, default=3072)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--model-name", default="GPT-OSS-120B")
    parser.add_argument(
        "--activation-function",
        choices=("swiglu", "silu"),
        default="swiglu",
    )
    parser.add_argument(
        "--bias",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--graph-iters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--print-summary", action="store_true", default=True)
    parser.add_argument(
        "--profile-ranges",
        action="store_true",
        help="Emit NVTX/ROCTx ranges around benchmark stages. Off by default because it can perturb ROCm timings.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("dp_size", "tp_size", "ep_size", "global_experts", "topk"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.global_experts < args.topk:
        raise ValueError("--global-experts must be >= --topk")
    if args.hidden_size <= 0 or args.padded_hidden_size <= 0:
        raise ValueError("hidden sizes must be positive")
    if args.padded_hidden_size < args.hidden_size:
        raise ValueError("--padded-hidden-size must be >= --hidden-size")
    if args.padded_hidden_size % 256 != 0:
        raise ValueError("--padded-hidden-size must be divisible by 256")
    if args.intermediate_size <= 0 or args.intermediate_size % 256 != 0:
        raise ValueError("--intermediate-size must be positive and divisible by 256")
    if any(m <= 0 for m in args.m):
        raise ValueError("--m values must be positive")
    if any(b <= 0 for b in args.batch_size):
        raise ValueError("--batch-size values must be positive")
    if len(args.batch_size) not in (1, len(args.m)):
        raise ValueError(
            "--batch-size must have length 1 or match the number of --m values"
        )
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")
    if args.graph_iters <= 0:
        raise ValueError("--graph-iters must be positive")
    if int(args.tp_size) != 1:
        raise ValueError("vLLM-equivalent MegaMoE profiling requires --tp-size=1")
    actual_config = (
        args.ep_size,
        args.global_experts,
        args.topk,
        args.hidden_size,
        args.padded_hidden_size,
        args.intermediate_size,
        args.activation_function,
        args.bias,
    )
    registered_configs = {
        (1, 32, 4, 2880, 3072, 3072, "swiglu", True),
        (2, 32, 4, 2880, 3072, 3072, "swiglu", True),
        (4, 32, 4, 2880, 3072, 3072, "swiglu", True),
        (8, 32, 4, 2880, 3072, 3072, "swiglu", True),
        (8, 128, 4, 2880, 3072, 3072, "swiglu", True),
        (8, 256, 8, 7168, 7168, 2048, "silu", False),
        (8, 384, 6, 7168, 7168, 3072, "silu", False),
    }
    if actual_config not in registered_configs:
        raise ValueError(
            "unsupported registered two-stage MegaMoE configuration: "
            f"{actual_config}"
        )
    if any(m > 1024 for m in args.m):
        raise ValueError(
            "the registered MegaMoE workspace supports at most 1024 tokens per rank"
        )


def init_topology(args: argparse.Namespace) -> Topology:
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

    if args.dp_size != args.ep_size:
        raise RuntimeError(
            "vLLM-equivalent MegaMoE profiling requires logical dp_size == ep_size; "
            f"got dp_size={args.dp_size}, ep_size={args.ep_size}"
        )
    expected_world = args.ep_size
    if world_size != expected_world:
        raise RuntimeError(
            f"torchrun world_size={world_size} does not match "
            f"the logical EP size = {expected_world}"
        )
    if args.global_experts % args.ep_size != 0:
        raise RuntimeError(
            f"global_experts={args.global_experts} must be divisible by ep_size={args.ep_size}"
        )

    ranks_per_dp = world_size
    dp_rank = rank
    rank_in_dp = rank
    local_experts = args.global_experts // args.ep_size
    return Topology(
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        ranks_per_dp=ranks_per_dp,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        dp_rank=dp_rank,
        rank_in_dp=rank_in_dp,
        local_experts=local_experts,
        local_expert_start=rank * local_experts,
    )


def make_tp_group(topo: Topology):
    if topo.tp_size != 1:
        raise RuntimeError("this benchmark only supports tp_size=1")
    return dist.group.WORLD


def workload_pairs(args: argparse.Namespace) -> Iterable[tuple[int, int]]:
    if len(args.batch_size) == 1:
        for m in args.m:
            yield int(args.batch_size[0]), int(m)
        return
    for batch, m in zip(args.batch_size, args.m, strict=True):
        yield int(batch), int(m)


def mask_negative_zero_native_fp4(words: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(words)
    for i in range(8):
        nibble = (words >> (i * 4)) & 0xF
        nibble = torch.where(nibble == 0x8, torch.zeros_like(nibble), nibble)
        out |= nibble << (i * 4)
    return out


def build_native_mxfp4_weights(
    *,
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w1_words = torch.randint(
        0,
        1 << 32,
        (local_experts, intermediate_size * 2, hidden_size // 8),
        dtype=torch.int64,
        device=device,
    ).to(torch.int32)
    w2_words = torch.randint(
        0,
        1 << 32,
        (local_experts, hidden_size, intermediate_size // 8),
        dtype=torch.int64,
        device=device,
    ).to(torch.int32)
    w1_q = (
        mask_negative_zero_native_fp4(w1_words)
        .view(torch.uint8)
        .reshape(local_experts, intermediate_size * 2, hidden_size // 2)
    )
    w2_q = (
        mask_negative_zero_native_fp4(w2_words)
        .view(torch.uint8)
        .reshape(local_experts, hidden_size, intermediate_size // 2)
    )
    fc1_scale = torch.randint(
        MXFP4_SCALE_MIN,
        MXFP4_SCALE_MAX + 1,
        (local_experts, intermediate_size * 2, hidden_size // 32),
        dtype=torch.uint8,
        device=device,
    )
    fc2_scale = torch.randint(
        MXFP4_SCALE_MIN,
        MXFP4_SCALE_MAX + 1,
        (local_experts, hidden_size, intermediate_size // 32),
        dtype=torch.uint8,
        device=device,
    )
    return (
        w1_q.contiguous(),
        w2_q.contiguous(),
        fc1_scale.contiguous(),
        fc2_scale.contiguous(),
    )


def build_petit_weights(
    *,
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    w1_q, w2_q, fc1_scale, fc2_scale = build_native_mxfp4_weights(
        local_experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    w1, fc1 = petit_kernel.repack_moe_kernel_layout(
        w1_q, fc1_scale, layout=petit_kernel.MoeKernelLayout.native_mxfp4
    )
    w2, fc2 = petit_kernel.repack_moe_kernel_layout(
        w2_q, fc2_scale, layout=petit_kernel.MoeKernelLayout.native_mxfp4
    )
    b1 = torch.zeros(
        (local_experts, 2, intermediate_size), dtype=torch.bfloat16, device=device
    )
    b2 = torch.zeros((local_experts, hidden_size), dtype=torch.bfloat16, device=device)
    b1 = petit_kernel.repack_moe_kernel_layout(
        b1,
        layout=petit_kernel.MoeKernelLayout.native_mxfp4,
    )
    b2 = petit_kernel.repack_moe_kernel_layout(
        b2,
        layout=petit_kernel.MoeKernelLayout.native_mxfp4,
    )
    return (
        w1.contiguous(),
        w2.contiguous(),
        fc1.contiguous(),
        fc2.contiguous(),
        b1.contiguous(),
        b2.contiguous(),
    )


def build_aiter_weights(
    *,
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from aiter import dtypes  # type: ignore
    from aiter.ops.shuffle import (  # type: ignore
        shuffle_scale_a16w4,
        shuffle_weight_a16w4,
    )

    w1_q, w2_q, fc1_scale, fc2_scale = build_native_mxfp4_weights(
        local_experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    w1_q = shuffle_weight_a16w4(w1_q.view(dtypes.fp4x2), 16, gate_up=True)
    w2_q = shuffle_weight_a16w4(w2_q.view(dtypes.fp4x2), 16, gate_up=False)
    fc1_scale = shuffle_scale_a16w4(
        fc1_scale.view(local_experts * intermediate_size * 2, hidden_size // 32),
        local_experts,
        gate_up=True,
    )
    fc2_scale = shuffle_scale_a16w4(
        fc2_scale.view(local_experts * hidden_size, intermediate_size // 32),
        local_experts,
        gate_up=False,
    )
    b1 = torch.zeros(
        (local_experts, intermediate_size * 2), dtype=torch.float32, device=device
    )
    b2 = torch.zeros((local_experts, hidden_size), dtype=torch.float32, device=device)
    return (
        w1_q.contiguous(),
        w2_q.contiguous(),
        fc1_scale.contiguous().view(dtypes.fp8_e8m0),
        fc2_scale.contiguous().view(dtypes.fp8_e8m0),
        b1,
        b2,
    )


def make_inputs(
    *,
    m: int,
    hidden_size: int,
    padded_hidden_size: int,
    global_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.randn(
        (m, padded_hidden_size), dtype=torch.float32, device=device
    ).to(torch.bfloat16)
    hidden[:, hidden_size:] = 0
    reduced_hidden = torch.empty_like(hidden)
    router_logits = torch.empty((m, global_experts), dtype=torch.float32, device=device)
    return hidden.contiguous(), reduced_hidden, router_logits


def make_topk_buffers(
    *, m: int, topk: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = torch.empty((m, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((m, topk), dtype=torch.float32, device=device)
    return topk_ids, topk_weights


def import_aiter_topk():
    from aiter.fused_moe import fused_topk  # type: ignore

    return fused_topk


def range_call(enabled: bool, name: str, fn: Callable[[], object]) -> object:
    if not enabled:
        return fn()
    nvtx.range_push(name)
    try:
        return fn()
    finally:
        nvtx.range_pop()


def stage_tp_reduce(
    hidden: torch.Tensor,
    reduced_hidden: torch.Tensor,
    tp_group,
    tp_size: int,
) -> None:
    if tp_size <= 1:
        return
    reduced_hidden.copy_(hidden)
    dist.all_reduce(reduced_hidden, op=dist.ReduceOp.SUM, group=tp_group)


def stage_topk(
    reduced_hidden: torch.Tensor,
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    topk: int,
    fused_topk: Callable,
) -> None:
    router_logits.copy_(reduced_hidden[:, : router_logits.size(1)].float())
    fused_topk(
        reduced_hidden,
        router_logits,
        topk,
        True,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )


class AgRsEPDispatcher:
    def __init__(
        self,
        *,
        m: int,
        topk: int,
        world_size: int,
        hidden_size: int,
        device: torch.device,
    ) -> None:
        self.gathered_hidden = torch.empty(
            (m * world_size, hidden_size), dtype=torch.bfloat16, device=device
        )
        self.gathered_weights = torch.empty(
            (m * world_size, topk), dtype=torch.float32, device=device
        )
        self.gathered_expert_ids = torch.empty(
            (m * world_size, topk), dtype=torch.int32, device=device
        )
        self.out = torch.empty((m, hidden_size), dtype=torch.bfloat16, device=device)

    def dispatch(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        dist.all_gather_into_tensor(self.gathered_hidden, hidden.contiguous())
        dist.all_gather_into_tensor(self.gathered_weights, topk_weights.contiguous())
        dist.all_gather_into_tensor(self.gathered_expert_ids, topk_ids.contiguous())

    def combine(self, local_out: torch.Tensor) -> torch.Tensor:
        dist.reduce_scatter_tensor(
            self.out,
            local_out.contiguous(),
            op=dist.ReduceOp.SUM,
        )
        return self.out


def make_petit_backend(
    *,
    topo: Topology,
    args: argparse.Namespace,
    m: int,
    device: torch.device,
) -> tuple[Callable[[], torch.Tensor], Callable[[], None]]:
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp  # type: ignore

    w1, w2, fc1_scale, fc2_scale, w1_bias, w2_bias = build_petit_weights(
        local_experts=topo.local_experts,
        hidden_size=args.padded_hidden_size,
        intermediate_size=args.intermediate_size,
        device=device,
    )
    config = petit_kernel.MegaMoeConfig(
        world_size=topo.ep_size,
        num_experts=args.global_experts,
        topk=args.topk,
        model_dim=args.hidden_size,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=petit_kernel.MegaMoeActivationFunction(
            args.activation_function
        ),
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=args.intermediate_size,
        has_bias=args.bias,
    )
    heap = petit_kernel.create_vmm_symmetric_heap(topo.world_size)
    views = config.input_views(heap, m)
    if views.scales is None:
        raise RuntimeError("MXFP4 MegaMoE workspace is missing activation scales")
    out = torch.empty((m, args.padded_hidden_size), dtype=torch.bfloat16, device=device)

    def prepare(
        reduced_hidden: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> None:
        input_q, input_scale = downcast_to_mxfp(
            reduced_hidden[:, : args.hidden_size], torch.uint8, axis=-1
        )
        views.tokens.copy_(input_q)
        views.scales.copy_(input_scale)
        views.expert_ids.copy_(topk_ids)
        views.expert_weights.copy_(topk_weights)

    def compute() -> torch.Tensor:
        return config.run(
            heap,
            w1,
            w2,
            fc1_scale,
            fc2_scale,
            m,
            w13_bias=w1_bias if args.bias else None,
            w2_bias=w2_bias if args.bias else None,
            out=out,
        )

    return compute, prepare


def make_aiter_backend(
    *,
    topo: Topology,
    args: argparse.Namespace,
    m: int,
    device: torch.device,
) -> tuple[
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
    Callable[[], torch.Tensor],
]:
    from aiter import ActivationType, QuantType  # type: ignore
    from aiter.fused_moe import fused_moe  # type: ignore

    dispatcher = AgRsEPDispatcher(
        m=m,
        topk=args.topk,
        world_size=topo.world_size,
        hidden_size=args.padded_hidden_size,
        device=device,
    )
    expert_mask = torch.zeros((args.global_experts,), dtype=torch.int32, device=device)
    expert_mask[
        topo.local_expert_start : topo.local_expert_start + topo.local_experts
    ] = 1
    w1, w2, fc1_scale, fc2_scale, w1_bias, w2_bias = build_aiter_weights(
        local_experts=topo.local_experts,
        hidden_size=args.padded_hidden_size,
        intermediate_size=args.intermediate_size,
        device=device,
    )

    def prepare(
        reduced_hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        dispatcher.dispatch(reduced_hidden, topk_ids, topk_weights)

    def compute() -> torch.Tensor:
        local_out = fused_moe(
            dispatcher.gathered_hidden,
            w1,
            w2,
            dispatcher.gathered_weights,
            dispatcher.gathered_expert_ids,
            expert_mask=expert_mask,
            activation=(
                ActivationType.Swiglu
                if args.activation_function == "swiglu"
                else ActivationType.Silu
            ),
            quant_type=QuantType.per_1x32,
            w1_scale=fc1_scale,
            w2_scale=fc2_scale,
            dtype=torch.bfloat16,
            hidden_pad=args.padded_hidden_size - args.hidden_size,
            intermediate_pad=0,
            bias1=w1_bias if args.bias else None,
            bias2=w2_bias if args.bias else None,
        )
        return dispatcher.combine(local_out)

    return prepare, compute


def make_flydsl_backend(
    *,
    topo: Topology,
    args: argparse.Namespace,
    m: int,
    device: torch.device,
) -> tuple[
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
    Callable[[], torch.Tensor],
]:
    from tools.benchmarks.moe.flydsl_megamoe_adapter import create_flydsl_megamoe

    w1, w2, w1_scale, w2_scale, _, _ = build_aiter_weights(
        local_experts=topo.local_experts,
        hidden_size=args.padded_hidden_size,
        intermediate_size=args.intermediate_size,
        device=device,
    )
    moe = create_flydsl_megamoe(
        rank=topo.rank,
        world_size=topo.world_size,
        model_dim=args.padded_hidden_size,
        inter_dim=args.intermediate_size,
        experts=args.global_experts,
        topk=args.topk,
        max_tokens_per_rank=1 << (m - 1).bit_length(),
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )
    state: dict[str, torch.Tensor] = {}

    def prepare(
        reduced_hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        state["input"], state["scale"] = moe.quantize(reduced_hidden)
        state["topk_ids"] = topk_ids
        state["topk_weights"] = topk_weights

    def compute() -> torch.Tensor:
        return moe.forward_prequant(
            state["input"],
            state["scale"],
            state["topk_weights"],
            state["topk_ids"],
        )

    return prepare, compute


def event_ms(fn: Callable[[], object], repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def distributed_stats(
    value: float, device: torch.device
) -> tuple[float, float, float, float]:
    local = torch.tensor([value], dtype=torch.float32, device=device)
    max_value = local.clone()
    dist.all_reduce(max_value, op=dist.ReduceOp.MAX)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    values = torch.stack(gathered).flatten().sort().values
    count = values.numel()

    def percentile(p: float) -> float:
        idx = min(count - 1, max(0, int(math.ceil(p * count) - 1)))
        return float(values[idx].item())

    return float(max_value.item()), percentile(0.5), percentile(0.9), percentile(0.99)


def benchmark_with_graph(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
    graph_iters: int,
) -> tuple[float, int, bool]:
    # The eager warmup is required for shape-specific AITER/Triton compilation.
    # Synchronize before capture so none of that one-time work can leak into the
    # graph or its timing.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
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

        # A graph's first replay can instantiate/upload kernels and register
        # collective resources. Warm up the captured path as well; otherwise
        # AITER's first-replay overhead is included in the measured average.
        warmup_replays = math.ceil(warmup / graph_iters)
        for _ in range(warmup_replays):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()

        num_replays = math.ceil(repeat / graph_iters)
        total_iters = num_replays * graph_iters
        ms = event_ms(lambda: graph.replay(), num_replays) / graph_iters
        return ms, total_iters, True
    except RuntimeError as exc:
        torch.cuda.synchronize()
        raise RuntimeError(
            "CUDA graph capture/replay failed; this benchmark requires CUDA graph timing"
        ) from exc


def route_histogram(topk_ids: torch.Tensor, global_experts: int) -> list[int]:
    counts = torch.bincount(
        topk_ids.flatten().to(torch.long), minlength=global_experts
    ).to(torch.int64)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return [int(v) for v in counts.detach().cpu().tolist()]


def output_is_valid(tensor: torch.Tensor, device: torch.device) -> bool:
    ok = torch.tensor(
        [int(torch.isfinite(tensor.float()).all().item())],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())


def write_results(row: dict[str, object], args: argparse.Namespace) -> None:
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.csv.exists()
        with args.csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def run_one(
    args: argparse.Namespace, topo: Topology, tp_group, batch_size: int, m: int
) -> dict[str, object]:
    device = torch.device("cuda", topo.local_rank)
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    torch.manual_seed(args.seed + topo.rank * 17 + m)
    torch.cuda.manual_seed_all(args.seed + topo.rank * 17 + m)

    hidden, reduced_hidden, router_logits = make_inputs(
        m=m,
        hidden_size=args.hidden_size,
        padded_hidden_size=args.padded_hidden_size,
        global_experts=args.global_experts,
        device=device,
    )
    if topo.tp_size == 1:
        reduced_hidden = hidden
    topk_ids, topk_weights = make_topk_buffers(m=m, topk=args.topk, device=device)
    fused_topk = import_aiter_topk()

    stage_tp_reduce(hidden, reduced_hidden, tp_group, topo.tp_size)
    stage_topk(
        reduced_hidden,
        router_logits,
        topk_ids,
        topk_weights,
        args.topk,
        fused_topk,
    )
    torch.cuda.synchronize()
    dist.barrier()

    if args.backend == "petit":
        petit_compute, petit_prepare = make_petit_backend(
            topo=topo,
            args=args,
            m=m,
            device=device,
        )

        def backend_prepare() -> None:
            petit_prepare(reduced_hidden, topk_ids, topk_weights)

        def backend_compute() -> torch.Tensor:
            return petit_compute()

    elif args.backend == "aiter":
        aiter_prepare, aiter_compute = make_aiter_backend(
            topo=topo,
            args=args,
            m=m,
            device=device,
        )

        def backend_prepare() -> None:
            aiter_prepare(reduced_hidden, topk_ids, topk_weights)

        def backend_compute() -> torch.Tensor:
            return aiter_compute()

    else:
        flydsl_prepare, flydsl_compute = make_flydsl_backend(
            topo=topo,
            args=args,
            m=m,
            device=device,
        )

        def backend_prepare() -> None:
            flydsl_prepare(reduced_hidden, topk_ids, topk_weights)

        def backend_compute() -> torch.Tensor:
            return flydsl_compute()

    def reduce_fn() -> None:
        range_call(
            args.profile_ranges,
            f"{args.backend}:tp_reduce_for_topk",
            lambda: stage_tp_reduce(hidden, reduced_hidden, tp_group, topo.tp_size),
        )

    def topk_fn() -> None:
        range_call(
            args.profile_ranges,
            f"{args.backend}:topk",
            lambda: stage_topk(
                reduced_hidden,
                router_logits,
                topk_ids,
                topk_weights,
                args.topk,
                fused_topk,
            ),
        )

    def prepare_fn() -> None:
        range_call(
            args.profile_ranges, f"{args.backend}:prep_dispatch", backend_prepare
        )

    def compute_fn() -> torch.Tensor:
        return range_call(
            args.profile_ranges,
            f"{args.backend}:moe_compute_combine",
            backend_compute,
        )

    def total_fn() -> torch.Tensor:
        return range_call(
            args.profile_ranges,
            f"{args.backend}:total",
            lambda: (
                reduce_fn(),
                topk_fn(),
                prepare_fn(),
                compute_fn(),
            )[-1],
        )

    total_local, total_iters, graph = benchmark_with_graph(
        total_fn,
        warmup=args.warmup,
        repeat=args.repeat,
        graph_iters=args.graph_iters,
    )

    output = total_fn()
    # MegaMoE only defines the logical GPT-OSS hidden columns; the 192 padding
    # columns are transport/compute padding and are intentionally unspecified.
    valid_output = output_is_valid(output[:, : args.hidden_size], device)
    hist = route_histogram(topk_ids, args.global_experts)

    total_max, total_p50, total_p90, total_p99 = distributed_stats(total_local, device)

    routes = m * args.topk * topo.world_size
    tps = (
        (m * topo.world_size) / (total_max * 1.0e-3) if total_max > 0 else float("inf")
    )
    route_tps = routes / (total_max * 1.0e-3) if total_max > 0 else float("inf")
    row: dict[str, object] = {
        "model": args.model_name,
        "backend": args.backend,
        "batch_size": batch_size,
        "m": m,
        "dp_size": topo.dp_size,
        "tp_size": topo.tp_size,
        "ep_size": topo.ep_size,
        "world_size": topo.world_size,
        "comparison_mode": "two_stage_full_path_bf16_boundary",
        "physical_world_size": topo.world_size,
        "logical_dp_size": topo.dp_size,
        "logical_ep_size": topo.ep_size,
        "dispatch_semantics": (
            "fused_megamoe_ep"
            if args.backend == "petit"
            else (
                "vllm_allgather_reducescatter_ep"
                if args.backend == "aiter"
                else "flydsl_megamoe_mori_shmem"
            )
        ),
        "global_experts": args.global_experts,
        "local_experts": topo.local_experts,
        "topk": args.topk,
        "hidden_size": args.hidden_size,
        "padded_hidden_size": args.padded_hidden_size,
        "intermediate_size": args.intermediate_size,
        "sm_count": sm_count,
        "stages": 2,
        "activation": args.activation_function,
        "has_bias": int(args.bias),
        "expert_compute": (
            "petit_a4w4_two_stage"
            if args.backend == "petit"
            else (
                "vllm_aiter_a16w4_fused_moe"
                if args.backend == "aiter"
                else "aiter_flydsl_megamoe_v2_a8w4"
            )
        ),
        "timed_boundary": "bf16_hidden_to_bf16_combined_output",
        "routing_metadata_timed": 1,
        "vllm_ep_backend": (
            "allgather_reducescatter" if args.backend == "aiter" else ""
        ),
        "graph": int(graph),
        "total_iters": total_iters,
        "valid_output": int(valid_output),
        "tokens_per_s": tps,
        "routes_per_s": route_tps,
        "total_ms": total_max,
        "total_p50_ms": total_p50,
        "total_p90_ms": total_p90,
        "total_p99_ms": total_p99,
        "route_histogram": json.dumps(hist),
    }
    return row


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("CUDA/HIP device is not available.", file=sys.stderr)
        return 2

    topo = init_topology(args)
    tp_group = make_tp_group(topo)

    try:
        if args.backend == "flydsl":
            from tools.benchmarks.moe.flydsl_megamoe_adapter import (
                initialize_flydsl_runtime,
            )

            initialize_flydsl_runtime()
        for batch_size, m in workload_pairs(args):
            row = run_one(args, topo, tp_group, batch_size, m)
            if topo.rank == 0:
                write_results(row, args)
                if args.print_summary:
                    print(
                        "backend={backend} dp={dp_size} tp={tp_size} ep={ep_size} "
                        "world={world_size} batch={batch_size} m={m} graph={graph} "
                        "total_iters={total_iters} total_ms={total_ms:.4f} "
                        "tokens_per_s={tokens_per_s:.2f} "
                        "valid_output={valid_output}".format(**row),
                        flush=True,
                    )
            dist.barrier()
    finally:
        if args.backend == "flydsl":
            from tools.benchmarks.moe.flydsl_megamoe_adapter import (
                finalize_flydsl_runtime,
            )

            finalize_flydsl_runtime()
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
