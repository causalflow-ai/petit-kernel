#!/usr/bin/env python3

import argparse
import math
import sys
from typing import Callable, Dict, Tuple

import torch

import petit_kernel

BLOCK_N = 128
BLOCK_K = 128
SORTED_TOKEN_PADDING = 32


def native_fp8_e4m3_dtype(device: torch.device) -> torch.dtype:
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    if arch.startswith(("gfx950", "gfx1200", "gfx1201")) and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Fused MoE FP8 blockscale kernel (petit or aiter)."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="petit",
        choices=["petit", "aiter"],
        help="Backend implementation to benchmark.",
    )
    parser.add_argument(
        "--weight-format",
        type=str,
        default="fp8",
        choices=["fp8", "mxfp4"],
        help="Weight format to benchmark. MXFP4 is supported for backend=petit.",
    )
    parser.add_argument("--tokens", type=int, default=256, help="Number of tokens.")
    parser.add_argument("--dim", type=int, default=4096, help="Model dimension.")
    parser.add_argument("--inter-dim", "--inter_dim", dest="inter_dim", type=int, default=1024, help="Intermediate dimension.")
    parser.add_argument("--experts", type=int, default=8, help="Number of experts.")
    parser.add_argument("--topk", type=int, default=2, help="Top-k routes per token.")
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
    return parser.parse_args()


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
    if args.weight_format == "mxfp4" and args.backend != "petit":
        raise ValueError("weight-format=mxfp4 is only supported for backend=petit.")


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
        weight_format: str,
    ):
        self.device = device
        self.tokens = tokens
        self.dim = dim
        self.inter_dim = inter_dim
        self.experts = experts
        self.topk = topk
        self.weight_format = weight_format

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
        max_num_tokens_padded = int(topk_ids.numel() + self.experts * SORTED_TOKEN_PADDING - self.topk)
        max_num_m_blocks = (max_num_tokens_padded + SORTED_TOKEN_PADDING - 1) // SORTED_TOKEN_PADDING
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

            sorted_expert_ids_num = (tokens_num + SORTED_TOKEN_PADDING - 1) // SORTED_TOKEN_PADDING
            tokens_num_pad = sorted_expert_ids_num * SORTED_TOKEN_PADDING
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

    def build(self) -> Dict[str, torch.Tensor]:
        experts = self.experts
        dim = self.dim
        inter_dim = self.inter_dim
        model_blocks = dim // BLOCK_K
        w2_row_blocks = dim // BLOCK_N
        inter_blocks = inter_dim // BLOCK_K
        fp8_dtype = native_fp8_e4m3_dtype(self.device)

        input_q = self._sample_spiky_input((self.tokens, dim)).to(fp8_dtype)
        input_scale = self._rand_positive_normal((self.tokens, model_blocks), self.SCALE_INV_MEAN, self.SCALE_INV_STD)
        if self.weight_format == "fp8":
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
        else:
            w1_q, w2_q, fc1_scale, fc2_scale = self._build_mxfp4_weights()

        topk_ids = torch.randint(
            low=0, high=experts, size=(self.tokens, self.topk), dtype=torch.int32, device=self.device
        )
        topk_weights = torch.rand((self.tokens, self.topk), dtype=torch.float32, device=self.device)
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

        if self.weight_format == "fp8":
            w1_q_kernel = self._pack_blocks(w1_q, block_n=16, block_k=32).reshape(experts, inter_dim * 2, dim)
            w2_q_kernel = self._pack_blocks(w2_q, block_n=16, block_k=32).reshape(experts, dim, inter_dim)
            input_scale_kernel = input_scale.contiguous()
            fc1_scale_kernel = fc1_scale.contiguous()
            fc2_scale_kernel = (
                fc2_scale.contiguous()
                .view(experts, w2_row_blocks, inter_blocks)
                .transpose(1, 2)
                .reshape(experts, inter_blocks * w2_row_blocks)
                .contiguous()
            )
        else:
            w1_q_kernel, fc1_scale_kernel = petit_kernel.repack_moe_mxfp4_kernel_layout(
                w1_q, fc1_scale
            )
            w2_q_kernel, fc2_scale_kernel = petit_kernel.repack_moe_mxfp4_kernel_layout(
                w2_q, fc2_scale
            )
            input_scale_kernel = input_scale.t().contiguous()

        return {
            "input_q": input_q,
            "w1_q_kernel": w1_q_kernel,
            "w2_q_kernel": w2_q_kernel,
            "sorted_token_ids": sorted_token_ids,
            "sorted_weights": sorted_weights,
            "sorted_expert_ids": sorted_expert_ids,
            "num_valid_ids": num_valid_ids,
            "num_valid_m_blocks": num_valid_m_blocks,
            "topk": self.topk,
            "input_scale": input_scale.contiguous(),
            "input_scale_kernel": input_scale_kernel,
            "fc1_scale_kernel": fc1_scale_kernel,
            "fc2_scale_kernel": fc2_scale_kernel,
            "weight_format": self.weight_format,
        }


def make_run_fn(
    backend: str, data: Dict[str, torch.Tensor], num_persistent_tgs: int
) -> Tuple[Callable[[], torch.Tensor], str]:
    if backend == "petit":
        if data["weight_format"] == "mxfp4":
            def run() -> torch.Tensor:
                return petit_kernel.fused_moe_fp8_blockscale_g1u1_mxfp4(
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
                    num_persistent_tgs,
                )
        else:
            def run() -> torch.Tensor:
                return petit_kernel.fused_moe_fp8_blockscale_g1u1(
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
                    num_persistent_tgs,
                )

        return run, "pybind"

    if num_persistent_tgs > 0:
        raise RuntimeError(
            "Persistent launch forcing is only supported for backend=petit; "
            "backend=aiter uses its own kernel heuristic."
        )

    try:
        import aiter  # type: ignore
    except Exception as exc:
        raise RuntimeError("Failed to import aiter for backend=aiter.") from exc

    input_scale_aiter = torch.empty_like(data["input_scale"])
    num_rows = torch.tensor([data["input_q"].shape[0]], dtype=torch.int32, device=data["input_q"].device)
    aiter.partial_transpose(input_scale_aiter, data["input_scale"], num_rows=num_rows)

    def run_aiter() -> torch.Tensor:
        out = torch.zeros(
            (data["input_q"].shape[0], data["input_q"].shape[1]),
            dtype=torch.bfloat16,
            device=data["input_q"].device,
        )
        aiter.fmoe_fp8_blockscale_g1u1(
            out,
            data["input_q"],
            data["w1_q_kernel"],
            data["w2_q_kernel"],
            data["sorted_token_ids"],
            data["sorted_weights"],
            data["sorted_expert_ids"],
            data["num_valid_ids"],
            int(data["topk"]),
            input_scale_aiter,
            data["fc1_scale_kernel"],
            data["fc2_scale_kernel"],
            "",
            BLOCK_N,
            BLOCK_K,
            None,
        )
        return out

    return run_aiter, "asm"


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


def compute_metrics(args: argparse.Namespace, data: Dict[str, torch.Tensor], avg_ms: float) -> Tuple[float, float]:
    routes = args.tokens * args.topk
    flops = 6.0 * routes * args.dim * args.inter_dim
    tflops = flops / (avg_ms * 1e-3) / 1e12

    byte_tensors = [
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
    ]
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
    if not (hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")):
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
        weight_format=args.weight_format,
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
    print(
        f"backend={args.backend} impl={impl} graph={1 if used_graph else 0} "
        f"weight_format={args.weight_format} "
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
