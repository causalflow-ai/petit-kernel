"""Benchmark the standalone MegaMoE MXFP4 token shuffle with torchrun."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

import petit_kernel
from petit_kernel import _test_ops

SUPPORTED_EP_SIZES = (2, 4, 8)
GLOBAL_EXPERTS = 128
TOPK = 4
MXFP4_ROW_BYTES = 1536
NUM_WORKGROUPS = 256
IMPLEMENTATIONS = ("pull", "direct_push")
SELECTABLE_IMPLEMENTATIONS = ("pull", "pull_epoch", "direct_push")
ROUTING_DISTRIBUTIONS = ("uniform", "single_rank", "skewed", "invalid")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the current MegaMoE MXFP4 token shuffle."
    )
    parser.add_argument(
        "--tokens",
        type=positive_int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024],
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=20)
    parser.add_argument("--repeat", type=positive_int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--implementation",
        choices=(*SELECTABLE_IMPLEMENTATIONS, "both", "all"),
        default="both",
    )
    parser.add_argument(
        "--routing",
        choices=ROUTING_DISTRIBUTIONS,
        default="uniform",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--epoch-wrap",
        action="store_true",
        help="seed epoch state at UINT32_MAX before timing pull_epoch",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()
    if any(tokens > 1024 for tokens in args.tokens):
        parser.error("token counts must not exceed 1024")
    return args


def validate_launch(world_size: int, local_rank: int) -> str:
    if world_size not in SUPPORTED_EP_SIZES:
        raise RuntimeError(f"EP must be one of {SUPPORTED_EP_SIZES}")
    if int(os.environ.get("LOCAL_WORLD_SIZE", world_size)) != world_size:
        raise RuntimeError("the VMM symmetric heap requires one torchrun node")
    properties = torch.cuda.get_device_properties(local_rank)
    architecture = getattr(properties, "gcnArchName", properties.name)
    devices: list[tuple[str, int] | None] = [None] * world_size
    dist.all_gather_object(
        devices, (architecture, properties.multi_processor_count)
    )
    if any(
        device is None
        or not device[0].startswith(("gfx940", "gfx941", "gfx942", "gfx950"))
        or device[1] < NUM_WORKGROUPS
        for device in devices
    ):
        raise RuntimeError(
            "token shuffle requires supported GPUs with at least "
            f"{NUM_WORKGROUPS} compute units; got {devices}"
        )
    return architecture


def make_inputs(
    tokens: int,
    rank: int,
    world_size: int,
    seed: int,
    routing: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + rank * 17 + tokens)
    rows = torch.randint(
        0,
        256,
        (tokens, MXFP4_ROW_BYTES),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    # Top-k over random logits gives unique, approximately uniform experts and
    # matches the route occupancy of the full-path MegaMoE benchmark.
    logits = torch.rand(
        (tokens, GLOBAL_EXPERTS), device=device, generator=generator
    )
    ids = logits.topk(TOPK, dim=-1).indices.to(torch.int32).contiguous()
    if routing == "single_rank":
        destination = rank % world_size
        first_expert = destination * (GLOBAL_EXPERTS // world_size)
        ids[:] = torch.arange(
            first_expert,
            first_expert + TOPK,
            dtype=torch.int32,
            device=device,
        )
    elif routing == "skewed":
        skewed_tokens = max(1, tokens * 7 // 8)
        ids[:skewed_tokens] = torch.arange(
            TOPK, dtype=torch.int32, device=device
        )
    elif routing == "invalid":
        ids[:, -1] = -1
    weights = torch.rand(
        (tokens, TOPK), device=device, generator=generator
    ).contiguous()
    return rows, ids, weights


def reference_dispatch(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, ids, weights = inputs
    tokens = rows.size(0)
    local_experts = GLOBAL_EXPERTS // world_size
    route_rows = rows[:, None, :].expand(-1, TOPK, -1).reshape(
        -1, MXFP4_ROW_BYTES
    )
    experts = ids.flatten()
    route_weights = weights.flatten()
    route_keys = rank * tokens * TOPK + torch.arange(
        tokens * TOPK, dtype=torch.int64, device=rows.device
    )
    valid = (experts >= 0) & (experts < GLOBAL_EXPERTS)
    route_rows = route_rows[valid]
    route_weights = route_weights[valid]
    route_keys = route_keys[valid]
    experts = experts[valid]
    destinations = torch.div(experts, local_experts, rounding_mode="floor")
    order = torch.argsort(destinations, stable=True)
    send_counts = torch.bincount(destinations, minlength=world_size).to(
        torch.int64
    )
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    input_splits = [int(value) for value in send_counts.cpu().tolist()]
    output_splits = [int(value) for value in recv_counts.cpu().tolist()]
    received = int(recv_counts.sum().item())
    def exchange(tensor: torch.Tensor) -> torch.Tensor:
        send = tensor[order].contiguous()
        output = torch.empty(
            (received, *send.shape[1:]), dtype=send.dtype, device=send.device
        )
        dist.all_to_all_single(
            output,
            send,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )
        return output

    return (
        exchange(route_rows),
        exchange(route_weights),
        exchange(route_keys),
        exchange(experts),
    )


def unpack_shuffle(
    shuffled: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rank: int,
    world_size: int,
    tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, weights, metadata, counts_tensor = shuffled
    counts = [int(value) for value in counts_tensor.cpu().tolist()]
    indices: list[torch.Tensor] = []
    experts: list[torch.Tensor] = []
    offset = 0
    for local_expert, count in enumerate(counts):
        indices.append(torch.arange(offset, offset + count, device=rows.device))
        experts.append(
            torch.full(
                (count,),
                rank * (GLOBAL_EXPERTS // world_size) + local_expert,
                dtype=torch.int32,
                device=rows.device,
            )
        )
        offset += math.ceil(count / 32) * 32
    selected = torch.cat(indices)
    selected_metadata = metadata[selected]
    route_keys = (
        ((selected_metadata >> 32) & 0xFF) * tokens * TOPK
        + (selected_metadata & 0xFFFF_FFFF)
    )
    return rows[selected], weights[selected], route_keys, torch.cat(experts)


def check_result(
    shuffled: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rank: int,
    world_size: int,
    tokens: int,
) -> None:
    actual = unpack_shuffle(shuffled, rank, world_size, tokens)
    actual_order = torch.argsort(actual[2])
    expected_order = torch.argsort(expected[2])
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor[actual_order],
            expected_tensor[expected_order],
            rtol=0,
            atol=0,
        )


def percentile(samples: list[float], value: float) -> float:
    ordered = sorted(samples)
    return ordered[int((len(ordered) - 1) * value)]


def append_csv(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def append_jsonl(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(result, separators=(",", ":")) + "\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def implementation_ops(implementation: str):
    if implementation == "pull":
        return (
            _test_ops.token_shuffle_mxfp4,
            _test_ops.token_shuffle_mxfp4_prepare,
            _test_ops.token_shuffle_mxfp4_run,
        )
    if implementation == "pull_epoch":
        return (
            _test_ops.token_shuffle_epoch_mxfp4,
            _test_ops.token_shuffle_epoch_mxfp4_prepare,
            _test_ops.token_shuffle_epoch_mxfp4_run,
        )
    return (
        _test_ops.token_shuffle_direct_push_mxfp4,
        _test_ops.token_shuffle_direct_push_mxfp4_prepare,
        _test_ops.token_shuffle_direct_push_mxfp4_run,
    )


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        architecture = validate_launch(world_size, local_rank)
        if args.implementation == "both":
            implementations = IMPLEMENTATIONS
        elif args.implementation == "all":
            implementations = SELECTABLE_IMPLEMENTATIONS
        else:
            implementations = (args.implementation,)
        # All implementations share the production MegaMoE layout. Allocate
        # one heap and reuse it across implementations and every M in the sweep.
        workspace = petit_kernel.create_vmm_symmetric_heap(world_size)
        revision = git_commit()
        pull_p50_by_tokens: dict[int, float] = {}

        for tokens in args.tokens:
            inputs = make_inputs(
                tokens,
                rank,
                world_size,
                args.seed,
                args.routing,
                device,
            )
            expected = (
                reference_dispatch(inputs, rank, world_size)
                if args.check
                else None
            )
            for implementation in implementations:
                checked_op, prepare_op, run_op = implementation_ops(
                    implementation
                )
                if expected is not None:
                    shuffled = checked_op(workspace, *inputs)
                    torch.cuda.synchronize()
                    check_result(shuffled, expected, rank, world_size, tokens)

                prepare_op(workspace, *inputs)
                if args.epoch_wrap and implementation == "pull_epoch":
                    _test_ops.token_shuffle_epoch_seed_wrap(workspace)
                local_experts = GLOBAL_EXPERTS // world_size
                expert_counts = torch.empty(
                    local_experts, dtype=torch.int32, device=device
                )
                profile_cycles = torch.empty(
                    1, dtype=torch.int64, device=device
                )
                for _ in range(args.warmup):
                    run_op(
                        workspace, tokens, expert_counts, profile_cycles
                    )
                torch.cuda.synchronize()
                dist.barrier(device_ids=[local_rank])

                elapsed_samples: list[float] = []
                cycle_samples: list[float] = []
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                for _ in range(args.repeat):
                    start.record()
                    run_op(
                        workspace, tokens, expert_counts, profile_cycles
                    )
                    end.record()
                    end.synchronize()
                    elapsed_us = torch.tensor(
                        start.elapsed_time(end) * 1000.0, device=device
                    )
                    cycle_samples.append(float(profile_cycles.item()))
                    dist.all_reduce(elapsed_us, op=dist.ReduceOp.MAX)
                    elapsed_samples.append(float(elapsed_us.item()))

                local_cycles = torch.tensor(
                    cycle_samples, dtype=torch.float64, device=device
                )
                gathered_cycles = [
                    torch.empty_like(local_cycles) for _ in range(world_size)
                ]
                dist.all_gather(gathered_cycles, local_cycles)

                def max_rank_cycle_percentile(
                    value: float,
                    rank_cycle_samples: list[torch.Tensor] = gathered_cycles,
                ) -> float:
                    return max(
                        percentile(rank_cycles.cpu().tolist(), value)
                        for rank_cycles in rank_cycle_samples
                    )

                mean_us = sum(elapsed_samples) / len(elapsed_samples)
                p50_us = percentile(elapsed_samples, 0.50)
                if implementation == "pull":
                    pull_p50_by_tokens[tokens] = p50_us
                speedup = (
                    pull_p50_by_tokens[tokens] / p50_us
                    if tokens in pull_p50_by_tokens
                    else 1.0
                )
                global_tokens = tokens * world_size
                result: dict[str, object] = {
                    "backend": "petit_token_shuffle_mxfp4",
                    "implementation": implementation,
                    "protocol": (
                        "fixed_role_compact_push"
                        if implementation == "direct_push"
                        else "global_barrier_pull"
                    ),
                    "routing": args.routing,
                    "commit": revision,
                    "gcn_arch_name": architecture,
                    "ep_size": world_size,
                    "tokens_per_rank": tokens,
                    "experts": GLOBAL_EXPERTS,
                    "topk": TOPK,
                    "hidden_size": 2880,
                    "workgroups": NUM_WORKGROUPS,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "check": args.check,
                    "mean_us": mean_us,
                    "p50_us": p50_us,
                    "p90_us": percentile(elapsed_samples, 0.90),
                    "p99_us": percentile(elapsed_samples, 0.99),
                    "speedup_vs_pull_p50": speedup,
                    "profile_cycles_max_rank_p50": (
                        max_rank_cycle_percentile(0.50)
                    ),
                    "profile_cycles_max_rank_p90": (
                        max_rank_cycle_percentile(0.90)
                    ),
                    "profile_cycles_max_rank_p99": (
                        max_rank_cycle_percentile(0.99)
                    ),
                    "tokens_per_s": global_tokens * 1_000_000.0 / mean_us,
                    "routes_per_s": (
                        global_tokens * TOPK * 1_000_000.0 / mean_us
                    ),
                }
                if rank == 0:
                    print(
                        json.dumps(result, separators=(",", ":")),
                        flush=True,
                    )
                    if args.csv:
                        append_csv(args.csv, result)
                    if args.jsonl:
                        append_jsonl(args.jsonl, result)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
