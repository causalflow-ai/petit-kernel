from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

import petit_kernel

_MXFP4 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def random_mxfp4(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    low = torch.randint(
        0, 16, shape, dtype=torch.uint8, device=device, generator=generator
    )
    high = torch.randint(
        0, 16, shape, dtype=torch.uint8, device=device, generator=generator
    )
    low[low == 8] = 0
    high[high == 8] = 0
    return (low | (high << 4)).contiguous()


def dequant_mxfp4(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_MXFP4, dtype=torch.float32, device=values.device)
    unpacked = torch.empty(
        (*values.shape[:-1], values.shape[-1] * 2),
        dtype=torch.float32,
        device=values.device,
    )
    unpacked[..., 0::2] = table[(values & 0xF).long()]
    unpacked[..., 1::2] = table[(values >> 4).long()]
    scale = torch.pow(2.0, scales.float() - 127.0)
    return (
        unpacked.view(*unpacked.shape[:-1], scales.shape[-1], 32)
        * scale.unsqueeze(-1)
    ).flatten(-2)


def quantize_dequant_mxfp4(values: torch.Tensor) -> torch.Tensor:
    rows = values.to(torch.bfloat16).float().view(values.size(0), -1, 32)
    absmax = rows.abs().amax(dim=-1)
    required = (absmax / 6.0).clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.pow(2.0, torch.ceil(torch.log2(required)))
    scale = torch.where(absmax < 1.0e-12, torch.ones_like(scale), scale)
    normalized = rows / scale.unsqueeze(-1)
    magnitudes = torch.tensor(_MXFP4[:8], device=values.device)
    nearest = (normalized.abs().unsqueeze(-1) - magnitudes).abs().argmin(-1)
    return (
        torch.copysign(magnitudes[nearest], normalized) * scale.unsqueeze(-1)
    ).flatten(1)


def exchange(
    tensor: torch.Tensor, send_splits: list[int], recv_splits: list[int]
) -> torch.Tensor:
    output = torch.empty(
        (sum(recv_splits), *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_to_all_single(
        output,
        tensor,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )
    return output


def reference(
    values: torch.Tensor,
    input_scales: torch.Tensor | None,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    raw_w13: torch.Tensor,
    raw_w2: torch.Tensor,
    raw_s13: torch.Tensor,
    raw_s2: torch.Tensor,
    raw_b13: torch.Tensor,
    raw_b2: torch.Tensor,
    *,
    world_size: int,
    experts: int,
    topk: int,
    hidden: int,
    compute_hidden: int,
    intermediate: int,
    packed_input: bool,
) -> torch.Tensor:
    x = dequant_mxfp4(values, input_scales) if packed_input else values.float()
    x = torch.nn.functional.pad(x, (0, compute_hidden - hidden))
    tokens = values.size(0)
    routes = tokens * topk
    route_x = x[:, None, :].expand(-1, topk, -1).reshape(routes, compute_hidden)
    route_ids = topk_ids.flatten()
    route_weights = topk_weights.flatten()
    local_experts = experts // world_size
    destinations = torch.div(route_ids, local_experts, rounding_mode="floor")
    order = torch.argsort(destinations, stable=True)
    send_counts = torch.bincount(destinations, minlength=world_size).to(torch.int64)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    send_splits = [int(count) for count in send_counts.cpu()]
    recv_splits = [int(count) for count in recv_counts.cpu()]

    recv_x = exchange(route_x[order].contiguous(), send_splits, recv_splits)
    recv_weights = exchange(
        route_weights[order, None].contiguous(), send_splits, recv_splits
    ).flatten()
    local_ids = (route_ids % local_experts).to(torch.int64)
    recv_experts = exchange(
        local_ids[order, None].contiguous(), send_splits, recv_splits
    ).flatten()

    # Matrices are shared by local experts; expert-specific biases still make
    # incorrect routing metadata observable.
    w13 = dequant_mxfp4(raw_w13[0], raw_s13[0]).transpose(0, 1)
    w2 = dequant_mxfp4(raw_w2[0], raw_s2[0]).transpose(0, 1)
    hidden_values = recv_x @ w13 + raw_b13[recv_experts].float()
    gate = hidden_values[:, :intermediate].clamp(max=7.0)
    up = hidden_values[:, intermediate:].clamp(-7.0, 7.0)
    activated = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
    activated = (
        quantize_dequant_mxfp4(activated)
        if packed_input
        else activated.to(torch.bfloat16).float()
    )
    computed = (activated @ w2 + raw_b2[recv_experts].float())[
        :, :hidden
    ] * recv_weights[:, None]

    returned = exchange(computed, recv_splits, send_splits)
    route_output = torch.empty_like(returned)
    route_output[order] = returned
    return route_output.view(tokens, topk, hidden).sum(1)


def routing(
    tokens: int,
    rank: int,
    world_size: int,
    experts: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = torch.arange(tokens, dtype=torch.int32, device=device)[:, None]
    route = torch.arange(topk, dtype=torch.int32, device=device)[None, :]
    local_experts = experts // world_size
    destination = (rank + token + token // 16 + route) % world_size
    local = (token * topk + route) % local_experts
    ids = (destination * local_experts + local).to(torch.int32).contiguous()
    unnormalized = 1.0 + ((token + 2 * route) % 7).float()
    weights = (unnormalized / unnormalized.sum(1, keepdim=True)).contiguous()
    return ids, weights


def check_output(actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = (actual.float() - expected).abs()
    assert torch.isfinite(actual).all()
    assert error.numel() == 0 or error.mean().item() < 0.10
    assert error.numel() == 0 or torch.quantile(error, 0.99).item() < 0.40


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", choices=("bf16", "mxfp4"), required=True)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--registered-shape", action="store_true")
    parser.add_argument("--zero-token-rank", action="store_true")
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--num-experts", type=int)
    args = parser.parse_args()

    if args.num_experts is not None and not args.registered_shape:
        parser.error("--num-experts requires --registered-shape")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    try:
        if args.registered_shape:
            experts = args.num_experts or 32
            topk, hidden = 4, 2880
        else:
            experts = 1 if world_size == 1 else 2 * world_size
            topk = 1 if world_size == 1 else 2
            hidden = 256
        intermediate = 3072
        max_tokens = 17
        tokens = 0 if args.zero_token_rank and rank == world_size - 1 else max_tokens
        local_experts = experts // world_size
        generator = torch.Generator(device=device).manual_seed(0x4D4F45 + rank)
        packed_input = args.activation == "mxfp4"
        config = petit_kernel.MegaMoeConfig(
            world_size=world_size,
            num_experts=experts,
            topk=topk,
            model_dim=hidden,
            activation=petit_kernel.MegaMoeActivation(args.activation),
            stages=(
                petit_kernel.MegaMoeStages.two_stage
                if args.two_stage
                else petit_kernel.MegaMoeStages.one_stage
            ),
        )
        compute_hidden = config.compute_model_dim

        if packed_input:
            values = random_mxfp4((tokens, hidden // 2), generator, device)
            input_scales = torch.randint(
                119,
                123,
                (tokens, hidden // 32),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            )
        else:
            values = (
                torch.randn((tokens, hidden), device=device, generator=generator)
                * 0.125
            ).to(torch.bfloat16)
            input_scales = None

        raw_w13 = (
            random_mxfp4(
                (1, 2 * intermediate, compute_hidden // 2), generator, device
            )
            .expand(local_experts, -1, -1)
            .clone()
        )
        raw_w2 = (
            random_mxfp4(
                (1, compute_hidden, intermediate // 2), generator, device
            )
            .expand(local_experts, -1, -1)
            .clone()
        )
        raw_s13 = (
            torch.randint(
                118,
                122,
                (1, 2 * intermediate, compute_hidden // 32),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            )
            .expand(local_experts, -1, -1)
            .clone()
        )
        raw_s2 = (
            torch.randint(
                118,
                122,
                (1, compute_hidden, intermediate // 32),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            )
            .expand(local_experts, -1, -1)
            .clone()
        )
        raw_b13 = (
            torch.randn(
                (local_experts, 2 * intermediate),
                device=device,
                generator=generator,
            )
            * 0.125
        ).to(torch.bfloat16)
        raw_b2 = (
            torch.randn(
                (local_experts, compute_hidden), device=device, generator=generator
            )
            * 0.125
        ).to(torch.bfloat16)
        topk_ids, topk_weights = routing(
            tokens, rank, world_size, experts, topk, device
        )

        layout = petit_kernel.MoeKernelLayout.native_mxfp4
        w13, s13 = petit_kernel.repack_moe_kernel_layout(
            raw_w13, raw_s13, layout=layout
        )
        w2, s2 = petit_kernel.repack_moe_kernel_layout(
            raw_w2, raw_s2, layout=layout
        )
        b13 = petit_kernel.repack_moe_kernel_layout(raw_b13, layout=layout)
        b2 = petit_kernel.repack_moe_kernel_layout(raw_b2, layout=layout)

        expected = reference(
            values,
            input_scales,
            topk_ids,
            topk_weights,
            raw_w13,
            raw_w2,
            raw_s13,
            raw_s2,
            raw_b13,
            raw_b2,
            world_size=world_size,
            experts=experts,
            topk=topk,
            hidden=hidden,
            compute_hidden=compute_hidden,
            intermediate=intermediate,
            packed_input=packed_input,
        )
        heap = petit_kernel.create_vmm_symmetric_heap(world_size)
        views = config.input_views(heap, max_tokens)
        views.tokens[:tokens].copy_(values)
        views.expert_ids[:tokens].copy_(topk_ids)
        views.expert_weights[:tokens].copy_(topk_weights)
        if packed_input:
            assert views.scales is not None
            views.scales[:tokens].copy_(input_scales)
        else:
            assert views.scales is None

        first = config.run(
            heap,
            w13,
            w2,
            s13,
            s2,
            tokens,
            w13_bias=b13,
            w2_bias=b2,
        )
        caller_out = torch.empty(
            (tokens, hidden), dtype=torch.bfloat16, device=device
        )
        second = config.run(
            heap,
            w13,
            w2,
            s13,
            s2,
            tokens,
            w13_bias=b13,
            w2_bias=b2,
            out=caller_out,
        )
        torch.cuda.synchronize(device)
        assert first.shape == (tokens, compute_hidden)
        assert first.is_contiguous()
        assert second is caller_out
        assert second.data_ptr() == caller_out.data_ptr()
        check_output(first[:, :hidden], expected)
        assert second.shape == (tokens, hidden)
        assert second.is_contiguous()
        check_output(second, expected)

        if args.cuda_graph:
            graph_values = values.clone()
            graph_ids = topk_ids.clone()
            graph_weights = topk_weights.clone()
            graph_scales = input_scales.clone() if packed_input else None
            graph_out = torch.empty(
                (tokens, compute_hidden), dtype=torch.bfloat16, device=device
            )
            dist.barrier()
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    views.tokens[:tokens].copy_(graph_values)
                    views.expert_ids[:tokens].copy_(graph_ids)
                    views.expert_weights[:tokens].copy_(graph_weights)
                    if packed_input:
                        views.scales[:tokens].copy_(graph_scales)
                    config.run(
                        heap,
                        w13,
                        w2,
                        s13,
                        s2,
                        tokens,
                        w13_bias=b13,
                        w2_bias=b2,
                        out=graph_out,
                    )
            warmup_stream.synchronize()
            dist.barrier()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                views.tokens[:tokens].copy_(graph_values)
                views.expert_ids[:tokens].copy_(graph_ids)
                views.expert_weights[:tokens].copy_(graph_weights)
                if packed_input:
                    views.scales[:tokens].copy_(graph_scales)
                graph_ret = config.run(
                    heap,
                    w13,
                    w2,
                    s13,
                    s2,
                    tokens,
                    w13_bias=b13,
                    w2_bias=b2,
                    out=graph_out,
                )
            assert graph_ret.data_ptr() == graph_out.data_ptr()

            for _ in range(2):
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize(device)
                check_output(graph_ret[:, :hidden], expected)
        dist.barrier()
        if rank == 0:
            graph_suffix = " with CUDA graph" if args.cuda_graph else ""
            print(
                f"MegaMoE {'two-stage ' if args.two_stage else ''}"
                f"{args.activation} passed with EP={world_size}"
                f"{graph_suffix}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
