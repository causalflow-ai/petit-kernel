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


def prepare_mxfp4_rows(
    values: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    value_bytes = values.size(1)
    scale_bytes = scales.size(1)
    scale_row_bytes = (scale_bytes + 15) & ~15
    storage = torch.zeros(
        (values.size(0), value_bytes + scale_row_bytes),
        dtype=torch.uint8,
        device=values.device,
    )
    external_values = storage[:, :value_bytes]
    external_scales = storage[:, value_bytes : value_bytes + scale_bytes]
    external_values.copy_(values)
    external_scales.copy_(scales)
    return external_values, external_scales


def expect_runtime_error(fn, message: str) -> None:
    try:
        fn()
    except RuntimeError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"expected RuntimeError containing {message!r}")


def expect_value_error(fn, message: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


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
    if values.size(0) == 0:
        return values
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
    parser.add_argument("--uneven-tokens", action="store_true")
    parser.add_argument("--varying-tokens", action="store_true")
    parser.add_argument("--skewed-routing", action="store_true")
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--graph-layers", type=int, default=1)
    args = parser.parse_args()

    if args.num_experts is not None and not args.registered_shape:
        parser.error("--num-experts requires --registered-shape")
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.graph_layers <= 0:
        parser.error("--graph-layers must be positive")

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
        max_tokens = args.tokens
        tokens = (
            rank % max_tokens + 1
            if args.uneven_tokens
            else (
                0
                if args.zero_token_rank and rank == world_size - 1
                else max_tokens
            )
        )
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
        if args.skewed_routing:
            topk_ids = (
                torch.arange(topk, dtype=torch.int32, device=device)
                .expand(tokens, -1)
                .contiguous()
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
        alternate_values = values
        alternate_scales = input_scales
        alternate_ids = topk_ids
        alternate_weights = topk_weights
        alternate_expected = expected
        if args.repeat > 1:
            if packed_input:
                alternate_values = random_mxfp4(
                    (tokens, hidden // 2), generator, device
                )
                alternate_scales = torch.randint(
                    120,
                    124,
                    (tokens, hidden // 32),
                    dtype=torch.uint8,
                    device=device,
                    generator=generator,
                )
            else:
                alternate_values = (
                    torch.randn(
                        (tokens, hidden), device=device, generator=generator
                    )
                    * 0.25
                ).to(torch.bfloat16)
            alternate_ids = ((topk_ids + local_experts) % experts).contiguous()
            alternate_weights = torch.flip(topk_weights, dims=(1,)).contiguous()
            alternate_expected = reference(
                alternate_values,
                alternate_scales,
                alternate_ids,
                alternate_weights,
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
            assert (
                (alternate_expected.float() - expected.float()).abs().mean().item()
                > 0.01
            )
        external_values = values
        external_scales = input_scales
        alternate_external_values = alternate_values
        alternate_external_scales = alternate_scales
        if packed_input:
            assert input_scales is not None
            assert alternate_scales is not None
            external_values, external_scales = prepare_mxfp4_rows(
                values, input_scales
            )
            alternate_external_values, alternate_external_scales = (
                prepare_mxfp4_rows(alternate_values, alternate_scales)
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
        external = second
        if world_size > 1 and packed_input:
            external = config.run(
                heap,
                w13,
                w2,
                s13,
                s2,
                tokens,
                w13_bias=b13,
                w2_bias=b2,
                inputs=petit_kernel.MegaMoeInputViews(
                    external_values,
                    external_scales,
                    topk_ids,
                    topk_weights,
                ),
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
        check_output(external[:, :hidden], expected)

        if args.varying_tokens:
            varying_out = torch.empty(
                (tokens, compute_hidden), dtype=torch.bfloat16, device=device
            )
            for current_tokens in (tokens, 1, 1, tokens):
                config.run(
                    heap,
                    w13,
                    w2,
                    s13,
                    s2,
                    current_tokens,
                    w13_bias=b13,
                    w2_bias=b2,
                    out=varying_out[:current_tokens],
                )
                torch.cuda.synchronize(device)
                check_output(
                    varying_out[:current_tokens, :hidden],
                    expected[:current_tokens],
                )

        if args.repeat > 1:
            repeated_out = torch.empty(
                (tokens, compute_hidden), dtype=torch.bfloat16, device=device
            )
            repeated = torch.empty(
                (args.repeat, tokens, hidden),
                dtype=torch.bfloat16,
                device=device,
            )
            for iteration in range(args.repeat):
                alternate = iteration % 2 != 0
                current_values = alternate_values if alternate else values
                current_scales = (
                    alternate_scales if alternate else input_scales
                )
                current_ids = alternate_ids if alternate else topk_ids
                current_weights = (
                    alternate_weights if alternate else topk_weights
                )
                views.tokens[:tokens].copy_(current_values)
                views.expert_ids[:tokens].copy_(current_ids)
                views.expert_weights[:tokens].copy_(current_weights)
                if packed_input:
                    views.scales[:tokens].copy_(current_scales)
                config.run(
                    heap,
                    w13,
                    w2,
                    s13,
                    s2,
                    tokens,
                    w13_bias=b13,
                    w2_bias=b2,
                    out=repeated_out,
                )
                repeated[iteration].copy_(repeated_out[:, :hidden])
            torch.cuda.synchronize(device)
            repeated_expected = torch.stack(
                [
                    alternate_expected if iteration % 2 else expected
                    for iteration in range(args.repeat)
                ]
            )
            repeated_error = (
                repeated.float() - repeated_expected.float()
            ).abs()
            assert torch.isfinite(repeated).all()
            repeated_mean_error = repeated_error.mean(dim=(1, 2))
            assert repeated_mean_error.max().item() < 0.10, (
                f"rank={rank} per-iteration mean errors="
                f"{repeated_mean_error.cpu().tolist()}, "
                f"last-vs-first-pattern="
                f"{(repeated[-1].float() - expected.float()).abs().mean().item()}, "
                f"last-abs-mean={repeated[-1].float().abs().mean().item()}, "
                f"expected-abs-mean="
                f"{alternate_expected.float().abs().mean().item()}"
            )
            assert (
                torch.quantile(repeated_error.flatten(1), 0.99, dim=1)
                .max()
                .item()
                < 0.40
            )

        if packed_input and tokens > 0:

            def run_external(**kwargs: object) -> torch.Tensor:
                return config.run(
                    heap,
                    w13,
                    w2,
                    s13,
                    s2,
                    tokens,
                    w13_bias=b13,
                    w2_bias=b2,
                    **kwargs,
                )

            if world_size == 1:
                expect_runtime_error(
                    lambda: run_external(
                        inputs=petit_kernel.MegaMoeInputViews(
                            external_values,
                            external_scales,
                            topk_ids,
                            topk_weights,
                        ),
                    ),
                    "unsupported",
                )
            else:
                expect_value_error(
                    lambda: run_external(
                        inputs=petit_kernel.MegaMoeInputViews(
                            external_values, None, topk_ids, topk_weights
                        )
                    ),
                    "include scales",
                )
                expect_runtime_error(
                    lambda: run_external(
                        inputs=petit_kernel.MegaMoeInputViews(
                            external_values[:, :-1],
                            external_scales,
                            topk_ids,
                            topk_weights,
                        )
                    ),
                    "input_tokens must be",
                )
                expect_value_error(
                    lambda: run_external(
                        inputs=petit_kernel.MegaMoeInputViews(
                            external_values,
                            external_scales,
                            topk_ids.long(),
                            topk_weights,
                        )
                    ),
                    "input_topk_ids has invalid dtype",
                )

        if args.cuda_graph:
            def graph_external_inputs(
                alternate: bool,
            ) -> dict[str, object]:
                if not packed_input or world_size == 1:
                    return {}
                return {
                    "inputs": petit_kernel.MegaMoeInputViews(
                        alternate_external_values
                        if alternate
                        else external_values,
                        alternate_external_scales
                        if alternate
                        else external_scales,
                        alternate_ids if alternate else topk_ids,
                        alternate_weights if alternate else topk_weights,
                    )
                }

            graph_outputs = [
                torch.empty(
                    (tokens, compute_hidden),
                    dtype=torch.bfloat16,
                    device=device,
                )
                for _ in range(args.graph_layers)
            ]
            dist.barrier()
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    for layer in range(args.graph_layers):
                        layer_alternate = layer % 2 != 0
                        views.tokens[:tokens].copy_(
                            alternate_values if layer_alternate else values
                        )
                        views.expert_ids[:tokens].copy_(
                            alternate_ids if layer_alternate else topk_ids
                        )
                        views.expert_weights[:tokens].copy_(
                            alternate_weights
                            if layer_alternate
                            else topk_weights
                        )
                        if packed_input:
                            views.scales[:tokens].copy_(
                                alternate_scales
                                if layer_alternate
                                else input_scales
                            )
                        config.run(
                            heap,
                            w13,
                            w2,
                            s13,
                            s2,
                            tokens,
                            w13_bias=b13,
                            w2_bias=b2,
                            out=graph_outputs[layer],
                            **graph_external_inputs(layer_alternate),
                        )
            warmup_stream.synchronize()
            dist.barrier()

            graph = torch.cuda.CUDAGraph()
            graph_returns = []
            with torch.cuda.graph(graph):
                for layer in range(args.graph_layers):
                    layer_alternate = layer % 2 != 0
                    views.tokens[:tokens].copy_(
                        alternate_values if layer_alternate else values
                    )
                    views.expert_ids[:tokens].copy_(
                        alternate_ids if layer_alternate else topk_ids
                    )
                    views.expert_weights[:tokens].copy_(
                        alternate_weights if layer_alternate else topk_weights
                    )
                    if packed_input:
                        views.scales[:tokens].copy_(
                            alternate_scales
                            if layer_alternate
                            else input_scales
                        )
                    graph_returns.append(
                        config.run(
                            heap,
                            w13,
                            w2,
                            s13,
                            s2,
                            tokens,
                            w13_bias=b13,
                            w2_bias=b2,
                            out=graph_outputs[layer],
                            **graph_external_inputs(layer_alternate),
                        )
                    )
            for graph_ret, graph_out in zip(graph_returns, graph_outputs):
                assert graph_ret.data_ptr() == graph_out.data_ptr()

            for iteration in range(args.repeat):
                alternate = iteration % 2 != 0
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize(device)
                for layer, graph_ret in enumerate(graph_returns):
                    check_output(
                        graph_ret[:, :hidden],
                        alternate_expected if layer % 2 else expected,
                    )
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
