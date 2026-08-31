from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

import petit_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=("dsv32", "dsv4"), required=True)
    parser.add_argument("--capture-sizes", default="128,120,112")
    parser.add_argument("--layers", type=int, default=61)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--skewed", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise ValueError("DeepSeek MegaMoE multigraph test requires EP8")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    try:
        capture_sizes = [int(size) for size in args.capture_sizes.split(",")]
        max_tokens = max(capture_sizes)
        if args.shape == "dsv32":
            experts, topk, inter_dim = 256, 8, 2048
        else:
            experts, topk, inter_dim = 384, 6, 3072
        hidden = 7168
        compute_hidden = 7168
        local_experts = experts // world_size

        config = petit_kernel.MegaMoeConfig(
            world_size=world_size,
            num_experts=experts,
            topk=topk,
            model_dim=hidden,
            activation=petit_kernel.MegaMoeActivation.mxfp4,
            activation_function=petit_kernel.MegaMoeActivationFunction.silu,
            stages=petit_kernel.MegaMoeStages.two_stage,
            inter_dim=inter_dim,
            has_bias=False,
        )
        heap = petit_kernel.create_vmm_symmetric_heap(world_size)
        views = config.input_views(heap, max_tokens)
        assert views.scales is not None

        w13 = torch.zeros(
            (local_experts, 2 * inter_dim, compute_hidden // 2),
            dtype=torch.uint8,
            device=device,
        )
        w2 = torch.zeros(
            (local_experts, compute_hidden, inter_dim // 2),
            dtype=torch.uint8,
            device=device,
        )
        s13 = torch.full(
            (local_experts, 2 * inter_dim, compute_hidden // 32),
            127,
            dtype=torch.uint8,
            device=device,
        )
        s2 = torch.full(
            (local_experts, compute_hidden, inter_dim // 32),
            127,
            dtype=torch.uint8,
            device=device,
        )

        graphs: list[torch.cuda.CUDAGraph] = []
        graph_inputs: list[tuple[torch.Tensor, ...]] = []
        graph_outputs: list[list[torch.Tensor]] = []
        for tokens in capture_sizes:
            token = torch.arange(tokens, dtype=torch.int32, device=device)[:, None]
            route = torch.arange(topk, dtype=torch.int32, device=device)[None, :]
            destination = (rank + token + route) % world_size
            local_expert = (token * topk + route) % local_experts
            expert_ids = (destination * local_experts + local_expert).contiguous()
            if args.skewed:
                expert_ids = torch.arange(
                    topk, dtype=torch.int32, device=device
                ).expand(tokens, -1).contiguous()
            expert_weights = torch.full(
                (tokens, topk),
                1.0 / topk,
                dtype=torch.float32,
                device=device,
            )
            hidden_states = torch.zeros(
                (tokens, hidden), dtype=torch.bfloat16, device=device
            )
            values = torch.zeros(
                (tokens, hidden // 2), dtype=torch.uint8, device=device
            )
            scales = torch.full(
                (tokens, hidden // 32), 127, dtype=torch.uint8, device=device
            )
            outputs = [
                torch.empty(
                    (tokens, compute_hidden), dtype=torch.bfloat16, device=device
                )
                for _ in range(args.layers)
            ]

            for layer in range(args.layers):
                if args.quantize:
                    values, scales = config.quantize(hidden_states)
                views.tokens[:tokens].copy_(values)
                views.scales[:tokens].copy_(scales)
                views.expert_ids[:tokens].copy_(expert_ids)
                views.expert_weights[:tokens].copy_(expert_weights)
                config.run(heap, w13, w2, s13, s2, tokens, out=outputs[layer])
            torch.cuda.synchronize(device)
            dist.barrier()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for layer in range(args.layers):
                    if args.quantize:
                        values, scales = config.quantize(hidden_states)
                    views.tokens[:tokens].copy_(values)
                    views.scales[:tokens].copy_(scales)
                    views.expert_ids[:tokens].copy_(expert_ids)
                    views.expert_weights[:tokens].copy_(expert_weights)
                    config.run(heap, w13, w2, s13, s2, tokens, out=outputs[layer])
            torch.cuda.synchronize(device)
            dist.barrier()
            if rank == 0:
                print(f"captured {tokens}", flush=True)
            graphs.append(graph)
            graph_inputs.append(
                (hidden_states, values, scales, expert_ids, expert_weights)
            )
            graph_outputs.append(outputs)

        for tokens, graph, outputs in zip(capture_sizes, graphs, graph_outputs):
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(device)
            for output in outputs:
                if not torch.isfinite(output).all():
                    raise AssertionError(f"non-finite output for {tokens} tokens")
            if rank == 0:
                print(f"replayed {tokens}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
