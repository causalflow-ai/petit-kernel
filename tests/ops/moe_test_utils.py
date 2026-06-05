from __future__ import annotations

import torch


def build_padded_sorted_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the expert-major routing metadata consumed by fused MoE kernels."""
    if topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2:
        raise ValueError(
            "top-k IDs and weights must be rank-2 tensors of equal shape"
        )

    tokens, topk = topk_ids.shape
    max_routes = topk_ids.numel() + num_experts * block_size - topk
    max_blocks = (max_routes + block_size - 1) // block_size
    sentinel = (topk << 24) | tokens

    sorted_token_ids = torch.full((max_routes,), sentinel, dtype=torch.int32)
    sorted_weights = torch.zeros((max_routes,), dtype=torch.float32)
    sorted_expert_ids = torch.full((max_blocks,), -1, dtype=torch.int32)

    expert_ids = topk_ids.to("cpu", dtype=torch.int32).reshape(-1)
    route_weights = topk_weights.to("cpu", dtype=torch.float32).reshape(-1)
    token_ids = (
        torch.arange(tokens, dtype=torch.int32)
        .unsqueeze(1)
        .expand(tokens, topk)
        .reshape(-1)
    )
    slot_ids = (
        torch.arange(topk, dtype=torch.int32)
        .unsqueeze(0)
        .expand(tokens, topk)
        .reshape(-1)
    )

    order = torch.argsort(expert_ids, stable=True)
    expert_ids = expert_ids[order]
    route_weights = route_weights[order]
    token_ids = token_ids[order]
    slot_ids = slot_ids[order]

    route_begin = 0
    output_begin = 0
    block_begin = 0
    expert_counts = torch.bincount(expert_ids, minlength=num_experts)
    for expert, count in enumerate(expert_counts):
        route_count = int(count)
        if route_count == 0:
            continue

        route_end = route_begin + route_count
        output_end = output_begin + route_count
        sorted_token_ids[output_begin:output_end] = (
            slot_ids[route_begin:route_end] << 24
        ) | token_ids[route_begin:route_end]
        sorted_weights[output_begin:output_end] = route_weights[
            route_begin:route_end
        ]

        expert_blocks = (route_count + block_size - 1) // block_size
        sorted_expert_ids[block_begin : block_begin + expert_blocks] = expert
        route_begin = route_end
        output_begin += expert_blocks * block_size
        block_begin += expert_blocks

    num_valid_ids = torch.tensor([output_begin, tokens], dtype=torch.int32)
    device = topk_ids.device
    return (
        sorted_token_ids.to(device),
        sorted_weights.to(device),
        sorted_expert_ids.to(device),
        num_valid_ids.to(device),
    )
