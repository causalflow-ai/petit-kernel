import torch

# Replay-like cases are tuned to produce non-trivial stage-2 route outputs and
# catch W2 layout/advancement regressions behaviorally.
SENSITIVE_ATOL = 2.05e-2
TOPK_IDS = (
    (9, 18, 29, 30, 25, 4, 11, 3),
    (15, 12, 20, 23, 31, 7, 6, 21),
)


def topk_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = torch.tensor(TOPK_IDS, device=device, dtype=torch.int32)
    token_idx = torch.arange(
        topk_ids.size(0), dtype=torch.float32, device=device
    ).unsqueeze(1)
    slot_idx = torch.arange(
        topk_ids.size(1), dtype=torch.float32, device=device
    ).unsqueeze(0)
    topk_weights = 2.0 * (0.25 + 0.025 * ((token_idx + 3 * slot_idx) % 9))
    return topk_ids, topk_weights
