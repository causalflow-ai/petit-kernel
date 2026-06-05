import glob
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import petit_kernel
from fused_moe_test_data import NUMERICALLY_SENSITIVE_ATOL
from fused_moe_test_data import numerically_sensitive_topk
from moe_test_utils import build_padded_sorted_routing


BLOCK_K = 128
W13_FP4_ZERO_PROB = 0.119
W13_FP4_MAG_MEAN = 3.10
W13_FP4_MAG_STD = 1.76
W2_FP4_ZERO_PROB = 0.299
W2_FP4_MAG_MEAN = 2.58
W2_FP4_MAG_STD = 1.77
FC1_E8M0_MEAN = 117.0
FC1_E8M0_STD = 1.45
FC2_E8M0_MEAN = 122.0
FC2_E8M0_STD = 1.15


def _dequantize_input(
    input_q: torch.Tensor, input_scale: torch.Tensor
) -> torch.Tensor:
    tokens, model_dim = input_q.shape
    blocks = input_q.to(torch.float32).view(tokens, model_dim // BLOCK_K, BLOCK_K)
    return (blocks * input_scale.to(torch.float32).unsqueeze(-1)).reshape(
        tokens, model_dim
    )


def _quantize_input(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, model_dim = x.shape
    blocks = x.view(tokens, model_dim // BLOCK_K, BLOCK_K)
    absmax = blocks.abs().amax(dim=-1).clamp_min(1e-6)
    q_scale = 240.0 / absmax
    input_scale = absmax / 240.0
    input_q = (
        (blocks * q_scale.unsqueeze(-1))
        .to(_native_fp8_dtype_or_skip(x.device))
        .view(tokens, model_dim)
    )
    return input_q.contiguous(), input_scale.contiguous()


def _native_fp8_dtype_or_skip(device: torch.device) -> torch.dtype:
    if not (
        hasattr(torch, "float8_e4m3fn")
        or hasattr(torch, "float8_e4m3fnuz")
    ):
        pytest.skip("torch float8_e4m3 dtype is required")
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    if arch.startswith(("gfx950", "gfx1200", "gfx1201")) and hasattr(
        torch, "float8_e4m3fn"
    ):
        return torch.float8_e4m3fn
    if hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _require_fp8_e4m3() -> None:
    _native_fp8_dtype_or_skip(torch.device("cuda"))


def _rand_moment_mxfp4_tensor(
    shape: tuple[int, ...],
    device: torch.device,
    *,
    zero_prob: float,
    magnitude_mean: float,
    magnitude_std: float,
) -> torch.Tensor:
    def sample_nibbles() -> torch.Tensor:
        zero = torch.rand(shape, device=device) < zero_prob
        magnitude = (magnitude_mean + magnitude_std * torch.randn(
            shape, device=device
        )).round().clamp_(1, 7).to(torch.uint8)
        sign = torch.randint(0, 2, shape, device=device, dtype=torch.uint8) << 3
        return torch.where(zero, torch.zeros_like(magnitude), magnitude | sign)

    lo = sample_nibbles()
    hi = sample_nibbles()
    return ((hi << 4) | lo).contiguous()


def _rand_moment_e8m0_scales(
    shape: tuple[int, ...], device: torch.device, *, mean: float, stddev: float
) -> torch.Tensor:
    return (mean + stddev * torch.randn(shape, device=device)).round().clamp_(
        1, 237
    ).to(torch.uint8).contiguous()

def _dequantize_mxfp4(
    qweight_u8: torch.Tensor, scales_e8m0: torch.Tensor
) -> torch.Tensor:
    codebook = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=qweight_u8.device,
    )
    nibbles = torch.stack(
        (qweight_u8 & 0xF, qweight_u8 >> 4), dim=-1
    ).flatten(-2)
    values = codebook[nibbles.to(torch.long)]
    scales = (scales_e8m0.to(torch.int32) << 23).view(torch.float32)
    return (values * scales.repeat_interleave(32, dim=-1)).transpose(0, 1)


def _reference_fused_moe_chunked(
    input_q: torch.Tensor,
    w13_q: torch.Tensor,
    w2_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    input_scale: torch.Tensor,
    fc1_scale: torch.Tensor,
    fc2_scale: torch.Tensor,
) -> torch.Tensor:
    tokens, model_dim = input_q.shape
    experts, _, packed_inter = w2_q.shape
    inter_dim = packed_inter * 2

    x = _dequantize_input(input_q, input_scale)
    out = torch.zeros(
        (tokens, model_dim), dtype=torch.float32, device=input_q.device
    )

    for expert in range(experts):
        expert_mask = topk_ids == expert
        if not torch.any(expert_mask):
            continue

        token_ids, route_ids = torch.where(expert_mask)
        w13 = _dequantize_mxfp4(w13_q[expert], fc1_scale[expert])
        w2 = _dequantize_mxfp4(w2_q[expert], fc2_scale[expert])

        for idx in range(token_ids.numel()):
            token = int(token_ids[idx].item())
            route = int(route_ids[idx].item())

            stage1_acc = torch.zeros(
                (2 * inter_dim,), dtype=torch.float32, device=x.device
            )
            for k256 in range(0, model_dim, 256):
                x_256 = x[token, k256 : k256 + 256].to(torch.float32)
                w13_256 = w13[k256 : k256 + 256, :].to(torch.float32)
                stage1_acc += x_256 @ w13_256

            gate = stage1_acc[:inter_dim].unsqueeze(0)
            up = stage1_acc[inter_dim:].unsqueeze(0)
            activated = F.silu(gate) * up
            route_out = activated @ w2
            weighted = route_out * topk_weights[token, route].to(torch.float32)
            out[token] += weighted.squeeze(0)

    return out.to(torch.bfloat16)


def _prepare_dump_case(
    obj: dict,
) -> tuple[torch.Tensor, ...]:
    reference = obj["reference"]

    topk_ids = reference["topk_ids"].to(torch.int32).contiguous()
    topk_weights = reference["topk_weights"].to(torch.float32).contiguous()

    hidden_states = reference.get("hidden_states")
    if hidden_states is None:
        pytest.skip("dump missing reference hidden_states; cannot regenerate inputs")
    input_q_fp8, input_scale = _quantize_input(hidden_states.to(torch.float32).contiguous())
    input_q_u8 = input_q_fp8.view(torch.uint8).contiguous()
    input_scale_layout = input_scale.transpose(0, 1).contiguous()

    w13_q_packed, fc1_scale_packed = petit_kernel.repack_moe_kernel_layout(
        reference["w13_q"].contiguous(),
        reference["fc1_scale"].contiguous(),
        layout=petit_kernel.MoeKernelLayout.petit_mxfp4,
    )
    w2_q_packed, fc2_scale_packed = petit_kernel.repack_moe_kernel_layout(
        reference["w2_q"].contiguous(),
        reference["fc2_scale"].contiguous(),
        layout=petit_kernel.MoeKernelLayout.petit_mxfp4,
    )
    return (
        input_q_u8,
        input_scale_layout,
        topk_ids,
        topk_weights,
        w13_q_packed,
        fc1_scale_packed,
        w2_q_packed,
        fc2_scale_packed,
    )


def _collect_dump_paths() -> list[Path]:
    pattern_env = os.environ.get(
        "PETIT_MOE_DUMP_GLOB", "build/petit_moe_smoke_v2.layer3.pid194525.pt"
    )
    max_cases = int(os.environ.get("PETIT_MOE_DUMP_MAX_CASES", "1"))

    found: list[Path] = []
    for pattern in pattern_env.split(os.pathsep):
        pattern = pattern.strip()
        if not pattern:
            continue
        matches = sorted(glob.glob(pattern))
        if matches:
            found.extend(Path(match).resolve() for match in matches)
        else:
            path = Path(pattern)
            if path.exists():
                found.append(path.resolve())

    if max_cases > 0:
        found = found[:max_cases]
    return found


def _run_fused_moe_case(
    input_q: torch.Tensor,
    input_scale: torch.Tensor,
    w13_q: torch.Tensor,
    w2_q: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_scale: torch.Tensor,
    fc2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    w13_q_packed, fc1_scale_packed = petit_kernel.repack_moe_kernel_layout(
        w13_q,
        fc1_scale,
        layout=petit_kernel.MoeKernelLayout.petit_mxfp4,
    )
    w2_q_packed, fc2_scale_packed = petit_kernel.repack_moe_kernel_layout(
        w2_q,
        fc2_scale,
        layout=petit_kernel.MoeKernelLayout.petit_mxfp4,
    )
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
        build_padded_sorted_routing(
            topk_ids, topk_weights, w2_q_packed.size(0)
        )
    )
    out = torch.zeros(
        (input_q.size(0), input_q.size(1)),
        dtype=torch.bfloat16,
        device=input_q.device,
    )
    petit_kernel.fused_moe_fp8_blockscale_g1u1_mxfp4(
        input_q,
        w13_q_packed,
        w2_q_packed,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        int(topk_ids.size(1)),
        input_scale.transpose(0, 1).contiguous(),
        fc1_scale_packed,
        fc2_scale_packed,
        out=out,
    )
    ref = _reference_fused_moe_chunked(
        input_q,
        w13_q,
        w2_q,
        topk_weights,
        topk_ids,
        input_scale,
        fc1_scale,
        fc2_scale,
    )
    return out, ref


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA device required")
def test_fused_moe_blockscale_fp8_mxfp4_matches_reference() -> None:
    _require_fp8_e4m3()

    torch.manual_seed(1234)
    device = torch.device("cuda")

    tokens = 2
    model_dim = 7168
    inter_dim = 2048
    experts = 32
    topk = 8

    input_f = (
        torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
        * 0.12
    )
    spike_mask = torch.rand((tokens, model_dim), device=device) < 0.002
    input_f += spike_mask * torch.randn(
        (tokens, model_dim), device=device
    ) * 5.0
    input_q, input_scale = _quantize_input(input_f)

    w13_q = _rand_moment_mxfp4_tensor(
        (experts, 2 * inter_dim, model_dim // 2),
        device,
        zero_prob=W13_FP4_ZERO_PROB,
        magnitude_mean=W13_FP4_MAG_MEAN,
        magnitude_std=W13_FP4_MAG_STD,
    )
    w2_q = _rand_moment_mxfp4_tensor(
        (experts, model_dim, inter_dim // 2),
        device,
        zero_prob=W2_FP4_ZERO_PROB,
        magnitude_mean=W2_FP4_MAG_MEAN,
        magnitude_std=W2_FP4_MAG_STD,
    )
    fc1_scale = _rand_moment_e8m0_scales(
        (experts, 2 * inter_dim, model_dim // 32),
        device,
        mean=FC1_E8M0_MEAN,
        stddev=FC1_E8M0_STD,
    )
    fc2_scale = _rand_moment_e8m0_scales(
        (experts, model_dim, inter_dim // 32),
        device,
        mean=FC2_E8M0_MEAN,
        stddev=FC2_E8M0_STD,
    )
    topk_ids, topk_weights = numerically_sensitive_topk(device)

    out, ref = _run_fused_moe_case(
        input_q,
        input_scale,
        w13_q,
        w2_q,
        topk_ids,
        topk_weights,
        fc1_scale,
        fc2_scale,
    )

    assert torch.isfinite(out.float()).all()
    assert out.abs().max().item() > 0
    torch.testing.assert_close(
        out, ref, rtol=0.0, atol=NUMERICALLY_SENSITIVE_ATOL
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA device required")
def test_fused_moe_blockscale_fp8_mxfp4_dump_replay_matches_reference() -> None:
    _require_fp8_e4m3()

    dump_paths = _collect_dump_paths()
    if not dump_paths:
        pytest.skip(
            "no dump matched PETIT_MOE_DUMP_GLOB "
            "(default: build/petit_moe_smoke_v2.layer3.pid194525.pt)"
        )

    atol = float(os.environ.get("PETIT_MOE_DUMP_ATOL", "2.5e-2"))
    rtol = float(os.environ.get("PETIT_MOE_DUMP_RTOL", "0.0"))
    fp8_dtype = _native_fp8_dtype_or_skip(torch.device("cuda"))

    for dump_path in dump_paths:
        obj = torch.load(dump_path, map_location="cuda")
        reference = obj["reference"]

        (
            input_q_u8,
            input_scale_layout,
            topk_ids,
            topk_weights,
            w13_q_packed,
            fc1_scale_packed,
            w2_q_packed,
            fc2_scale_packed,
        ) = _prepare_dump_case(obj)
        moe_sorting = pytest.importorskip("aiter.fused_moe").moe_sorting
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
            topk_ids,
            topk_weights,
            w2_q_packed.size(0),
            input_q_u8.size(1),
            torch.bfloat16,
            32,
        )
        out = torch.zeros(
            (input_q_u8.size(0), input_q_u8.size(1)),
            dtype=torch.bfloat16,
            device=input_q_u8.device,
        )
        petit_kernel.fused_moe_fp8_blockscale_g1u1_mxfp4(
            input_q_u8,
            w13_q_packed.contiguous(),
            w2_q_packed.contiguous(),
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            int(topk_ids.size(1)),
            input_scale_layout.contiguous(),
            fc1_scale_packed.contiguous(),
            fc2_scale_packed.contiguous(),
            out=out,
        )
        torch.cuda.synchronize()
        ref = _reference_fused_moe_chunked(
            input_q_u8.view(fp8_dtype).contiguous(),
            reference["w13_q"].contiguous(),
            reference["w2_q"].contiguous(),
            topk_weights,
            topk_ids,
            input_scale_layout.transpose(0, 1).contiguous(),
            reference["fc1_scale"].contiguous(),
            reference["fc2_scale"].contiguous(),
        )

        diff = (out.float() - ref.float()).abs()
        max_diff = float(diff.max().item())
        max_idx = int(diff.argmax().item())
        token_idx = max_idx // out.shape[1]
        col_idx = max_idx % out.shape[1]
        with torch.no_grad():
            allclose = torch.all(diff <= (atol + rtol * ref.float().abs())).item()
        assert allclose, (
            f"dump replay mismatch: path={dump_path} "
            f"max_abs_diff={max_diff} token={token_idx} col={col_idx} "
            f"out={float(out[token_idx, col_idx].float().item())} "
            f"ref={float(ref[token_idx, col_idx].float().item())} "
            f"atol={atol} rtol={rtol}"
        )
