import glob
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import petit_kernel
from fused_moe_replay_like import SENSITIVE_ATOL as REPLAY_LIKE_SENSITIVE_ATOL
from fused_moe_replay_like import topk_tensors as replay_like_topk_tensors


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


def _dequantize_input(input_q: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    tokens, model_dim = input_q.shape
    blocks = input_q.to(torch.float32).view(tokens, model_dim // BLOCK_K, BLOCK_K)
    return (blocks * input_scale.to(torch.float32).unsqueeze(-1)).reshape(tokens, model_dim)


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
    if not (hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")):
        pytest.skip("torch float8_e4m3 dtype is required")
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    if arch.startswith(("gfx950", "gfx1200", "gfx1201")) and hasattr(torch, "float8_e4m3fn"):
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
        magnitude = (
            magnitude_mean + magnitude_std * torch.randn(shape, device=device)
        ).round().clamp_(1, 7).to(torch.uint8)
        sign = torch.randint(0, 2, shape, device=device, dtype=torch.uint8) << 3
        return torch.where(zero, torch.zeros_like(magnitude), magnitude | sign)

    lo = sample_nibbles()
    hi = sample_nibbles()
    return ((hi << 4) | lo).contiguous()


def _rand_moment_e8m0_scales(
    shape: tuple[int, ...], device: torch.device, *, mean: float, stddev: float
) -> torch.Tensor:
    return (
        mean + stddev * torch.randn(shape, device=device)
    ).round().clamp_(1, 237).to(torch.uint8).contiguous()



def _dequantize_mxfp4_quark(qweight_u8: torch.Tensor, scales_e8m0: torch.Tensor) -> torch.Tensor:
    try:
        from quark.torch.kernel.mx.triton import dq_mxfp4_triton
    except ImportError as exc:
        pytest.skip(f"quark Triton MXFP4 dequant is unavailable: {exc}")

    try:
        return (
            dq_mxfp4_triton(
                qweight_u8.contiguous(),
                scales_e8m0.contiguous(),
                torch.bfloat16,
            )
            .transpose(0, 1)
            .contiguous()
            .to(torch.float32)
        )
    except Exception as exc:
        pytest.skip(f"quark Triton MXFP4 dequant failed: {exc}")


def _reference_fused_moe_quark_chunked(
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
    out = torch.zeros((tokens, model_dim), dtype=torch.float32, device=input_q.device)

    for expert in range(experts):
        expert_mask = topk_ids == expert
        if not torch.any(expert_mask):
            continue

        token_ids, route_ids = torch.where(expert_mask)
        w13 = _dequantize_mxfp4_quark(w13_q[expert], fc1_scale[expert])
        w2 = _dequantize_mxfp4_quark(w2_q[expert], fc2_scale[expert])

        for idx in range(token_ids.numel()):
            token = int(token_ids[idx].item())
            route = int(route_ids[idx].item())

            stage1_acc = torch.zeros((2 * inter_dim,), dtype=torch.float32, device=x.device)
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
    experts = int(reference["w2_q"].size(0))

    hidden_states = reference.get("hidden_states")
    if hidden_states is None:
        pytest.skip("dump missing reference hidden_states; cannot regenerate inputs")
    input_q_fp8, input_scale = _quantize_input(hidden_states.to(torch.float32).contiguous())
    input_q_u8 = input_q_fp8.view(torch.uint8).contiguous()
    input_scale_layout = input_scale.transpose(0, 1).contiguous()

    model_dim = int(input_q_u8.size(1))
    moe_sorting = pytest.importorskip("aiter.fused_moe").moe_sorting
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        model_dim,
        torch.bfloat16,
    )

    w13_q_packed, fc1_scale_packed = petit_kernel.repack_moe_mxfp4_kernel_layout(
        reference["w13_q"].contiguous(),
        reference["fc1_scale"].contiguous(),
    )
    w2_q_packed, fc2_scale_packed = petit_kernel.repack_moe_mxfp4_kernel_layout(
        reference["w2_q"].contiguous(),
        reference["fc2_scale"].contiguous(),
    )
    return (
        input_q_u8,
        input_scale_layout,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
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


def _run_petit_moe_kernel(
    input_q: torch.Tensor,
    w13_q_packed: torch.Tensor,
    w2_q_packed: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    input_scale_layout: torch.Tensor,
    fc1_scale_packed: torch.Tensor,
    fc2_scale_packed: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return petit_kernel.fused_moe_fp8_blockscale_g1u1_mxfp4(
        input_q,
        w13_q_packed,
        w2_q_packed,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale_layout,
        fc1_scale_packed,
        fc2_scale_packed,
        out=out,
    )


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
    experts = int(w2_q.size(0))
    topk = int(topk_ids.size(1))
    input_scale_layout = input_scale.transpose(0, 1).contiguous()
    moe_sorting = pytest.importorskip("aiter.fused_moe").moe_sorting

    w13_q_packed, fc1_scale_packed = petit_kernel.repack_moe_mxfp4_kernel_layout(
        w13_q, fc1_scale
    )
    w2_q_packed, fc2_scale_packed = petit_kernel.repack_moe_mxfp4_kernel_layout(
        w2_q, fc2_scale
    )
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, input_q.size(1), torch.bfloat16
    )

    out = _run_petit_moe_kernel(
        input_q,
        w13_q_packed,
        w2_q_packed,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale_layout,
        fc1_scale_packed,
        fc2_scale_packed,
    )
    ref = _reference_fused_moe_quark_chunked(
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
def test_fused_moe_blockscale_fp8_mxfp4_reuses_output_buffer() -> None:
    _require_fp8_e4m3()

    torch.manual_seed(20260430)
    device = torch.device("cuda")

    tokens = 5
    model_dim = 256
    inter_dim = 256
    experts = 4
    topk = 2

    input_f = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * 0.12
    input_q, input_scale = _quantize_input(input_f)
    input_scale_layout = input_scale.transpose(0, 1).contiguous()

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
    w13_q_packed, fc1_scale_packed = petit_kernel.repack_moe_mxfp4_kernel_layout(
        w13_q, fc1_scale
    )
    w2_q_packed, fc2_scale_packed = petit_kernel.repack_moe_mxfp4_kernel_layout(
        w2_q, fc2_scale
    )

    topk_ids = torch.randint(0, experts, (tokens, topk), device=device, dtype=torch.int32)
    topk_weights = torch.rand((tokens, topk), device=device, dtype=torch.float32)
    moe_sorting = pytest.importorskip("aiter.fused_moe").moe_sorting
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, torch.bfloat16
    )

    expected = _run_petit_moe_kernel(
        input_q,
        w13_q_packed,
        w2_q_packed,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale_layout,
        fc1_scale_packed,
        fc2_scale_packed,
    )
    out_buffer = torch.empty_like(expected)
    actual = _run_petit_moe_kernel(
        input_q,
        w13_q_packed,
        w2_q_packed,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale_layout,
        fc1_scale_packed,
        fc2_scale_packed,
        out=out_buffer,
    )

    assert actual.data_ptr() == out_buffer.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA device required")
def test_fused_moe_blockscale_fp8_mxfp4_skips_invalid_expert_group() -> None:
    _require_fp8_e4m3()

    torch.manual_seed(20260502)
    device = torch.device("cuda")

    tokens = 4
    model_dim = 256
    inter_dim = 256
    experts = 2
    topk = 1
    block_size = 32

    input_f = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * 0.1
    input_q, input_scale = _quantize_input(input_f)
    input_scale_layout = input_scale.transpose(0, 1).contiguous()

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
    w13_q_packed, fc1_scale_packed = petit_kernel.repack_moe_mxfp4_kernel_layout(
        w13_q, fc1_scale
    )
    w2_q_packed, fc2_scale_packed = petit_kernel.repack_moe_mxfp4_kernel_layout(
        w2_q, fc2_scale
    )

    sorted_token_ids = torch.zeros((block_size,), device=device, dtype=torch.int32)
    sorted_token_ids[:tokens] = torch.arange(tokens, device=device, dtype=torch.int32)
    sorted_weights = torch.ones((block_size,), device=device, dtype=torch.float32)
    sorted_expert_ids = torch.tensor([-1], device=device, dtype=torch.int32)
    num_valid_ids = torch.tensor([block_size, tokens], device=device, dtype=torch.int32)

    out = _run_petit_moe_kernel(
        input_q,
        w13_q_packed,
        w2_q_packed,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale_layout,
        fc1_scale_packed,
        fc2_scale_packed,
    )

    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0.0, atol=0.0)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA device required")
def test_fused_moe_blockscale_fp8_mxfp4_cuda_graph_replay_updates_output() -> None:
    _require_fp8_e4m3()

    torch.manual_seed(20260501)
    device = torch.device("cuda")

    tokens = 4
    model_dim = 256
    inter_dim = 256
    experts = 4
    topk = 2

    input_f_a = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * 0.12
    input_f_b = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * 0.12
    input_q_a, input_scale_a = _quantize_input(input_f_a)
    input_q_b, input_scale_b = _quantize_input(input_f_b)
    input_scale_layout_a = input_scale_a.transpose(0, 1).contiguous()
    input_scale_layout_b = input_scale_b.transpose(0, 1).contiguous()

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
    w13_q_packed, fc1_scale_packed = petit_kernel.repack_moe_mxfp4_kernel_layout(
        w13_q, fc1_scale
    )
    w2_q_packed, fc2_scale_packed = petit_kernel.repack_moe_mxfp4_kernel_layout(
        w2_q, fc2_scale
    )

    topk_ids = torch.randint(0, experts, (tokens, topk), device=device, dtype=torch.int32)
    topk_weights = torch.rand((tokens, topk), device=device, dtype=torch.float32)
    moe_sorting = pytest.importorskip("aiter.fused_moe").moe_sorting
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, torch.bfloat16
    )

    expected_a = _run_petit_moe_kernel(
        input_q_a,
        w13_q_packed,
        w2_q_packed,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale_layout_a,
        fc1_scale_packed,
        fc2_scale_packed,
    )
    expected_b = _run_petit_moe_kernel(
        input_q_b,
        w13_q_packed,
        w2_q_packed,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        input_scale_layout_b,
        fc1_scale_packed,
        fc2_scale_packed,
    )

    static_input_q = input_q_a.clone()
    static_input_scale_layout = input_scale_layout_a.clone()
    out_buffer = torch.empty_like(expected_a)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _run_petit_moe_kernel(
            static_input_q,
            w13_q_packed,
            w2_q_packed,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            static_input_scale_layout,
            fc1_scale_packed,
            fc2_scale_packed,
            out=out_buffer,
        )
    assert actual.data_ptr() == out_buffer.data_ptr()

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_buffer, expected_a, rtol=0.0, atol=0.0)

    static_input_q.copy_(input_q_b)
    static_input_scale_layout.copy_(input_scale_layout_b)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_buffer, expected_b, rtol=0.0, atol=0.0)


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

    input_f = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * 0.12
    spike_mask = torch.rand((tokens, model_dim), device=device) < 0.002
    input_f = input_f + spike_mask * torch.randn((tokens, model_dim), device=device) * 5.0
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
    topk_ids, topk_weights = replay_like_topk_tensors(device)

    out, ref = _run_fused_moe_case(
        input_q, input_scale, w13_q, w2_q, topk_ids, topk_weights, fc1_scale, fc2_scale
    )

    assert torch.isfinite(out.float()).all()
    assert out.abs().max().item() > 0
    torch.testing.assert_close(out, ref, rtol=0.0, atol=REPLAY_LIKE_SENSITIVE_ATOL)


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
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            w13_q_packed,
            fc1_scale_packed,
            w2_q_packed,
            fc2_scale_packed,
        ) = _prepare_dump_case(obj)
        topk = int(topk_ids.size(1))

        out = _run_petit_moe_kernel(
            input_q_u8,
            w13_q_packed.contiguous(),
            w2_q_packed.contiguous(),
            sorted_token_ids.contiguous(),
            sorted_weights.contiguous(),
            sorted_expert_ids.contiguous(),
            num_valid_ids.contiguous(),
            topk,
            input_scale_layout.contiguous(),
            fc1_scale_packed.contiguous(),
            fc2_scale_packed.contiguous(),
        )
        torch.cuda.synchronize()
        ref = _reference_fused_moe_quark_chunked(
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
