from pathlib import Path

import pytest
import torch

import petit_kernel


def _dequant_nvfp4(qweights_u8: torch.Tensor, scales_e4m3: torch.Tensor) -> torch.Tensor:
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=qweights_u8.device,
        dtype=torch.float32,
    )
    lo = qweights_u8 & 0x0F
    hi = qweights_u8 >> 4
    dequant = torch.empty((qweights_u8.size(0), qweights_u8.size(1) * 2), device=qweights_u8.device, dtype=torch.float32)
    dequant[:, 0::2] = fp4_values[lo.long()]
    dequant[:, 1::2] = fp4_values[hi.long()]
    return (dequant.view(qweights_u8.size(0), -1, 16) * scales_e4m3.float().unsqueeze(-1)).view(qweights_u8.size(0), -1)


def _gemm_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() @ b.t().float()).to(a.dtype)


NVFP4_CASES = [
    (64, 128, 256, 1234),
    (96, 64, 512, 2026),
]

MXFP4_CASES = [
    (64, 128, 256, 1234),
    (96, 96, 512, 2026),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA device required")
@pytest.mark.parametrize(("m", "n", "k", "seed"), NVFP4_CASES)
def test_nvfp4_gemm_matches_reference(m: int, n: int, k: int, seed: int) -> None:
    torch.manual_seed(seed)

    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    qweights_u8 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scales_e4m3 = (torch.rand((n, k // 16), device="cuda") * 3.5 + 0.25).to(torch.float8_e4m3fn)
    global_scale = torch.rand((1,), dtype=torch.float32, device="cuda") * 1.5 + 0.5

    b_packed_u32 = petit_kernel.repack_nvfp4(qweights_u8.contiguous().view(torch.int32), n, k)
    s_processed = petit_kernel.process_nvfp4_scales(scales_e4m3, n, k)
    c_hip = petit_kernel.mul_nvfp4_a16(a, b_packed_u32, s_processed, global_scale, m, n, k, -1)

    b_ref = _dequant_nvfp4(qweights_u8, scales_e4m3) * global_scale.item()
    c_ref = _gemm_ref(a, b_ref)
    torch.testing.assert_close(c_hip, c_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA device required")
@pytest.mark.parametrize(("m", "n", "k", "seed"), MXFP4_CASES)
def test_mxfp4_gemm_matches_quark_reference(
    monkeypatch: pytest.MonkeyPatch,
    m: int,
    n: int,
    k: int,
    seed: int,
) -> None:
    monkeypatch.setenv("QUARK_MXFP4_IMPL", "hip")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(Path("build/torch_extensions").resolve()))
    pytest.importorskip("quark")
    from quark.torch.kernel.mx import dq_mxfp4

    torch.manual_seed(seed)

    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    qweights_u8 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scales_e8m0 = torch.randint(1, 238, (n, k // 32), dtype=torch.uint8, device="cuda")
    global_scale = torch.rand((1,), dtype=torch.float32, device="cuda") * 1.5 + 0.5

    b_packed_u32 = petit_kernel.repack_mxfp4(qweights_u8.contiguous().view(torch.int32), n, k)
    s_processed = petit_kernel.process_mxfp4_scales(scales_e8m0, n, k)
    c_hip = petit_kernel.mul_mxfp4_a16(a, b_packed_u32, s_processed, global_scale, m, n, k, -1)

    try:
        b_dequant = dq_mxfp4(qweights_u8, scales_e8m0, torch.bfloat16)
    except Exception as exc:
        pytest.skip(f"quark mxfp4 kernel unavailable: {exc}")

    c_ref = ((a.float() @ b_dequant.t().float()) * global_scale.item()).to(a.dtype)
    torch.testing.assert_close(c_hip, c_ref, rtol=2e-2, atol=2e-2)
