from __future__ import annotations

import torch


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


_PETIT_SIGN_OFFSETS = (7, 15, 23, 31, 24, 16, 8, 0)
_PETIT_VALUE_OFFSETS = (1, 9, 17, 25, 28, 20, 12, 4)


def _bitreverse3(v: torch.Tensor) -> torch.Tensor:
    return ((v & 0x1) << 2) | (v & 0x2) | ((v & 0x4) >> 2)

def _petit_format_words(words: torch.Tensor) -> torch.Tensor:
    _require(words.dtype == torch.int32, "words must be int32")

    out = torch.zeros_like(words)
    for i in range(8):
        sgn_off = _PETIT_SIGN_OFFSETS[i]
        val_off = _PETIT_VALUE_OFFSETS[i]

        u = (words >> (i * 4)) & 0xF
        val = u & 0x7
        sgn = (u >> 3) * (val != 0).to(torch.int32)
        if i >= 4:
            # Petit reshuffles each 8-element group by bit-reversing the
            # 3-bit magnitude in the upper 4 lanes.
            val = _bitreverse3(val)
        out |= sgn << sgn_off
        out |= val << val_off
    return out


def _pack_mxfp4_weight_kernel_layout(qweight_u8: torch.Tensor) -> torch.Tensor:
    _require(qweight_u8.dtype == torch.uint8, "qweight_u8 must be uint8")
    _require(qweight_u8.ndim == 2, "qweight_u8 must be rank-2")
    _require(qweight_u8.is_contiguous(), "qweight_u8 must be contiguous")

    size_n = qweight_u8.size(0)
    size_k = qweight_u8.size(1) * 2
    _require(size_n % 64 == 0, "size_n must be divisible by 64")
    _require(size_k % 64 == 0, "size_k must be divisible by 64")
    _require(size_k % 128 == 0, "size_k must be divisible by 128")

    words = qweight_u8.view(torch.int32).reshape(
        size_n // 16,
        16,
        size_k // 128,
        4,
        4,
    )
    words = _petit_format_words(words).permute(0, 2, 4, 1, 3)
    return words.contiguous().view(torch.uint8).reshape(size_n, size_k // 2)


# 4x64x128 strided scales with 64 columns, K-major
def _pack_mxfp4_scale_kernel_layout(scales_e8m0: torch.Tensor) -> torch.Tensor:
    _require(scales_e8m0.dtype == torch.uint8, "scales_e8m0 must be uint8")
    size_n = scales_e8m0.size(0)

    words = scales_e8m0.reshape(
        size_n // 256,
        4,
        4,
        4,
        4,
        scales_e8m0.size(1) // 4,
        4,
    )
    return words.permute(0, 5, 2, 3, 1, 6, 4).contiguous()


def repack_moe_mxfp4_kernel_layout(
    qweight_u8: torch.Tensor,
    scales_e8m0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if qweight_u8.ndim == 2:
        _require(scales_e8m0.ndim == 2, "scales_e8m0 must be rank-2")
        _require(
            qweight_u8.size(0) == scales_e8m0.size(0),
            "row dimension mismatch",
        )
        _require(
            qweight_u8.size(1) * 2 == scales_e8m0.size(1) * 32,
            "shape mismatch",
        )
        return (
            _pack_mxfp4_weight_kernel_layout(qweight_u8),
            _pack_mxfp4_scale_kernel_layout(scales_e8m0),
        )

    _require(qweight_u8.ndim == 3, "qweight_u8 must be rank-2 or rank-3")
    _require(scales_e8m0.ndim == 3, "scales_e8m0 must be rank-2 or rank-3")
    experts = qweight_u8.size(0)
    qw = _pack_mxfp4_weight_kernel_layout(qweight_u8.view(-1, qweight_u8.size(2)))
    so = _pack_mxfp4_scale_kernel_layout(scales_e8m0.view(-1, scales_e8m0.size(2)))
    return (
        qw.view(experts, qweight_u8.size(1), -1),
        so.view(experts, scales_e8m0.size(1), -1),
    )
