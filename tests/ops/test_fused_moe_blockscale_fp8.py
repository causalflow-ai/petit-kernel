import pytest
import torch
import torch.nn.functional as F
import petit_kernel
from fused_moe_replay_like import SENSITIVE_ATOL as REPLAY_LIKE_SENSITIVE_ATOL
from fused_moe_replay_like import topk_tensors as replay_like_topk_tensors


KERNEL_SYMBOL = "_ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256E"

DEEPSEEK_V32_EXP = {
    "dim": 7168,
    "moe_inter_dim": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "n_activated_experts": 8,
    "n_expert_groups": 8,
}

BLOCK_SHAPE = (128, 128)
DEEPSEEK_EXPERTS = (
    DEEPSEEK_V32_EXP["n_routed_experts"] // DEEPSEEK_V32_EXP["n_expert_groups"]
    + DEEPSEEK_V32_EXP["n_shared_experts"]
)
DEEPSEEK_TOPK = DEEPSEEK_V32_EXP["n_activated_experts"] + DEEPSEEK_V32_EXP["n_shared_experts"]

TEST_CONFIGS = [
    pytest.param(
        8,
        DEEPSEEK_V32_EXP["dim"],
        DEEPSEEK_V32_EXP["moe_inter_dim"],
        DEEPSEEK_EXPERTS,
        DEEPSEEK_TOPK,
        id="deepseek_like",
    ),
]
PER_ELEMENT_ATOL = 5e-2
PER_ELEMENT_RTOL = 5e-2
ULTRA_HARSH_PER_ELEMENT_ATOL = 1.0
ULTRA_HARSH_PER_ELEMENT_RTOL = 5e-2


def _run_petit_fp8_kernel(
    kernel_data: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    topk: int,
) -> torch.Tensor:
    return petit_kernel.fused_moe_fp8_blockscale_g1u1(
        kernel_data["input_q"],
        kernel_data["w1_q_shuffled"],
        kernel_data["w2_q_shuffled"],
        data["sorted_token_ids"],
        data["sorted_weights"],
        data["sorted_expert_ids"],
        data["num_valid_ids"],
        topk,
        kernel_data["input_scale_aiter"],
        kernel_data["fc1_scale_aiter"],
        kernel_data["fc2_scale_aiter"],
    )


class FusedMoESiluTorchOps:
    def __init__(
        self,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        self.block_n, self.block_k = BLOCK_SHAPE
        self.out_dtype = out_dtype

    def _dequantize_input(
        self,
        input_q: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> torch.Tensor:
        tokens, model_dim = input_q.shape
        blocks = input_q.to(torch.float32).view(tokens, model_dim // self.block_k, self.block_k)
        return (blocks * input_scale.to(torch.float32).unsqueeze(-1)).reshape(tokens, model_dim)

    def _dequantize_weight(
        self,
        weight_q: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        experts, rows, cols = weight_q.shape
        row_blocks = rows // self.block_n
        col_blocks = cols // self.block_k

        dense_q = weight_q.to(torch.float32)
        blocks = (
            dense_q.view(experts, row_blocks, self.block_n, col_blocks, self.block_k)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        scaled_blocks = blocks * scale.to(torch.float32).view(
            experts, row_blocks, col_blocks, 1, 1
        )
        return (
            scaled_blocks.permute(0, 1, 3, 2, 4)
            .reshape(experts, rows, cols)
            .contiguous()
        )

    def forward(
        self,
        input_q: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        input_scale: torch.Tensor,
        fc1_scale: torch.Tensor,
        fc2_scale: torch.Tensor,
    ) -> torch.Tensor:
        tokens, model_dim = input_q.shape
        experts, _, inter_dim = w2_q.shape

        input_f = self._dequantize_input(input_q, input_scale)
        out = torch.zeros((tokens, model_dim), dtype=torch.float32, device=input_q.device)
        topk_weights_f = topk_weights.to(torch.float32)

        for expert_idx in range(experts):
            expert_mask = topk_ids == expert_idx
            if not torch.any(expert_mask):
                continue

            token_ids, route_ids = torch.where(expert_mask)
            token_ids = token_ids.to(torch.int64)
            route_ids = route_ids.to(torch.int64)

            x_e = input_f[token_ids]
            w1_e = self._dequantize_weight(
                w1_q[expert_idx : expert_idx + 1],
                fc1_scale[expert_idx : expert_idx + 1],
            )[0]
            stage1 = torch.matmul(x_e, w1_e.t())
            gate, up = stage1.split([inter_dim, inter_dim], dim=-1)
            activated = F.silu(gate) * up

            w2_e = self._dequantize_weight(
                w2_q[expert_idx : expert_idx + 1],
                fc2_scale[expert_idx : expert_idx + 1],
            )[0]
            route_out = torch.matmul(activated, w2_e.t())
            weighted = route_out * topk_weights_f[token_ids, route_ids].unsqueeze(1)
            out.index_add_(0, token_ids, weighted)

        return out.to(self.out_dtype)


class FmoeBlockscaleFp8AiterTestDataBuilder:
    AITER_STYLE_WEIGHT_DIVISOR = 60.0
    ULTRA_HARSH_WEIGHT_STD = 40.0
    ULTRA_HARSH_WEIGHT_MEAN = 0.0
    ULTRA_HARSH_SCALE_INV_STD = 2.0e-4
    ULTRA_HARSH_SCALE_INV_MEAN = 1.0e-3
    INPUT_STD = 1.0
    INPUT_MEAN = 0.1
    INPUT_SPIKE_PROB = 0.002
    INPUT_SPIKE_STD = 16.0

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        tokens: int,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
    ):
        self.device = device
        self.dtype = dtype
        self.tokens = tokens
        self.model_dim = model_dim
        self.inter_dim = inter_dim
        self.experts = experts
        self.topk = topk
        block_n, block_k = BLOCK_SHAPE
        assert tokens > 0 and experts > 0 and topk > 0
        assert topk <= experts, f"topk ({topk}) must be <= experts ({experts})"
        assert model_dim % 256 == 0, f"model_dim ({model_dim}) must be divisible by 256"
        assert model_dim % block_k == 0 and model_dim % block_n == 0
        assert inter_dim % block_k == 0 and (inter_dim * 2) % block_n == 0

    def _rand_positive_normal(
        self,
        shape: tuple[int, ...],
        mean: float,
        std: float,
    ) -> torch.Tensor:
        x = torch.randn(shape, dtype=torch.float32, device=self.device) * std + mean
        return x.clamp_min(1e-8)

    def _sample_spiky_input(
        self,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        tokens, model_dim = shape
        base = (
            torch.randn(shape, dtype=torch.float32, device=self.device) * self.INPUT_STD
            + self.INPUT_MEAN
        )
        spike_mask = (
            torch.rand(shape, dtype=torch.float32, device=self.device) < self.INPUT_SPIKE_PROB
        )
        spikes = torch.randn(shape, dtype=torch.float32, device=self.device) * self.INPUT_SPIKE_STD
        token_ramp = torch.linspace(-1.0, 1.0, tokens, dtype=torch.float32, device=self.device)
        dim_ramp = torch.linspace(-1.0, 1.0, model_dim, dtype=torch.float32, device=self.device)
        structured = 0.35 * token_ramp.unsqueeze(1) + 0.15 * dim_ramp.unsqueeze(0)
        return base + structured + spikes * spike_mask.to(torch.float32)

    @staticmethod
    def _pack_blocks(x: torch.Tensor, block_n: int, block_k: int) -> torch.Tensor:
        experts, rows, cols = x.shape
        pack_k = 16
        kk = block_k // pack_k
        return (
            x.view(experts, rows // block_n, block_n, cols // block_k, kk, pack_k)
            .permute(0, 1, 3, 4, 2, 5)
            .contiguous()
        )

    @staticmethod
    def _mod4_mult(idx: torch.Tensor) -> torch.Tensor:
        return torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32, device=idx.device)[
            idx & 3
        ]

    @staticmethod
    def _block4_mult(idx: torch.Tensor) -> torch.Tensor:
        return torch.tensor([1.0, 1.0, 2.0, 4.0], dtype=torch.float32, device=idx.device)[
            idx & 3
        ]

    def _apply_ultra_harsh_stage1_scale_pattern(
        self,
        input_scale: torch.Tensor,
        fc1_scale: torch.Tensor,
    ) -> None:
        block_n, block_k = BLOCK_SHAPE
        n_blocks = self.model_dim // block_k
        w1_row_blocks = (self.inter_dim * 2) // block_n

        token_idx = torch.arange(self.tokens, dtype=torch.int64, device=self.device)
        block_idx = torch.arange(n_blocks, dtype=torch.int64, device=self.device)
        token_m = self._mod4_mult(token_idx)
        block_m = self._block4_mult(block_idx)
        input_scale.mul_(token_m.unsqueeze(1) * block_m.unsqueeze(0))

        row_idx = torch.arange(w1_row_blocks, dtype=torch.int64, device=self.device)
        row_m = self._mod4_mult(row_idx)
        fc1_view = fc1_scale.view(self.experts, w1_row_blocks, n_blocks)
        fc1_view.mul_(row_m.view(1, w1_row_blocks, 1) * block_m.view(1, 1, n_blocks))

    def _build_padded_sorted_routing(
        self,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        block_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_num_tokens_padded = int(topk_ids.numel() + self.experts * block_size - self.topk)
        max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
        init_val = (self.topk << 24) | self.tokens

        sorted_token_ids = torch.full((max_num_tokens_padded,), init_val, dtype=torch.int32)
        sorted_weights = torch.zeros((max_num_tokens_padded,), dtype=torch.float32)
        sorted_expert_ids = torch.full((max_num_m_blocks,), -1, dtype=torch.int32)

        topk_ids_cpu = topk_ids.to("cpu", dtype=torch.int32)
        topk_weights_cpu = topk_weights.to("cpu", dtype=torch.float32)
        token_ids_cpu = (
            torch.arange(self.tokens, dtype=torch.int32)
            .unsqueeze(1)
            .expand(self.tokens, self.topk)
            .reshape(-1)
        )
        slot_ids_cpu = (
            torch.arange(self.topk, dtype=torch.int32).unsqueeze(0).expand(self.tokens, self.topk).reshape(-1)
        )
        expert_ids_cpu = topk_ids_cpu.reshape(-1)
        route_weights_cpu = topk_weights_cpu.reshape(-1)

        order = torch.argsort(expert_ids_cpu, stable=True)
        token_ids_cpu = token_ids_cpu[order]
        slot_ids_cpu = slot_ids_cpu[order]
        expert_ids_cpu = expert_ids_cpu[order]
        route_weights_cpu = route_weights_cpu[order]

        expert_counts = torch.bincount(expert_ids_cpu, minlength=self.experts)
        route_begin = 0
        sorted_ids_begin = 0
        sorted_expert_ids_begin = 0
        for expert in range(self.experts):
            tokens_num = int(expert_counts[expert].item())
            if tokens_num == 0:
                continue
            route_end = route_begin + tokens_num
            sorted_token_ids[sorted_ids_begin:sorted_ids_begin + tokens_num] = (
                (slot_ids_cpu[route_begin:route_end] << 24) | token_ids_cpu[route_begin:route_end]
            )
            sorted_weights[sorted_ids_begin:sorted_ids_begin + tokens_num] = route_weights_cpu[
                route_begin:route_end
            ]

            sorted_expert_ids_num = (tokens_num + block_size - 1) // block_size
            tokens_num_pad = sorted_expert_ids_num * block_size
            sorted_expert_ids[
                sorted_expert_ids_begin:sorted_expert_ids_begin + sorted_expert_ids_num
            ] = expert

            route_begin = route_end
            sorted_ids_begin += tokens_num_pad
            sorted_expert_ids_begin += sorted_expert_ids_num

        # [num_valid_ids_padded, token_count]
        num_valid_ids = torch.tensor([sorted_ids_begin, self.tokens], dtype=torch.int32)
        return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids

    def to_aiter_kernel_layout(
        self,
        *,
        input_q: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        input_scale: torch.Tensor,
        fc1_scale: torch.Tensor,
        fc2_scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        aiter = pytest.importorskip("aiter")
        shuffle_weight = pytest.importorskip("aiter.ops.shuffle").shuffle_weight
        input_scale_aiter = torch.empty_like(input_scale)
        num_rows = torch.tensor([self.tokens], dtype=torch.int32, device=self.device)
        aiter.partial_transpose(input_scale_aiter, input_scale, num_rows=num_rows)
        return {
            "input_q": input_q,
            "w1_q_shuffled": shuffle_weight(w1_q, (16, 16)),
            "w2_q_shuffled": shuffle_weight(w2_q, (16, 16)),
            "input_scale_aiter": input_scale_aiter,
            "fc1_scale_aiter": fc1_scale.contiguous(),
            "fc2_scale_aiter": fc2_scale.contiguous(),
        }

    def _finalize_routing(
        self,
        *,
        input_q: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        input_scale: torch.Tensor,
        fc1_scale: torch.Tensor,
        fc2_scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
            self._build_padded_sorted_routing(
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
        )
        return {
            "input_q": input_q,
            "w1_q": w1_q,
            "w2_q": w2_q,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "input_scale": input_scale,
            "fc1_scale": fc1_scale,
            "fc2_scale": fc2_scale,
            "sorted_token_ids": sorted_token_ids.to(self.device),
            "sorted_weights": sorted_weights.to(self.device),
            "sorted_expert_ids": sorted_expert_ids.to(self.device),
            "num_valid_ids": num_valid_ids.to(self.device),
        }

    def build(self) -> dict[str, torch.Tensor]:
        block_n, block_k = BLOCK_SHAPE
        aiter = pytest.importorskip("aiter")
        fused_topk = pytest.importorskip("aiter.fused_moe").fused_topk
        quant_dtype = aiter.dtypes.fp8
        w1_rows = self.inter_dim * 2
        w1_row_blocks = w1_rows // block_n
        w2_row_blocks = self.model_dim // block_n
        model_blocks = self.model_dim // block_k
        inter_blocks = self.inter_dim // block_k

        input_f = torch.randn((self.tokens, self.model_dim), dtype=self.dtype, device=self.device)
        score = torch.randn((self.tokens, self.experts), dtype=self.dtype, device=self.device)
        topk_weights, topk_ids = fused_topk(input_f, score, self.topk, True)
        topk_weights = topk_weights.to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)

        w1 = (
            torch.randn(
                (self.experts, w1_rows, self.model_dim),
                dtype=self.dtype,
                device=self.device,
            )
            / self.AITER_STYLE_WEIGHT_DIVISOR
        )
        w2 = (
            torch.randn(
                (self.experts, self.model_dim, self.inter_dim),
                dtype=self.dtype,
                device=self.device,
            )
            / self.AITER_STYLE_WEIGHT_DIVISOR
        )

        w1_blocks = (
            w1.view(self.experts, w1_row_blocks, block_n, model_blocks, block_k)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(self.experts, w1_row_blocks, model_blocks, block_n * block_k)
        )
        w1_q_blocks, fc1_scale = aiter.pertoken_quant(w1_blocks, quant_dtype=quant_dtype)
        w1_q = (
            w1_q_blocks.view(self.experts, w1_row_blocks, model_blocks, block_n, block_k)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(self.experts, w1_rows, self.model_dim)
        )

        w2_blocks = (
            w2.view(self.experts, w2_row_blocks, block_n, inter_blocks, block_k)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(self.experts, w2_row_blocks, inter_blocks, block_n * block_k)
        )
        w2_q_blocks, fc2_scale = aiter.pertoken_quant(w2_blocks, quant_dtype=quant_dtype)
        w2_q = (
            w2_q_blocks.view(self.experts, w2_row_blocks, inter_blocks, block_n, block_k)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(self.experts, self.model_dim, self.inter_dim)
        )

        input_q_blocks, input_scale = aiter.pertoken_quant(
            input_f.view(self.tokens, model_blocks, block_k),
            quant_dtype=quant_dtype,
        )
        input_q = input_q_blocks.view(self.tokens, self.model_dim).contiguous()

        return self._finalize_routing(
            input_q=input_q,
            w1_q=w1_q,
            w2_q=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scale=input_scale.squeeze(-1).to(torch.float32),
            fc1_scale=fc1_scale.view(self.experts, -1).to(torch.float32),
            fc2_scale=fc2_scale.view(self.experts, -1).to(torch.float32),
        )

    def build_ultra_harsh(self) -> dict[str, torch.Tensor]:
        block_n, block_k = BLOCK_SHAPE
        input_q = self._sample_spiky_input((self.tokens, self.model_dim)).to(
            torch.float8_e4m3fnuz
        )
        input_scale = self._rand_positive_normal(
            (self.tokens, self.model_dim // block_k),
            self.ULTRA_HARSH_SCALE_INV_MEAN,
            self.ULTRA_HARSH_SCALE_INV_STD,
        )
        w1_q = (
            torch.randn(
                (self.experts, self.inter_dim * 2, self.model_dim),
                dtype=torch.float32,
                device=self.device,
            )
            * self.ULTRA_HARSH_WEIGHT_STD
            + self.ULTRA_HARSH_WEIGHT_MEAN
        ).to(torch.float8_e4m3fnuz)
        w2_q = (
            torch.randn(
                (self.experts, self.model_dim, self.inter_dim),
                dtype=torch.float32,
                device=self.device,
            )
            * self.ULTRA_HARSH_WEIGHT_STD
            + self.ULTRA_HARSH_WEIGHT_MEAN
        ).to(torch.float8_e4m3fnuz)

        token_idx = torch.arange(self.tokens, dtype=torch.int64, device=self.device).unsqueeze(1)
        slot_idx = torch.arange(self.topk, dtype=torch.int64, device=self.device).unsqueeze(0)
        topk_ids = (token_idx * 7 + slot_idx * 5) % self.experts
        expert_perm = torch.randperm(self.experts, dtype=torch.int64, device=self.device)
        topk_ids = expert_perm[topk_ids].to(torch.int32)

        slot_decay = torch.pow(
            torch.full((self.topk,), 0.5, dtype=torch.float32, device=self.device),
            torch.arange(self.topk, dtype=torch.float32, device=self.device),
        )
        topk_weights = slot_decay.unsqueeze(0).expand(self.tokens, -1).contiguous()
        topk_weights = topk_weights * (
            1.0
            + 0.05
            * torch.randn((self.tokens, self.topk), dtype=torch.float32, device=self.device)
        ).clamp_min(0.1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        fc1_scale = self._rand_positive_normal(
            (self.experts, ((self.inter_dim * 2) // block_n) * (self.model_dim // block_k)),
            self.ULTRA_HARSH_SCALE_INV_MEAN,
            self.ULTRA_HARSH_SCALE_INV_STD,
        )
        fc2_scale = self._rand_positive_normal(
            (self.experts, (self.model_dim // block_n) * (self.inter_dim // block_k)),
            self.ULTRA_HARSH_SCALE_INV_MEAN,
            self.ULTRA_HARSH_SCALE_INV_STD,
        )
        self._apply_ultra_harsh_stage1_scale_pattern(input_scale, fc1_scale)

        return self._finalize_routing(
            input_q=input_q,
            w1_q=w1_q,
            w2_q=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scale=input_scale,
            fc1_scale=fc1_scale,
            fc2_scale=fc2_scale,
        )

    def build_replay_like_sensitive(self) -> dict[str, torch.Tensor]:
        block_n, block_k = BLOCK_SHAPE
        assert self.tokens == 2
        assert self.model_dim == 7168
        assert self.inter_dim == 2048
        assert self.experts == 32
        assert self.topk == 8

        input_q = self._sample_spiky_input((self.tokens, self.model_dim)).to(
            torch.float8_e4m3fnuz
        )
        input_scale = self._rand_positive_normal(
            (self.tokens, self.model_dim // block_k), 0.10, 0.02
        )
        w1_q = (
            1.2
            * torch.randn(
                (self.experts, self.inter_dim * 2, self.model_dim),
                dtype=torch.float32,
                device=self.device,
            )
        ).to(torch.float8_e4m3fnuz)
        w2_q = (
            torch.randn(
                (self.experts, self.model_dim, self.inter_dim),
                dtype=torch.float32,
                device=self.device,
            )
        ).to(torch.float8_e4m3fnuz)
        fc1_scale = self._rand_positive_normal(
            (self.experts, ((self.inter_dim * 2) // block_n) * (self.model_dim // block_k)),
            0.020,
            0.004,
        )
        fc2_scale = self._rand_positive_normal(
            (self.experts, (self.model_dim // block_n) * (self.inter_dim // block_k)),
            0.025,
            0.004,
        )
        topk_ids, topk_weights = replay_like_topk_tensors(self.device)

        return self._finalize_routing(
            input_q=input_q,
            w1_q=w1_q,
            w2_q=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scale=input_scale,
            fc1_scale=fc1_scale,
            fc2_scale=fc2_scale,
        )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="AITER kernel requires a GPU")
@pytest.mark.parametrize("tokens,model_dim,inter_dim,experts,topk", TEST_CONFIGS)
def test_fmoe_blockscale_fp8_reference_matches_aiter_kernel(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
):
    aiter = pytest.importorskip("aiter")
    if not hasattr(torch, "float8_e4m3fnuz"):
        pytest.skip("torch.float8_e4m3fnuz is required for this kernel")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    ref_impl = FusedMoESiluTorchOps(
        out_dtype=dtype,
    )

    torch.manual_seed(42)
    builder = FmoeBlockscaleFp8AiterTestDataBuilder(
        device=device,
        dtype=dtype,
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
    )
    data = builder.build()
    kernel_data = builder.to_aiter_kernel_layout(
        input_q=data["input_q"],
        w1_q=data["w1_q"],
        w2_q=data["w2_q"],
        input_scale=data["input_scale"],
        fc1_scale=data["fc1_scale"],
        fc2_scale=data["fc2_scale"],
    )

    ref_out = ref_impl.forward(
        input_q=data["input_q"],
        w1_q=data["w1_q"],
        w2_q=data["w2_q"],
        topk_weights=data["topk_weights"],
        topk_ids=data["topk_ids"],
        input_scale=data["input_scale"],
        fc1_scale=data["fc1_scale"],
        fc2_scale=data["fc2_scale"],
    )

    out_petit = _run_petit_fp8_kernel(kernel_data, data, topk)

    out_aiter = torch.zeros((tokens, model_dim), dtype=dtype, device=device)
    aiter.fmoe_fp8_blockscale_g1u1(
        out_aiter,
        kernel_data["input_q"],
        kernel_data["w1_q_shuffled"],
        kernel_data["w2_q_shuffled"],
        data["sorted_token_ids"],
        data["sorted_weights"],
        data["sorted_expert_ids"],
        data["num_valid_ids"],
        topk,
        kernel_data["input_scale_aiter"],
        kernel_data["fc1_scale_aiter"],
        kernel_data["fc2_scale_aiter"],
        "",
        BLOCK_SHAPE[0],
        BLOCK_SHAPE[1],
        None,
    )

    torch.testing.assert_close(
        ref_out,
        out_aiter,
        rtol=PER_ELEMENT_RTOL,
        atol=PER_ELEMENT_ATOL,
        msg=f"{KERNEL_SYMBOL}: reference and aiter outputs differ",
    )
    torch.testing.assert_close(
        out_petit,
        out_aiter,
        rtol=PER_ELEMENT_RTOL,
        atol=PER_ELEMENT_ATOL,
        msg="petit and aiter outputs differ",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Petit kernel requires a GPU")
def test_fmoe_blockscale_fp8_replay_like_sensitive_matches_reference():
    if not hasattr(torch, "float8_e4m3fnuz"):
        pytest.skip("torch.float8_e4m3fnuz is required for this kernel")

    tokens = 2
    model_dim = DEEPSEEK_V32_EXP["dim"]
    inter_dim = DEEPSEEK_V32_EXP["moe_inter_dim"]
    experts = 32
    topk = 8

    device = torch.device("cuda")
    dtype = torch.bfloat16
    ref_impl = FusedMoESiluTorchOps(out_dtype=dtype)

    torch.manual_seed(1234)
    builder = FmoeBlockscaleFp8AiterTestDataBuilder(
        device=device,
        dtype=dtype,
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
    )
    data = builder.build_replay_like_sensitive()
    kernel_data = builder.to_aiter_kernel_layout(
        input_q=data["input_q"],
        w1_q=data["w1_q"],
        w2_q=data["w2_q"],
        input_scale=data["input_scale"],
        fc1_scale=data["fc1_scale"],
        fc2_scale=data["fc2_scale"],
    )

    ref_out = ref_impl.forward(
        input_q=data["input_q"],
        w1_q=data["w1_q"],
        w2_q=data["w2_q"],
        topk_weights=data["topk_weights"],
        topk_ids=data["topk_ids"],
        input_scale=data["input_scale"],
        fc1_scale=data["fc1_scale"],
        fc2_scale=data["fc2_scale"],
    )

    out_petit = _run_petit_fp8_kernel(kernel_data, data, topk)

    torch.testing.assert_close(
        out_petit,
        ref_out,
        rtol=0.0,
        atol=REPLAY_LIKE_SENSITIVE_ATOL,
        msg="petit sensitive FP8 output differs from torch reference",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AITER kernel requires a GPU")
def test_fmoe_blockscale_fp8_petit_supports_preallocated_out_and_cuda_graph():
    if not hasattr(torch, "float8_e4m3fnuz"):
        pytest.skip("torch.float8_e4m3fnuz is required for this kernel")

    tokens = 40
    model_dim = DEEPSEEK_V32_EXP["dim"]
    inter_dim = DEEPSEEK_V32_EXP["moe_inter_dim"]
    experts = DEEPSEEK_EXPERTS
    topk = DEEPSEEK_TOPK

    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(42)
    builder = FmoeBlockscaleFp8AiterTestDataBuilder(
        device=device,
        dtype=dtype,
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
    )
    data = builder.build()
    kernel_data = builder.to_aiter_kernel_layout(
        input_q=data["input_q"],
        w1_q=data["w1_q"],
        w2_q=data["w2_q"],
        input_scale=data["input_scale"],
        fc1_scale=data["fc1_scale"],
        fc2_scale=data["fc2_scale"],
    )

    def run_kernel(out: torch.Tensor) -> torch.Tensor:
        return petit_kernel.fused_moe_fp8_blockscale_g1u1(
            kernel_data["input_q"],
            kernel_data["w1_q_shuffled"],
            kernel_data["w2_q_shuffled"],
            data["sorted_token_ids"],
            data["sorted_weights"],
            data["sorted_expert_ids"],
            data["num_valid_ids"],
            topk,
            kernel_data["input_scale_aiter"],
            kernel_data["fc1_scale_aiter"],
            kernel_data["fc2_scale_aiter"],
            out=out,
        )

    eager_out = torch.empty((tokens, model_dim), dtype=dtype, device=device)
    eager_ret = run_kernel(eager_out)
    assert eager_ret.data_ptr() == eager_out.data_ptr()
    eager_expected = eager_out.clone()

    graph_out = torch.empty_like(eager_out)
    for _ in range(2):
        run_kernel(graph_out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_ret = run_kernel(graph_out)

    graph.replay()
    torch.cuda.synchronize()

    assert graph_ret.data_ptr() == graph_out.data_ptr()
    torch.testing.assert_close(
        graph_out,
        eager_expected,
        rtol=PER_ELEMENT_RTOL,
        atol=PER_ELEMENT_ATOL,
        msg="petit CUDA graph replay output differs from eager output",
    )

    with pytest.raises(
        RuntimeError,
        match=r"input_scale must be contiguous with shape \[tokens, dim/128\]",
    ):
        petit_kernel.fused_moe_fp8_blockscale_g1u1(
            kernel_data["input_q"],
            kernel_data["w1_q_shuffled"],
            kernel_data["w2_q_shuffled"],
            data["sorted_token_ids"],
            data["sorted_weights"],
            data["sorted_expert_ids"],
            data["num_valid_ids"],
            topk,
            kernel_data["input_scale_aiter"].transpose(0, 1),
            kernel_data["fc1_scale_aiter"],
            kernel_data["fc2_scale_aiter"],
            out=torch.empty((tokens, model_dim), dtype=dtype, device=device),
        )


def _build_zero_route_smoke_repro_case(
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    # Captured from smoke-test decode on the Petit path:
    # hidden_shape=(64, 7168), topk=8, num_valid_ids=[0, 64].
    tokens = 64
    model_dim = DEEPSEEK_V32_EXP["dim"]
    inter_dim = DEEPSEEK_V32_EXP["moe_inter_dim"]
    topk = DEEPSEEK_V32_EXP["n_activated_experts"]

    return {
        "input_q": torch.zeros(
            (tokens, model_dim),
            dtype=torch.float8_e4m3fnuz,
            device=device,
        ),
        "w13_q": torch.empty(
            (0, inter_dim * 2, model_dim),
            dtype=torch.float8_e4m3fnuz,
            device=device,
        ),
        "w2_q": torch.empty(
            (0, model_dim, inter_dim),
            dtype=torch.float8_e4m3fnuz,
            device=device,
        ),
        "sorted_token_ids": torch.empty((0,), dtype=torch.int32, device=device),
        "sorted_weights": torch.empty((0,), dtype=torch.float32, device=device),
        "sorted_expert_ids": torch.empty((0,), dtype=torch.int32, device=device),
        "num_valid_ids": torch.tensor([0, tokens], dtype=torch.int32, device=device),
        "input_scale": torch.ones(
            (tokens, model_dim // BLOCK_SHAPE[1]),
            dtype=torch.float32,
            device=device,
        ),
        "fc1_scale": torch.empty(
            (0, (inter_dim * 2) // BLOCK_SHAPE[0], model_dim // BLOCK_SHAPE[1]),
            dtype=torch.float32,
            device=device,
        ),
        "fc2_scale": torch.empty(
            (0, model_dim // BLOCK_SHAPE[0], inter_dim // BLOCK_SHAPE[1]),
            dtype=torch.float32,
            device=device,
        ),
        "topk": topk,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Petit kernel requires a GPU")
def test_fmoe_blockscale_fp8_zero_route_smoke_repro_supports_cuda_graph():
    if not hasattr(torch, "float8_e4m3fnuz"):
        pytest.skip("torch.float8_e4m3fnuz is required for this kernel")

    device = torch.device("cuda")
    case = _build_zero_route_smoke_repro_case(device)
    out = torch.zeros(
        (int(case["num_valid_ids"][1].item()), case["input_q"].shape[1]),
        dtype=torch.bfloat16,
        device=device,
    )

    eager_ret = petit_kernel.fused_moe_fp8_blockscale_g1u1(
        case["input_q"],
        case["w13_q"],
        case["w2_q"],
        case["sorted_token_ids"],
        case["sorted_weights"],
        case["sorted_expert_ids"],
        case["num_valid_ids"],
        int(case["topk"]),
        case["input_scale"],
        case["fc1_scale"],
        case["fc2_scale"],
        out=out,
    )
    torch.cuda.synchronize()
    assert eager_ret.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.zeros_like(out))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_ret = petit_kernel.fused_moe_fp8_blockscale_g1u1(
            case["input_q"],
            case["w13_q"],
            case["w2_q"],
            case["sorted_token_ids"],
            case["sorted_weights"],
            case["sorted_expert_ids"],
            case["num_valid_ids"],
            int(case["topk"]),
            case["input_scale"],
            case["fc1_scale"],
            case["fc2_scale"],
            out=out,
        )

    graph.replay()
    torch.cuda.synchronize()

    assert graph_ret.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.zeros_like(out))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AITER kernel requires a GPU")
def test_fmoe_blockscale_fp8_single_expert_ultra_harsh_scale_matches_reference():
    aiter = pytest.importorskip("aiter")
    if not hasattr(torch, "float8_e4m3fnuz"):
        pytest.skip("torch.float8_e4m3fnuz is required for this kernel")

    tokens = 32
    model_dim = 512
    inter_dim = 512
    experts = 1
    topk = 1

    device = torch.device("cuda")
    dtype = torch.bfloat16
    ref_impl = FusedMoESiluTorchOps(out_dtype=dtype)

    torch.manual_seed(42)
    builder = FmoeBlockscaleFp8AiterTestDataBuilder(
        device=device,
        dtype=dtype,
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
    )
    data = builder.build_ultra_harsh()
    kernel_data = builder.to_aiter_kernel_layout(
        input_q=data["input_q"],
        w1_q=data["w1_q"],
        w2_q=data["w2_q"],
        input_scale=data["input_scale"],
        fc1_scale=data["fc1_scale"],
        fc2_scale=data["fc2_scale"],
    )

    ref_out = ref_impl.forward(
        input_q=data["input_q"],
        w1_q=data["w1_q"],
        w2_q=data["w2_q"],
        topk_weights=data["topk_weights"],
        topk_ids=data["topk_ids"],
        input_scale=data["input_scale"],
        fc1_scale=data["fc1_scale"],
        fc2_scale=data["fc2_scale"],
    )

    out_aiter = torch.zeros((tokens, model_dim), dtype=dtype, device=device)
    aiter.fmoe_fp8_blockscale_g1u1(
        out_aiter,
        kernel_data["input_q"],
        kernel_data["w1_q_shuffled"],
        kernel_data["w2_q_shuffled"],
        data["sorted_token_ids"],
        data["sorted_weights"],
        data["sorted_expert_ids"],
        data["num_valid_ids"],
        topk,
        kernel_data["input_scale_aiter"],
        kernel_data["fc1_scale_aiter"],
        kernel_data["fc2_scale_aiter"],
        "",
        BLOCK_SHAPE[0],
        BLOCK_SHAPE[1],
        None,
    )

    torch.testing.assert_close(
        out_aiter,
        ref_out,
        rtol=ULTRA_HARSH_PER_ELEMENT_RTOL,
        atol=ULTRA_HARSH_PER_ELEMENT_ATOL,
        msg=f"{KERNEL_SYMBOL}: single expert outputs differ",
    )
