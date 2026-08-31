from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import petit_kernel

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("mega_moe_worker.py")
MULTIGRAPH_WORKER = Path(__file__).with_name("mega_moe_multigraph_worker.py")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {
                "world_size": world_size,
                "num_experts": 32,
                "topk": 4,
                "model_dim": 2880,
            },
            id=f"gpt-oss-ep{world_size}",
        )
        for world_size in (1, 2, 4, 8)
    ]
    + [
        pytest.param(
            {
                "world_size": 8,
                "num_experts": 128,
                "topk": 4,
                "model_dim": 2880,
            },
            id="gpt-oss-120b",
        ),
        pytest.param(
            {
                "world_size": 8,
                "num_experts": 256,
                "topk": 8,
                "model_dim": 7168,
                "inter_dim": 2048,
                "activation_function": (
                    petit_kernel.MegaMoeActivationFunction.silu
                ),
                "has_bias": False,
            },
            id="deepseek-v3.2",
        ),
        pytest.param(
            {
                "world_size": 8,
                "num_experts": 384,
                "topk": 6,
                "model_dim": 7168,
                "inter_dim": 3072,
                "activation_function": (
                    petit_kernel.MegaMoeActivationFunction.silu
                ),
                "has_bias": False,
            },
            id="deepseek-v4",
        ),
    ],
)
def test_mega_moe_registered_configs(kwargs: dict[str, object]) -> None:
    config = petit_kernel.MegaMoeConfig(
        **kwargs,
        activation="mxfp4",
        stages=petit_kernel.MegaMoeStages.two_stage,
    )

    assert config.activation is petit_kernel.MegaMoeActivation.mxfp4
    assert config.stages is petit_kernel.MegaMoeStages.two_stage


@pytest.mark.parametrize(
    "kwargs",
    [
        {"world_size": 3},
        {"inter_dim": 4096},
        {"num_experts": 16},
        {"topk": 2},
        {"model_dim": 1024},
        {"activation": "bf16", "stages": petit_kernel.MegaMoeStages.two_stage},
        {"stages": petit_kernel.MegaMoeStages.one_stage},
        {"activation": "not-a-dtype"},
        {"stages": 2},
    ],
)
def test_mega_moe_config_rejects_unregistered_configs(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "world_size": 2,
        "num_experts": 32,
        "topk": 4,
        "model_dim": 2880,
        "activation": "mxfp4",
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        petit_kernel.MegaMoeConfig(**values)


def test_mega_moe_config_validates_workspace_and_launch_sizes() -> None:
    config = petit_kernel.MegaMoeConfig(
        world_size=1,
        num_experts=32,
        topk=4,
        model_dim=2880,
        activation="mxfp4",
    )
    for max_tokens in (0, config.max_tokens_per_rank + 1):
        with pytest.raises(ValueError, match="token capacity"):
            config.input_views(object(), max_tokens)

    dummy = torch.empty(0)
    with pytest.raises(ValueError, match="token count"):
        config.run(
            object(), dummy, dummy, dummy, dummy, config.max_tokens_per_rank + 1
        )
    with pytest.raises(ValueError, match="out must have shape"):
        config.run(
            object(),
            dummy,
            dummy,
            dummy,
            dummy,
            1,
            out=torch.empty((1, config.model_dim + 8)),
        )


@pytest.fixture(scope="module")
def mxfp4_quantizer_case() -> tuple[
    petit_kernel.MegaMoeConfig,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if not has_gfx950(1):
        pytest.skip("requires one gfx950 GPU")
    config = petit_kernel.MegaMoeConfig(
        world_size=8,
        num_experts=128,
        topk=4,
        model_dim=2880,
        activation="mxfp4",
        stages=petit_kernel.MegaMoeStages.two_stage,
    )
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.bfloat16,
        device="cuda",
    )
    values = table.repeat(8, 192)[:, : config.model_dim]
    expected_bytes = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
        dtype=torch.uint8,
        device="cuda",
    ).repeat(8, config.model_dim // 16)
    return config, table, values, expected_bytes


def test_mega_moe_mxfp4_quantizer_matches_native_encoding(
    mxfp4_quantizer_case,
) -> None:
    config, table, values, expected_bytes = mxfp4_quantizer_case
    quantized, scales = config.quantize(values)

    assert torch.equal(quantized, expected_bytes)
    assert torch.equal(scales, torch.full_like(scales, 127))
    assert scales.shape == (8, config.model_dim // 32)
    assert quantized.stride() == (1536, 1)
    assert scales.stride() == (1536, 1)
    assert scales.data_ptr() == quantized.data_ptr() + config.model_dim // 2
    physical_scales = scales.as_strided((8, 96), (1536, 1))
    assert not torch.count_nonzero(physical_scales[:, 90:])

    deepseek_config = petit_kernel.MegaMoeConfig(
        world_size=8,
        num_experts=256,
        topk=8,
        model_dim=7168,
        activation="mxfp4",
        activation_function=petit_kernel.MegaMoeActivationFunction.silu,
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=2048,
        has_bias=False,
    )
    deepseek_values = table.repeat(
        2, deepseek_config.model_dim // table.numel()
    )
    _, deepseek_scales = deepseek_config.quantize(deepseek_values)
    assert deepseek_scales.shape == (2, 224)
    assert deepseek_scales.stride() == (3808, 1)
    assert torch.equal(deepseek_scales, torch.full_like(deepseek_scales, 127))


def test_mega_moe_mxfp4_quantizer_reuses_output_and_handles_empty_values(
    mxfp4_quantizer_case,
) -> None:
    config, _, values, expected_bytes = mxfp4_quantizer_case
    quantized, _ = config.quantize(values)
    output_storage = torch.empty(
        (values.size(0), quantized.stride(0)),
        dtype=torch.uint8,
        device=values.device,
    )
    output_tokens = output_storage[:, : config.model_dim // 2]
    output_scales = output_storage[
        :, config.model_dim // 2 : config.model_dim // 2 + config.model_dim // 32
    ]
    output_views = petit_kernel.MegaMoeInputViews(
        output_tokens,
        output_scales,
        torch.empty((8, config.topk), dtype=torch.int32, device=values.device),
        torch.empty((8, config.topk), dtype=torch.float32, device=values.device),
    )
    returned_tokens, returned_scales = config.quantize(values, out=output_views)
    assert returned_tokens.data_ptr() == output_tokens.data_ptr()
    assert returned_scales.data_ptr() == output_scales.data_ptr()
    assert torch.equal(returned_tokens, expected_bytes)
    assert torch.equal(returned_scales, torch.full_like(returned_scales, 127))
    assert not torch.count_nonzero(output_storage[:, 1530:])

    empty_tokens, empty_scales = config.quantize(
        values[:0],
        out=petit_kernel.MegaMoeInputViews(
            output_tokens[:0],
            output_scales[:0],
            output_views.expert_ids[:0],
            output_views.expert_weights[:0],
        ),
    )
    assert empty_tokens.shape == (0, config.model_dim // 2)
    assert empty_scales.shape == (0, config.model_dim // 32)

    zeros, zero_scales = config.quantize(torch.zeros_like(values))
    assert not torch.count_nonzero(zeros)
    assert not torch.count_nonzero(zero_scales)
    assert zeros.stride() == (1536, 1)
    assert zero_scales.stride() == (1536, 1)
    assert zero_scales.data_ptr() == zeros.data_ptr() + config.model_dim // 2

def test_mega_moe_mxfp4_quantizer_validates_inputs(
    mxfp4_quantizer_case,
) -> None:
    config, _, values, _ = mxfp4_quantizer_case
    output_storage = torch.empty(
        (values.size(0), 1536), dtype=torch.uint8, device=values.device
    )
    output_tokens = output_storage[:, : config.model_dim // 2]
    output_scales = output_storage[
        :, config.model_dim // 2 : config.model_dim // 2 + config.model_dim // 32
    ]
    output_views = petit_kernel.MegaMoeInputViews(
        output_tokens,
        output_scales,
        torch.empty((8, config.topk), dtype=torch.int32, device=values.device),
        torch.empty((8, config.topk), dtype=torch.float32, device=values.device),
    )
    with pytest.raises(ValueError, match="shape"):
        config.quantize(values[:, :-32])
    with pytest.raises(RuntimeError, match="dtype"):
        config.quantize(values.float())
    with pytest.raises(ValueError, match="include scales"):
        config.quantize(
            values,
            out=petit_kernel.MegaMoeInputViews(
                output_tokens,
                None,
                output_views.expert_ids,
                output_views.expert_weights,
            ),
        )
    with pytest.raises(RuntimeError, match="view into the output rows"):
        config.quantize(
            values,
            out=petit_kernel.MegaMoeInputViews(
                output_tokens,
                output_storage[:, 1441:1531],
                output_views.expert_ids,
                output_views.expert_weights,
            ),
        )
    with pytest.raises(RuntimeError, match="output must be"):
        config.quantize(
            values,
            out=petit_kernel.MegaMoeInputViews(
                output_tokens[:, :-1],
                output_scales,
                output_views.expert_ids,
                output_views.expert_weights,
            ),
        )


def has_gfx950(count: int) -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= count
        and all(
            getattr(
                torch.cuda.get_device_properties(device), "gcnArchName", ""
            ).startswith("gfx950")
            for device in range(count)
        )
    )


def run_worker(
    world_size: int,
    activation: str,
    *,
    cuda_graph: bool = False,
    registered_shape: bool = False,
    zero_token_rank: bool = False,
    uneven_tokens: bool = False,
    varying_tokens: bool = False,
    skewed_routing: bool = False,
    two_stage: bool = False,
    num_experts: int | None = None,
    tokens: int = 17,
    repeat: int = 1,
    graph_layers: int = 1,
) -> None:
    if not has_gfx950(world_size):
        pytest.skip(f"requires {world_size} gfx950 GPUs")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        str(WORKER),
        "--activation",
        activation,
    ]
    if registered_shape:
        command.append("--registered-shape")
    if cuda_graph:
        command.append("--cuda-graph")
    if zero_token_rank:
        command.append("--zero-token-rank")
    if uneven_tokens:
        command.append("--uneven-tokens")
    if varying_tokens:
        command.append("--varying-tokens")
    if skewed_routing:
        command.append("--skewed-routing")
    if two_stage:
        command.append("--two-stage")
    if num_experts is not None:
        command.extend(("--num-experts", str(num_experts)))
    command.extend(
        (
            "--tokens",
            str(tokens),
            "--repeat",
            str(repeat),
            "--graph-layers",
            str(graph_layers),
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f"MegaMoE deadlocked\n{output}", pytrace=False)
    assert process.returncode == 0, output
    graph_suffix = " with CUDA graph" if cuda_graph else ""
    stage_prefix = "two-stage " if two_stage else ""
    assert (
        f"MegaMoE {stage_prefix}{activation} passed with EP="
        f"{world_size}{graph_suffix}" in output
    )


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_two_stage_mega_moe_matches_distributed_reference(
    world_size: int,
) -> None:
    run_worker(world_size, "mxfp4", registered_shape=True, two_stage=True)


def test_mega_moe_supports_cuda_graph() -> None:
    run_worker(
        2,
        "mxfp4",
        cuda_graph=True,
        registered_shape=True,
        two_stage=True,
    )


def test_two_stage_mega_moe_gpt_oss_120b_shape_supports_cuda_graph() -> None:
    run_worker(
        8,
        "mxfp4",
        cuda_graph=True,
        registered_shape=True,
        two_stage=True,
        num_experts=128,
    )


def test_two_stage_mega_moe_ep8_repeated_graph_outputs_are_complete() -> None:
    run_worker(
        8,
        "mxfp4",
        cuda_graph=True,
        registered_shape=True,
        two_stage=True,
        tokens=8,
        repeat=128,
        graph_layers=24,
    )


def test_two_stage_mega_moe_ep8_uneven_graph_outputs_are_complete() -> None:
    run_worker(
        8,
        "mxfp4",
        cuda_graph=True,
        registered_shape=True,
        two_stage=True,
        uneven_tokens=True,
        tokens=8,
        repeat=128,
        graph_layers=24,
    )


def test_two_stage_mega_moe_ep8_varying_token_counts_are_complete() -> None:
    run_worker(
        8,
        "mxfp4",
        registered_shape=True,
        two_stage=True,
        varying_tokens=True,
        tokens=8,
    )


def test_two_stage_mega_moe_ep8_skewed_graph_outputs_are_complete() -> None:
    run_worker(
        8,
        "mxfp4",
        cuda_graph=True,
        registered_shape=True,
        two_stage=True,
        skewed_routing=True,
        tokens=8,
        repeat=128,
        graph_layers=24,
    )


def test_deepseek_v32_mega_moe_supports_multiple_cuda_graphs() -> None:
    if not has_gfx950(8):
        pytest.skip("requires 8 gfx950 GPUs")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        str(MULTIGRAPH_WORKER),
        "--shape",
        "dsv32",
        "--capture-sizes",
        "128,120,112",
        "--layers",
        "61",
        "--quantize",
        "--skewed",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    assert process.returncode == 0, process.stdout
    assert "captured 112" in process.stdout
    assert "replayed 112" in process.stdout


def test_two_stage_mega_moe_handles_a_zero_token_rank() -> None:
    run_worker(
        2,
        "mxfp4",
        registered_shape=True,
        zero_token_rank=True,
        two_stage=True,
    )
