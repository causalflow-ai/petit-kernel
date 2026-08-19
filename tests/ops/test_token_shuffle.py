from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = (
    REPO_ROOT / "tools" / "benchmarks" / "moe" / "bench_token_shuffle.py"
)
TOKEN_SWEEP = (8, 16, 32, 64, 128, 256, 512, 1024)


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


def run_direct_push(
    world_size: int,
    *,
    routing: str,
    tokens: tuple[int, ...],
    repeat: int = 2,
    implementation: str = "direct_push",
    epoch_wrap: bool = False,
) -> None:
    if not has_gfx950(world_size):
        pytest.skip(f"requires {world_size} gfx950 GPUs")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        str(BENCHMARK),
        "--implementation",
        implementation,
        "--routing",
        routing,
        "--check",
        "--warmup",
        "1",
        "--repeat",
        str(repeat),
        "--tokens",
        *(str(value) for value in tokens),
    ]
    if epoch_wrap:
        command.append("--epoch-wrap")
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
        pytest.fail(
            f"direct-push token shuffle deadlocked\n{output}", pytrace=False
        )
    assert process.returncode == 0, output
    expected_implementations = (
        ("pull", "direct_push")
        if implementation == "both"
        else (
            ("pull", "direct_push", "pull_epoch")
            if implementation == "all"
            else (implementation,)
        )
    )
    for expected in expected_implementations:
        assert output.count(f'"implementation":"{expected}"') == len(
            tokens
        ), output


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_direct_push_token_shuffle_matches_all_to_all_sweep(
    world_size: int,
) -> None:
    run_direct_push(world_size, routing="uniform", tokens=TOKEN_SWEEP)


@pytest.mark.parametrize("routing", ["single_rank", "skewed", "invalid"])
def test_direct_push_token_shuffle_routing_edges(routing: str) -> None:
    run_direct_push(2, routing=routing, tokens=(8, 64, 1024))


def test_direct_push_token_shuffle_repeated_invocations_do_not_stall() -> None:
    run_direct_push(2, routing="skewed", tokens=(8,), repeat=50)


def test_pull_and_direct_push_share_workspace() -> None:
    run_direct_push(
        2,
        routing="skewed",
        tokens=(8, 64),
        implementation="both",
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_epoch_token_shuffle_matches_all_to_all_sweep(
    world_size: int,
) -> None:
    run_direct_push(
        world_size,
        routing="uniform",
        tokens=TOKEN_SWEEP,
        implementation="pull_epoch",
    )


@pytest.mark.parametrize("routing", ["single_rank", "skewed", "invalid"])
def test_epoch_token_shuffle_routing_edges(routing: str) -> None:
    run_direct_push(
        2,
        routing=routing,
        tokens=(8, 64, 1024),
        implementation="pull_epoch",
    )


def test_epoch_token_shuffle_repeated_invocations_do_not_stall() -> None:
    run_direct_push(
        2,
        routing="skewed",
        tokens=(8,),
        repeat=50,
        implementation="pull_epoch",
    )


def test_epoch_token_shuffle_wraparound() -> None:
    run_direct_push(
        2,
        routing="skewed",
        tokens=(8,),
        implementation="pull_epoch",
        epoch_wrap=True,
    )


def test_all_token_shuffle_implementations_share_workspace() -> None:
    run_direct_push(
        2,
        routing="skewed",
        tokens=(8, 64),
        implementation="all",
    )
