"""Run predefined local MoE comparisons for supported model shapes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TOKENS = (8, 16, 32, 64, 128, 256, 512, 1024)
WARMUP = 20
REPEAT = 200
GRAPH_ITERS = 16
BENCHMARK_PYTHON = Path(sys.executable)


@dataclass(frozen=True)
class ModelPreset:
    dim: int
    inter_dim: int
    experts: int
    topk: int
    backends: tuple[tuple[str, str], ...]


MODEL_PRESETS = {
    "gpt-oss-120b": ModelPreset(
        dim=3072,
        inter_dim=3072,
        experts=16,
        topk=4,
        backends=(
            ("petit", "mxfp4_native_mxfp4_openai_bias_2stage"),
            ("aiter", "aiter_mxfp4_openai_bias_2stage"),
        ),
    ),
    # EP=8: 256 / 8 routed experts plus one shared expert and route.
    "deepseekv3": ModelPreset(
        dim=7168,
        inter_dim=2048,
        experts=33,
        topk=9,
        backends=(
            ("petit", "mxfp4_native_mxfp4_silu_2stage"),
            ("aiter", "aiter_mxfp4_silu"),
        ),
    ),
    # DeepSeek-V4-Pro at EP=8: 384 / 8 routed experts plus one shared expert.
    "deepseekv4": ModelPreset(
        dim=7168,
        inter_dim=3072,
        experts=49,
        topk=7,
        backends=(
            ("petit", "mxfp4_native_mxfp4_silu_2stage"),
            ("aiter", "aiter_mxfp4_silu"),
        ),
    ),
}

INT_FIELDS = {
    "graph",
    "solution_id",
    "tokens",
    "dim",
    "inter_dim",
    "experts",
    "topk",
    "valid_ids",
    "valid_m_blocks",
    "persistent_tgs",
    "total_iters",
}
FLOAT_FIELDS = {"device_ms", "tflops", "gbps"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a predefined Petit/AITER local MoE sweep."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("all", *MODEL_PRESETS),
        default="gpt-oss-120b",
        help=(
            "Model shape and kernel pair to benchmark, or all presets "
            "(default: gpt-oss-120b)"
        ),
    )
    parser.add_argument(
        "--device",
        default="0",
        help="HIP_VISIBLE_DEVICES value for the single-GPU benchmark (default: 0)",
    )
    return parser.parse_args()


def parse_result(output: str, backend: str) -> dict[str, object]:
    prefix = f"backend={backend} "
    try:
        line = next(line for line in reversed(output.splitlines()) if line.startswith(prefix))
    except StopIteration as exc:
        raise RuntimeError(f"missing {backend} result line") from exc

    row: dict[str, object] = {}
    for field in line.split():
        key, value = field.split("=", 1)
        if key in INT_FIELDS:
            row[key] = int(value)
        elif key in FLOAT_FIELDS:
            row[key] = float(value)
        else:
            row[key] = value
    return row


def run_one(
    benchmark: Path,
    backend: str,
    kernel_variant: str,
    tokens: int,
    device: str,
    model: str,
) -> dict[str, object]:
    command = [
        str(BENCHMARK_PYTHON),
        str(benchmark),
        "--backend",
        backend,
        "--kernel-variant",
        kernel_variant,
        "--model",
        model,
        "--tokens",
        str(tokens),
        "--warmup",
        str(WARMUP),
        "--repeat",
        str(REPEAT),
        "--graph-iters",
        str(GRAPH_ITERS),
    ]
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = device
    print(f"$ {shlex.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=benchmark.parents[3],
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return parse_result(completed.stdout, backend)


def main() -> int:
    args = parse_args()
    benchmark = Path(__file__).with_name("bench_aiter_fused_moe.py").resolve()
    model_names = tuple(MODEL_PRESETS) if args.model == "all" else (args.model,)
    results: list[dict[str, object]] = []
    for model_name in model_names:
        preset = MODEL_PRESETS[model_name]
        results.extend(
            run_one(benchmark, backend, variant, tokens, args.device, model_name)
            for backend, variant in preset.backends
            for tokens in TOKENS
        )
    for row in results:
        row.update(
            warmup=WARMUP,
            repeat_requested=REPEAT,
            graph_iters=GRAPH_ITERS,
            device=args.device,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        fieldnames = list(
            dict.fromkeys(key for row in results for key in row)
        )
        with args.output.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(args.output)
        return 0
    args.output.write_text(
        json.dumps(
            {
                "config": {
                    "model": args.model,
                    "model_presets": {
                        model_name: {
                            "dim": MODEL_PRESETS[model_name].dim,
                            "inter_dim": MODEL_PRESETS[model_name].inter_dim,
                            "experts": MODEL_PRESETS[model_name].experts,
                            "topk": MODEL_PRESETS[model_name].topk,
                            "kernel_variants": dict(
                                MODEL_PRESETS[model_name].backends
                            ),
                        }
                        for model_name in model_names
                    },
                    "tokens": TOKENS,
                    "warmup": WARMUP,
                    "repeat": REPEAT,
                    "graph_iters": GRAPH_ITERS,
                    "device": args.device,
                    "python": str(BENCHMARK_PYTHON),
                },
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
