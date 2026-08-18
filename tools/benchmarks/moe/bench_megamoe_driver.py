"""Run the predefined GPT-OSS-120B three-backend MegaMoE comparison."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

# Edit these constants when a different benchmark sweep is needed.
BACKENDS = ("petit", "aiter", "flydsl")
TOKENS = (8, 16, 32, 64, 128, 256, 512, 1024)
GLOBAL_EXPERTS = 128
WARMUP = 20
REPEAT = 200
GRAPH_ITERS = 16
DEFAULT_BENCHMARK_PYTHON = Path(
    os.environ.get("BENCHMARK_PYTHON", sys.executable)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the predefined Petit/AITER/FlyDSL MegaMoE sweep."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_BENCHMARK_PYTHON,
        help=(
            "Python executable used for torchrun (default: BENCHMARK_PYTHON "
            "or the invoking interpreter)"
        ),
    )
    parser.add_argument(
        "--devices",
        help="optional HIP_VISIBLE_DEVICES list, for example 0,1,2,3,4,5,6,7",
    )
    args = parser.parse_args()
    if args.devices is not None:
        devices = tuple(
            item.strip() for item in args.devices.split(",") if item.strip()
        )
        if len(devices) != args.ep_size:
            parser.error(f"--devices must contain exactly {args.ep_size} entries")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        parser.error(f"--python is not an executable file: {args.python}")
    return args


def run_backend(
    benchmark: Path,
    backend: str,
    jsonl: Path,
    devices: str | None,
    ep_size: int,
    benchmark_python: Path,
) -> list[dict[str, object]]:
    token_args = [str(token) for token in TOKENS]
    command = [
        str(benchmark_python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={ep_size}",
        "--",
        str(benchmark),
        "--backend",
        backend,
        "--dp-size",
        str(ep_size),
        "--ep-size",
        str(ep_size),
        "--global-experts",
        str(GLOBAL_EXPERTS),
        "--m",
        *token_args,
        "--batch-size",
        *token_args,
        "--warmup",
        str(WARMUP),
        "--repeat",
        str(REPEAT),
        "--graph-iters",
        str(GRAPH_ITERS),
        "--jsonl",
        str(jsonl),
    ]
    env = os.environ.copy()
    if devices is not None:
        env["HIP_VISIBLE_DEVICES"] = devices
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=benchmark.parents[3], env=env, check=True)
    return [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    benchmark = Path(__file__).with_name("bench_megamoe.py").resolve()
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="bench-megamoe-") as temp_dir:
        for backend in BACKENDS:
            results.extend(
                run_backend(
                    benchmark,
                    backend,
                    Path(temp_dir) / f"{backend}.jsonl",
                    args.devices,
                    args.ep_size,
                    args.python,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "config": {
                    "backends": BACKENDS,
                    "tokens": TOKENS,
                    "ep_size": args.ep_size,
                    "global_experts": GLOBAL_EXPERTS,
                    "warmup": WARMUP,
                    "repeat": REPEAT,
                    "graph_iters": GRAPH_ITERS,
                    "python": str(args.python),
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
