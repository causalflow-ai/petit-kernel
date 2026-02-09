#!/usr/bin/env python3

import subprocess
import itertools
import argparse
import sys

# Representative (m, n, k) for cased used in Llama-3.3 70B / 70B.
ENTRIES = [
    (15, 4096, 4096),
    (15, 4096, 14336),
    (15, 6144, 4096),
    (15, 8192, 8192),
    (15, 8192, 28672),
    (15, 10240, 8192),
    (15, 28672, 4096),
    (15, 57344, 8192),
    (44, 1280, 8192),
    (44, 4096, 4096),
    (44, 4096, 14336),
    (44, 6144, 4096),
    (44, 7168, 8192),
    (44, 8192, 1024),
    (44, 8192, 3584),
    (44, 8192, 8192),
    (44, 8192, 28672),
    (44, 10240, 8192),
    (44, 28672, 4096),
    (44, 57344, 8192),
    (566, 1280, 8192),
    (566, 7168, 8192),
    (566, 8192, 1024),
    (566, 8192, 3584),
    (582, 4096, 4096),
    (582, 4096, 14336),
    (582, 6144, 4096),
    (582, 28672, 4096),
    (611, 4096, 4096),
    (611, 4096, 14336),
    (611, 6144, 4096),
    (611, 28672, 4096),
    (874, 8192, 8192),
    (874, 8192, 28672),
    (874, 10240, 8192),
    (874, 57344, 8192),
    (932, 8192, 8192),
    (932, 8192, 28672),
    (932, 10240, 8192),
    (932, 57344, 8192),
    (1003, 8192, 8192),
    (1003, 8192, 28672),
    (1003, 10240, 8192),
    (1003, 57344, 8192),
    (1324, 4096, 4096),
    (1324, 4096, 14336),
    (1324, 6144, 4096),
    (1324, 28672, 4096),
    (1340, 1280, 8192),
    (1340, 7168, 8192),
    (1340, 8192, 1024),
    (1340, 8192, 3584),
    (1466, 1280, 8192),
    (1466, 7168, 8192),
    (1466, 8192, 1024),
    (1466, 8192, 3584),
    (1906, 1280, 8192),
    (1906, 7168, 8192),
    (1906, 8192, 1024),
    (1906, 8192, 3584),
    (2084, 8192, 8192),
    (2084, 8192, 28672),
    (2084, 10240, 8192),
    (2084, 57344, 8192),
    (4314, 4096, 4096),
    (4314, 4096, 14336),
    (4314, 6144, 4096),
    (4314, 28672, 4096),
    (14437, 4096, 4096),
    (14437, 4096, 14336),
    (14437, 6144, 4096),
    (14437, 28672, 4096),
    (15961, 1280, 8192),
    (15961, 7168, 8192),
    (15961, 8192, 1024),
    (15961, 8192, 3584),
    (16375, 8192, 8192),
    (16375, 8192, 28672),
    (16375, 10240, 8192),
    (16375, 57344, 8192),
]

ENTRIES = [
    (16, 4096, 4096),
    (16, 4096, 14336),
    (16, 6144, 4096),
    (16, 8192, 8192),
    (16, 8192, 28672),
    (16, 10240, 8192),
    (16, 28672, 4096),
    (16, 57344, 8192),
    (256, 4096, 4096),
    (256, 4096, 14336),
    (256, 6144, 4096),
    (256, 8192, 8192),
    (256, 8192, 28672),
    (256, 10240, 8192),
    (256, 28672, 4096),
    (256, 57344, 8192),
    (512, 4096, 4096),
    (512, 4096, 14336),
    (512, 6144, 4096),
    (512, 8192, 8192),
    (512, 8192, 28672),
    (512, 10240, 8192),
    (512, 28672, 4096),
    (512, 57344, 8192),
]

# Default benchmark parameters
WARMUP = 5
REPEAT = 20
BATCH = 1
DEFAULT_BACKEND = "petit"
ALGO = "tune"
DEFAULT_ATYPE = "fp16"
DEFAULT_BTYPE = "nvfp4"
DEFAULT_CTYPE = "fp16"


def run_benchmark(m, n, k, args):
    cmd = [
        "bench_matmul",
        "-backend",
        f"{args.backend}",
        "-atype",
        f"{args.atype}",
        "-btype",
        f"{args.btype}",
        "-ctype",
        f"{args.ctype}",
        "-m",
        f"{m}",
        "-k",
        f"{k}",
        "-n",
        f"{n}",
        "-warmup",
        f"{WARMUP}",
        "-repeat",
        f"{REPEAT}",
        "-batch",
        f"{BATCH}",
        "-algo",
        f"{ALGO}",
    ]

    cmd_str = " ".join(cmd)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to run benchmark: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Run matrix multiplication benchmarks")
    parser.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        help=f"Backend to use for matrix multiplication (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--atype",
        type=str,
        default=DEFAULT_ATYPE,
        help=f"Data type of matrix A (default: {DEFAULT_ATYPE})",
    )
    parser.add_argument(
        "--btype",
        type=str,
        default=DEFAULT_BTYPE,
        help=f"Data type of matrix B (default: {DEFAULT_BTYPE})",
    )
    parser.add_argument(
        "--ctype",
        type=str,
        default=DEFAULT_CTYPE,
        help=f"Data type of matrix C (default: {DEFAULT_CTYPE})",
    )

    args = parser.parse_args()

    for m, n, k in ENTRIES:
        run_benchmark(m, n, k, args)


if __name__ == "__main__":
    main()
