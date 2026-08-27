#!/usr/bin/env python3
"""Benchmark AITER, local MoE, and MegaMoE on five-shot GSM8K serving."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import IO, Any, Self

DP_SIZE = 8
GSM8K_EXAMPLES = 1319
DEFAULT_CONCURRENCIES = (8, 16, 32, 64, 128)
GSM8K_FEWSHOT_DOC_IDS = (346, 6096, 4585, 187, 5242)
DEFAULT_VISIBLE_DEVICES = ",".join(str(index) for index in range(DP_SIZE))
REPO_ROOT = Path(__file__).resolve().parents[3]
DEEPSEEK_CHAT_TEMPLATE = """\
{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}\
{{ '<｜User｜>' + message['content'] }}{% elif message['role'] == 'assistant' %}\
{{ '<｜Assistant｜>' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}\
{{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}\
{{ '<｜Assistant｜></think>' }}{% endif %}\
"""


@dataclass(frozen=True)
class Backend:
    name: str
    moe_backend: str
    all2all_backend: str
    server_marker: str


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_subdir: str
    served_model_name: str
    max_model_len: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    aiter_server_marker: str
    kv_cache_dtype: str | None = None
    reasoning_parser: str | None = None
    reasoning_effort: str | None = None
    chat_template: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    served_model_name: str
    backend: str
    concurrency: int
    shots: int
    max_output_tokens: int
    completed: int
    failed: int
    duration_s: float
    request_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    mean_tpot_ms: float
    exact_match: float
    cap_hits: int
    max_observed_output_tokens: int
    server_startup_s: float
    benchmark_json: str
    dataset_file: str
    server_log: str
    benchmark_log: str


BACKENDS = {
    "megamoe": Backend(
        "megamoe", "petit", "petit", "Using Petit two-stage MegaMoE fused path"
    ),
    "local_moe": Backend(
        "local_moe",
        "petit",
        "allgather_reducescatter",
        "Using 'PETIT' Mxfp4 MoE backend.",
    ),
    "aiter": Backend(
        "aiter",
        "aiter",
        "allgather_reducescatter",
        "model-specific AITER Mxfp4 MoE marker",
    ),
}

MODELS = {
    "gpt_oss_120b": ModelProfile(
        "gpt_oss_120b",
        "gpt-oss-120b",
        "openai/gpt-oss-120b",
        8096,
        128,
        0.85,
        "Using 'AITER_MXFP4_BF16' Mxfp4 MoE backend",
        reasoning_parser="openai_gptoss",
        reasoning_effort="low",
    ),
    "dsv3": ModelProfile(
        "dsv3",
        "amd--DeepSeek-V3.2-mxfp4",
        "deepseek-ai/DeepSeek-V3.2",
        4096,
        128,
        0.90,
        "Using 'AITER_MXFP4_MXFP4' Mxfp4 MoE backend",
        kv_cache_dtype="fp8",
        chat_template=DEEPSEEK_CHAT_TEMPLATE,
    ),
    "dsv4": ModelProfile(
        "dsv4",
        "DeepSeek-V4-Pro-0813",
        "deepseek-ai/DeepSeek-V4-Pro",
        4096,
        1024,
        0.90,
        "Using 'AITER_MXFP4_BF16' Mxfp4 MoE backend",
        kv_cache_dtype="fp8",
        reasoning_parser="deepseek_v4",
        chat_template=DEEPSEEK_CHAT_TEMPLATE,
    ),
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description=(
            "Run full five-shot GSM8K serving sweeps against AITER, Petit local "
            "MoE, and Petit MegaMoE on eight GPUs."
        )
    )
    parser.add_argument(
        "--models", choices=tuple(MODELS), nargs="+", default=list(MODELS)
    )
    parser.add_argument(
        "--backends", choices=tuple(BACKENDS), nargs="+", default=list(BACKENDS)
    )
    parser.add_argument(
        "--concurrencies",
        type=positive_int,
        nargs="+",
        default=list(DEFAULT_CONCURRENCIES),
    )
    parser.add_argument("--max-gen-toks", type=positive_int, default=2048)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "Force every performance-probe request to max-gen-toks. Do not "
            "use this for accuracy validation."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture a bounded PyTorch CPU/GPU trace for each benchmark point.",
    )
    parser.add_argument("--profile-delay-iters", type=nonnegative_int, default=1)
    parser.add_argument("--profile-max-iters", type=positive_int, default=5)
    parser.add_argument("--num-fewshot", type=nonnegative_int, default=5)
    parser.add_argument("--num-warmups", type=nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_data") / f"vllm_gsm8k_moe_sweep_{timestamp}",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive_int, default=30308)
    parser.add_argument("--dp-rpc-port", type=positive_int, default=29908)
    parser.add_argument("--master-port", type=positive_int, default=29508)
    parser.add_argument("--api-server-count", type=positive_int, default=2)
    parser.add_argument("--visible-devices", default=DEFAULT_VISIBLE_DEVICES)
    parser.add_argument("--server-timeout", type=positive_int, default=1800)
    parser.add_argument("--shutdown-timeout", type=positive_int, default=90)
    parser.add_argument(
        "--limit",
        type=nonnegative_int,
        default=0,
        help="Use only the first N test examples; zero uses all 1,319.",
    )
    parser.add_argument("--vllm-command", default=None)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs for short correctness/debugging runs.",
    )
    parser.add_argument(
        "--cudagraph-capture-sizes",
        type=positive_int,
        nargs="+",
        default=None,
        help="Override graph capture sizes for focused correctness runs.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for option in ("models", "backends", "concurrencies"):
        values = getattr(args, option)
        if len(set(values)) != len(values):
            parser.error(f"--{option.replace('_', '-')} must not contain duplicates")
    if args.num_fewshot > len(GSM8K_FEWSHOT_DOC_IDS):
        parser.error(
            f"--num-fewshot cannot exceed {len(GSM8K_FEWSHOT_DOC_IDS)}"
        )
    if args.limit > GSM8K_EXAMPLES:
        parser.error(f"--limit cannot exceed {GSM8K_EXAMPLES}")
    if args.enforce_eager and args.cudagraph_capture_sizes is not None:
        parser.error(
            "--enforce-eager and --cudagraph-capture-sizes are mutually exclusive"
        )
    return args


def selected_models(args: argparse.Namespace) -> list[ModelProfile]:
    return [MODELS[name] for name in args.models]


def selected_backends(args: argparse.Namespace) -> list[Backend]:
    return [BACKENDS[name] for name in args.backends]


def model_path(args: argparse.Namespace, model: ModelProfile) -> Path:
    return args.model_root / model.model_subdir


def chat_template_path(args: argparse.Namespace, model: ModelProfile) -> Path:
    return args.output_dir / "runtime" / f"{model.name}_chat_template.jinja"


def prepare_chat_template(args: argparse.Namespace, model: ModelProfile) -> None:
    if model.chat_template is None:
        return
    path = chat_template_path(args, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != model.chat_template:
        path.write_text(model.chat_template, encoding="utf-8")


def expected_examples(args: argparse.Namespace) -> int:
    return args.limit or GSM8K_EXAMPLES


def readiness_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def server_marker(model: ModelProfile, backend: Backend) -> str:
    return model.aiter_server_marker if backend.name == "aiter" else backend.server_marker


def resolve_vllm_command(args: argparse.Namespace) -> str:
    command = args.vllm_command or shutil.which("vllm")
    if not command:
        raise RuntimeError("could not find the vllm executable on PATH")
    resolved = shutil.which(command) if os.sep not in command else command
    if not resolved or not os.path.isfile(resolved) or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"vllm executable is not runnable: {command}")
    return os.path.abspath(resolved)


def launcher_python(command: str) -> str:
    try:
        first_line = Path(command).open(encoding="utf-8").readline().strip()
    except OSError as exc:
        raise RuntimeError(f"could not inspect launcher {command}: {exc}") from exc
    interpreter = shlex.split(first_line[2:]) if first_line.startswith("#!") else []
    if not interpreter:
        raise RuntimeError(f"vLLM launcher has no usable shebang: {command}")
    if Path(interpreter[0]).name == "env":
        if len(interpreter) != 2 or not (resolved := shutil.which(interpreter[1])):
            raise RuntimeError(f"unsupported vLLM launcher shebang: {first_line}")
        return resolved
    return interpreter[0]


def installed_aiter_jit_dir(vllm_command: str) -> Path:
    probe = subprocess.run(
        [
            launcher_python(vllm_command),
            "-c",
            (
                "import importlib.util, pathlib; "
                "spec = importlib.util.find_spec('aiter'); "
                "assert spec is not None and spec.origin is not None; "
                "print(pathlib.Path(spec.origin).parent / 'jit')"
            ),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe.returncode:
        raise RuntimeError(f"could not locate AITER: {probe.stderr.strip()}")
    source = Path(probe.stdout.strip())
    if not source.is_dir():
        raise RuntimeError(f"AITER JIT directory does not exist: {source}")
    return source


def link_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for artifact in source.iterdir():
        link = destination / artifact.name
        if link.is_symlink():
            if link.resolve() != artifact.resolve():
                raise RuntimeError(f"cache link {link} points to the wrong artifact")
        elif link.exists():
            raise RuntimeError(f"cache artifact is not a symlink: {link}")
        else:
            link.symlink_to(artifact.resolve(), target_is_directory=artifact.is_dir())


def make_aiter_jit_cache(vllm_command: str, root: Path) -> Path:
    source = installed_aiter_jit_dir(vllm_command)
    cache = root / "aiter_jit"
    cache.mkdir(parents=True, exist_ok=True)
    for artifact in source.iterdir():
        if artifact.name == "flydsl_cache" or not artifact.is_file():
            continue
        link = cache / artifact.name
        if link.is_symlink():
            if link.resolve() != artifact.resolve():
                raise RuntimeError(f"cache link {link} points to the wrong artifact")
        elif link.exists():
            raise RuntimeError(f"cache artifact is not a symlink: {link}")
        else:
            link.symlink_to(artifact.resolve())
    flydsl_source = source / "flydsl_cache"
    flydsl_cache = cache / "flydsl_cache"
    if flydsl_source.is_dir():
        link_artifacts(flydsl_source, flydsl_cache)
    else:
        flydsl_cache.mkdir(exist_ok=True)
    return cache


def backend_environment_overrides(backend: Backend) -> dict[str, str]:
    return {
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_MOE": "1" if backend.name == "aiter" else "0",
        "VLLM_ROCM_USE_AITER_LINEAR": "0",
        "VLLM_ROCM_USE_AITER_RMSNORM": "0",
        "VLLM_ROCM_USE_AITER_MLA": "0",
        "VLLM_ROCM_USE_AITER_MHA": "0",
        "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": "0",
        "VLLM_ROCM_USE_AITER_FP8BMM": "0",
        "VLLM_ROCM_USE_AITER_FP4BMM": "0",
        "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": "0",
        "VLLM_ROCM_USE_AITER_TRITON_GEMM": "0",
        "VLLM_ROCM_USE_AITER_TRITON_ROPE": "0",
    }


def run_environment(
    args: argparse.Namespace,
    backend: Backend,
    aiter_jit_dir: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = args.visible_devices
    pythonpath = [str(REPO_ROOT)]
    if existing := env.get("PYTHONPATH"):
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    ld_path = ["/opt/rocm/lib", "/usr/local/lib"]
    if existing := env.get("LD_LIBRARY_PATH"):
        ld_path.append(existing)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_path)
    env["AITER_JIT_DIR"] = str(aiter_jit_dir)
    env["FLYDSL_RUNTIME_CACHE_DIR"] = str(aiter_jit_dir / "flydsl_cache")
    env["VLLM_CACHE_ROOT"] = str(args.output_dir / "runtime" / "vllm_cache_v2")
    env["VLLM_ENGINE_READY_TIMEOUT_S"] = str(args.server_timeout)
    env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
    env.update(backend_environment_overrides(backend))
    return env


def build_server_command(
    args: argparse.Namespace,
    model: ModelProfile,
    backend: Backend,
    vllm_command: str,
) -> list[str]:
    command = [
        vllm_command,
        "serve",
        str(model_path(args, model)),
        "--served-model-name",
        model.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-server-count",
        str(args.api_server_count),
        "--data-parallel-size",
        str(DP_SIZE),
        "--data-parallel-size-local",
        str(DP_SIZE),
        "--data-parallel-address",
        "127.0.0.1",
        "--data-parallel-rpc-port",
        str(args.dp_rpc_port),
        "--master-port",
        str(args.master_port),
        "--enable-expert-parallel",
        "--moe-backend",
        backend.moe_backend,
        "--all2all-backend",
        backend.all2all_backend,
        "--max-model-len",
        str(model.max_model_len),
        "--max-num-batched-tokens",
        str(model.max_num_batched_tokens),
        "--max-num-seqs",
        # Petit local-MoE graph capture requires its documented M=128
        # workspace even when a smoke test requests lower client concurrency.
        str(max(128, max(args.concurrencies))),
        "--gpu-memory-utilization",
        str(model.gpu_memory_utilization),
        "--trust-remote-code",
        "--no-async-scheduling",
        "--disable-log-stats",
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    else:
        compilation_config: dict[str, object] = {"cudagraph_mode": "FULL"}
        if args.cudagraph_capture_sizes is not None:
            compilation_config["cudagraph_capture_sizes"] = (
                args.cudagraph_capture_sizes
            )
        command.extend(("--compilation-config", json.dumps(compilation_config)))
    if model.kv_cache_dtype is not None:
        command.extend(("--kv-cache-dtype", model.kv_cache_dtype))
    if model.reasoning_parser is not None:
        command.extend(("--reasoning-parser", model.reasoning_parser))
    if model.chat_template is not None:
        command.extend(("--chat-template", str(chat_template_path(args, model))))
    if args.profile:
        profiler_dir = (
            args.output_dir / "profiles" / model.name / backend.name
        ).resolve()
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": str(profiler_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": False,
            "ignore_frontend": True,
            "delay_iterations": args.profile_delay_iters,
            "max_iterations": args.profile_max_iters,
        }
        command.extend(("--profiler-config", json.dumps(profiler_config)))
    return command


def dataset_path(args: argparse.Namespace) -> Path:
    return args.output_dir / f"gsm8k_{args.num_fewshot}shot.jsonl"


def create_gsm8k_dataset(args: argparse.Namespace, path: Path) -> None:
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) != expected_examples(args):
            raise RuntimeError(
                f"existing dataset {path} has {len(rows)} rows, expected "
                f"{expected_examples(args)}"
            )
        if any(int(row.get("output_tokens", -1)) != args.max_gen_toks for row in rows):
            raise RuntimeError(f"existing dataset {path} has the wrong output length")
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("the datasets package is required to build GSM8K") from exc
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    fewshot = "\n\n".join(
        f"Question: {train[doc_id]['question']}\nAnswer: {train[doc_id]['answer']}"
        for doc_id in GSM8K_FEWSHOT_DOC_IDS[: args.num_fewshot]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        for doc_id, row in enumerate(test.select(range(expected_examples(args)))):
            question = f"Question: {row['question']}\nAnswer:"
            prompt = f"{fewshot}\n\n{question}" if fewshot else question
            output.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "prompt": prompt,
                        "answer": row["answer"],
                        "output_tokens": args.max_gen_toks,
                    }
                )
                + "\n"
            )


def point_paths(
    args: argparse.Namespace,
    model: ModelProfile,
    backend: Backend,
    concurrency: int,
) -> tuple[Path, Path, Path]:
    point_dir = args.output_dir / model.name / backend.name / f"c{concurrency}"
    return (
        point_dir / "benchmark.json",
        point_dir / "benchmark.log",
        point_dir / "result.json",
    )


def build_benchmark_command(
    args: argparse.Namespace,
    model: ModelProfile,
    backend: Backend,
    concurrency: int,
    benchmark_json: Path,
    vllm_command: str,
) -> list[str]:
    command = [
        vllm_command,
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        f"http://{readiness_host(args.host)}:{args.port}",
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        model.served_model_name,
        "--tokenizer",
        str(model_path(args, model)),
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset_path(args)),
        "--no-oversample",
        "--disable-shuffle",
        "--custom-output-len",
        str(args.max_gen_toks),
        "--num-prompts",
        str(expected_examples(args)),
        "--num-warmups",
        str(args.num_warmups),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--seed",
        str(args.seed),
        "--temperature",
        "0",
        "--percentile-metrics",
        "ttft,tpot",
        "--metric-percentiles",
        "99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(benchmark_json.parent),
        "--result-filename",
        benchmark_json.name,
        "--metadata",
        f"stack={backend.name}",
        f"dp={DP_SIZE}",
        f"ep={DP_SIZE}",
        f"shots={args.num_fewshot}",
        f"max_output_tokens={args.max_gen_toks}",
        f"concurrency={concurrency}",
    ]
    if args.ignore_eos:
        command.append("--ignore-eos")
    if args.profile:
        command.append("--profile")
    if model.reasoning_effort is not None:
        command.extend(
            ("--extra-body", json.dumps({"reasoning_effort": model.reasoning_effort}))
        )
    if model.chat_template is not None:
        # The custom dataset's prompt is already the complete five-shot user
        # turn. Let the server apply the explicit model template once.
        command.append("--skip-chat-template")
    return command


def tail(path: Path, line_count: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    except OSError:
        return ""
    return "".join(lines[-line_count:])


class VllmServer:
    def __init__(
        self,
        command: Sequence[str],
        env: dict[str, str],
        log_path: Path,
        health_url: str,
        startup_timeout: int,
        shutdown_timeout: int,
        expected_marker: str,
    ) -> None:
        self.command = list(command)
        self.env = env
        self.log_path = log_path
        self.health_url = health_url
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.expected_marker = expected_marker
        self.process: subprocess.Popen[str] | None = None
        self._log_file: IO[str] | None = None
        self.startup_s = 0.0

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", encoding="utf-8")
        print(f"$ {shlex.join(self.command)}", flush=True)
        print(f"Server log: {self.log_path}", flush=True)
        started = time.monotonic()
        self.process = subprocess.Popen(
            self.command,
            env=self.env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = started + self.startup_timeout
        while time.monotonic() < deadline:
            if (return_code := self.process.poll()) is not None:
                raise RuntimeError(
                    f"vLLM server exited during startup with code {return_code}\n"
                    f"{tail(self.log_path)}"
                )
            try:
                with urllib.request.urlopen(self.health_url, timeout=2) as response:
                    if 200 <= response.status < 300:
                        self._log_file.flush()
                        log = self.log_path.read_text(encoding="utf-8", errors="replace")
                        if self.expected_marker not in log:
                            raise RuntimeError(
                                "vLLM became ready without expected backend marker "
                                f"{self.expected_marker!r}\n{tail(self.log_path)}"
                            )
                        self.startup_s = time.monotonic() - started
                        print(
                            f"vLLM ready after {self.startup_s:.1f}s; backend verified.",
                            flush=True,
                        )
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2)
        raise RuntimeError(
            f"vLLM server did not become ready within {self.startup_timeout}s\n"
            f"{tail(self.log_path)}"
        )

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                print("vLLM did not stop cleanly; sending SIGKILL.", file=sys.stderr)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> Self:
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()


def run_logged_command(
    command: Sequence[str], env: dict[str, str], log_path: Path
) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


HASH_ANSWER = re.compile(r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")
ANY_NUMBER = re.compile(r"-?\$?\s*\d[\d,]*(?:\.\d+)?")


def normalize_number(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.replace("$", "").replace(",", "").replace(" ", "").strip()
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    # Decimal.quantize() obeys the current context precision and can reject
    # arbitrarily long integer strings produced by a looping model response.
    # Fixed-point formatting does not impose that precision limit.
    formatted = format(number, "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def extract_answer(text: str | None) -> str | None:
    hash_matches = HASH_ANSWER.findall(text or "")
    if hash_matches:
        return normalize_number(hash_matches[-1])
    matches = ANY_NUMBER.findall(text or "")
    return normalize_number(matches[-1]) if matches else None


def score_responses(dataset_file: Path, generated_texts: Sequence[str]) -> float:
    rows = [json.loads(line) for line in dataset_file.read_text().splitlines() if line]
    if len(rows) != len(generated_texts):
        raise RuntimeError(
            f"detailed result has {len(generated_texts)} responses for {len(rows)} rows"
        )
    correct = 0
    for row, response in zip(rows, generated_texts):
        gold_matches = HASH_ANSWER.findall(row["answer"])
        if not gold_matches:
            raise RuntimeError(f"GSM8K row {row['doc_id']} has no #### answer")
        correct += extract_answer(response) == normalize_number(gold_matches[-1])
    return correct / len(rows)


def parse_benchmark_result(
    args: argparse.Namespace,
    model: ModelProfile,
    backend: Backend,
    concurrency: int,
    server: VllmServer,
    benchmark_json: Path,
    benchmark_log: Path,
) -> BenchmarkResult:
    try:
        raw: dict[str, Any] = json.loads(benchmark_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {benchmark_json}: {exc}") from exc
    expected = expected_examples(args)
    completed = int(raw.get("completed", -1))
    failed = int(raw.get("failed", -1))
    if completed != expected or failed != 0:
        raise RuntimeError(
            f"incomplete benchmark: completed={completed}, failed={failed}, "
            f"expected={expected}"
        )
    if int(raw.get("max_concurrency", -1)) != concurrency:
        raise RuntimeError(f"wrong concurrency recorded in {benchmark_json}")
    generated_texts = raw.get("generated_texts", [])
    output_lens = [int(length) for length in raw.get("output_lens", [])]
    if len(generated_texts) != expected or len(output_lens) != expected:
        raise RuntimeError(f"detailed fields are incomplete in {benchmark_json}")
    return BenchmarkResult(
        model=model.name,
        served_model_name=model.served_model_name,
        backend=backend.name,
        concurrency=concurrency,
        shots=args.num_fewshot,
        max_output_tokens=args.max_gen_toks,
        completed=completed,
        failed=failed,
        duration_s=float(raw["duration"]),
        request_throughput=float(raw["request_throughput"]),
        output_throughput=float(raw["output_throughput"]),
        mean_ttft_ms=float(raw["mean_ttft_ms"]),
        mean_tpot_ms=float(raw["mean_tpot_ms"]),
        exact_match=score_responses(dataset_path(args), generated_texts),
        cap_hits=sum(length == args.max_gen_toks for length in output_lens),
        max_observed_output_tokens=max(output_lens, default=0),
        server_startup_s=server.startup_s,
        benchmark_json=str(benchmark_json),
        dataset_file=str(dataset_path(args)),
        server_log=str(server.log_path),
        benchmark_log=str(benchmark_log),
    )


def load_completed_result(
    args: argparse.Namespace,
    model: ModelProfile,
    backend: Backend,
    concurrency: int,
) -> BenchmarkResult | None:
    if not args.resume:
        return None
    _, _, result_json = point_paths(args, model, backend, concurrency)
    if not result_json.is_file():
        return None
    try:
        result = BenchmarkResult(**json.loads(result_json.read_text()))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"Cannot reuse {result_json}: {exc}; rerunning.", flush=True)
        return None
    valid = (
        result.model == model.name
        and result.backend == backend.name
        and result.concurrency == concurrency
        and result.shots == args.num_fewshot
        and result.max_output_tokens == args.max_gen_toks
        and result.completed == expected_examples(args)
        and result.failed == 0
    )
    if not valid:
        print(f"Cannot reuse mismatched {result_json}; rerunning.", flush=True)
        return None
    print(f"Reusing {result_json}", flush=True)
    return result


def write_result(path: Path, result: BenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


def print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.model:12} {result.backend:9} c={result.concurrency:<3} "
        f"out={result.output_throughput:9.2f} tok/s "
        f"TTFT={result.mean_ttft_ms:9.2f} ms "
        f"TPOT={result.mean_tpot_ms:8.2f} ms "
        f"accuracy={result.exact_match:.4f}",
        flush=True,
    )


def run_backend_sweep(
    args: argparse.Namespace,
    model: ModelProfile,
    backend: Backend,
    vllm_command: str,
    aiter_jit_dir: Path,
) -> list[BenchmarkResult]:
    results: dict[int, BenchmarkResult] = {}
    missing: list[int] = []
    for concurrency in args.concurrencies:
        reusable = load_completed_result(args, model, backend, concurrency)
        if reusable is None:
            missing.append(concurrency)
        else:
            results[concurrency] = reusable
            print_result(reusable)
    if not missing:
        return [results[concurrency] for concurrency in args.concurrencies]

    env = run_environment(args, backend, aiter_jit_dir)
    server_log = args.output_dir / "logs" / f"{model.name}_{backend.name}_server.log"
    health_base = f"http://{readiness_host(args.host)}:{args.port}"
    server = VllmServer(
        build_server_command(args, model, backend, vllm_command),
        env,
        server_log,
        f"{health_base}/health",
        args.server_timeout,
        args.shutdown_timeout,
        server_marker(model, backend),
    )
    with server:
        for concurrency in missing:
            benchmark_json, benchmark_log, result_json = point_paths(
                args, model, backend, concurrency
            )
            if args.resume and benchmark_json.is_file() and not result_json.exists():
                print(f"Recovering completed raw result {benchmark_json}", flush=True)
                result = parse_benchmark_result(
                    args,
                    model,
                    backend,
                    concurrency,
                    server,
                    benchmark_json,
                    benchmark_log,
                )
                write_result(result_json, result)
                results[concurrency] = result
                print_result(result)
                continue
            if benchmark_json.exists() or result_json.exists():
                raise RuntimeError(
                    f"partial output exists below {benchmark_json.parent}; "
                    "remove it or use a new --output-dir"
                )
            run_logged_command(
                build_benchmark_command(
                    args, model, backend, concurrency, benchmark_json, vllm_command
                ),
                env,
                benchmark_log,
            )
            with urllib.request.urlopen(f"{health_base}/health", timeout=10) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError(f"server unhealthy after concurrency {concurrency}")
            result = parse_benchmark_result(
                args,
                model,
                backend,
                concurrency,
                server,
                benchmark_json,
                benchmark_log,
            )
            write_result(result_json, result)
            results[concurrency] = result
            print_result(result)
    return [results[concurrency] for concurrency in args.concurrencies]


def write_summaries(
    args: argparse.Namespace,
    results: Sequence[BenchmarkResult],
    vllm_command: str,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    summary = {
        "config": {
            "models": [asdict(model) for model in selected_models(args)],
            "backends": [
                {
                    **asdict(backend),
                    "environment_overrides": backend_environment_overrides(backend),
                }
                for backend in selected_backends(args)
            ],
            "concurrencies": args.concurrencies,
            "shots": args.num_fewshot,
            "max_output_tokens": args.max_gen_toks,
            "ignore_eos": args.ignore_eos,
            "profile": args.profile,
            "num_prompts": expected_examples(args),
            "num_warmups": args.num_warmups,
            "seed": args.seed,
            "model_root": str(args.model_root),
            "vllm_command": vllm_command,
            "dp_size": DP_SIZE,
            "ep_size": DP_SIZE,
            "visible_devices": args.visible_devices,
        },
        "results": [asdict(result) for result in results],
    }
    json_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {json_path}", flush=True)


def print_summary(results: Sequence[BenchmarkResult]) -> None:
    headers = (
        "model",
        "stack",
        "conc",
        "output tok/s",
        "TTFT ms",
        "TPOT ms",
        "accuracy",
    )
    rows = [
        (
            result.model,
            result.backend,
            str(result.concurrency),
            f"{result.output_throughput:.2f}",
            f"{result.mean_ttft_ms:.2f}",
            f"{result.mean_tpot_ms:.2f}",
            f"{result.exact_match:.4f}",
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("\n" + "  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def validate_port_free(host: str, port: int) -> None:
    bind_host = readiness_host(host)
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((bind_host, port))
        except OSError as exc:
            raise RuntimeError(f"port {bind_host}:{port} is unavailable: {exc}") from exc


def visible_device_ordinals(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise RuntimeError(
            "--visible-devices must be a comma-separated list of ordinals"
        ) from exc
    if len(devices) != DP_SIZE or len(set(devices)) != DP_SIZE:
        raise RuntimeError(f"DP={DP_SIZE} requires eight unique GPU ordinals")
    return devices


def validate_gpus(devices: Sequence[int]) -> None:
    rocminfo = shutil.which("rocminfo")
    if not rocminfo:
        raise RuntimeError("rocminfo is required to validate the ROCm GPUs")
    completed = subprocess.run(
        [rocminfo],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    gpu_agents = re.findall(r"^\s*Name:\s+gfx\d+\s*$", completed.stdout, re.MULTILINE)
    if completed.returncode or len(gpu_agents) <= max(devices):
        raise RuntimeError(
            f"requested GPU ordinal {max(devices)}, but found {len(gpu_agents)} agents"
        )


def validate_prerequisites(args: argparse.Namespace) -> str:
    command = resolve_vllm_command(args)
    for model in selected_models(args):
        if not model_path(args, model).joinpath("config.json").is_file():
            raise RuntimeError(f"model directory is incomplete: {model_path(args, model)}")
    devices = visible_device_ordinals(args.visible_devices)
    validate_gpus(devices)
    ports = (args.port, args.dp_rpc_port, args.master_port)
    if len(set(ports)) != len(ports):
        raise RuntimeError("HTTP, DP RPC, and master ports must be distinct")
    for port in ports:
        validate_port_free(args.host, port)
    return command


def print_dry_run(args: argparse.Namespace, vllm_command: str) -> None:
    for model in selected_models(args):
        for backend in selected_backends(args):
            print(f"$ {shlex.join(build_server_command(args, model, backend, vllm_command))}")
            for concurrency in args.concurrencies:
                benchmark_json, _, _ = point_paths(args, model, backend, concurrency)
                command = build_benchmark_command(
                    args, model, backend, concurrency, benchmark_json, vllm_command
                )
                print(f"$ {shlex.join(command)}")


def handle_termination(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, handle_termination)
    try:
        if args.dry_run:
            print_dry_run(args, args.vllm_command or shutil.which("vllm") or "vllm")
            return 0
        vllm_command = validate_prerequisites(args)
        create_gsm8k_dataset(args, dataset_path(args))
        for model in selected_models(args):
            prepare_chat_template(args, model)
        runtime_root = args.output_dir / "runtime"
        aiter_jit_dir = make_aiter_jit_cache(vllm_command, runtime_root)
        results: list[BenchmarkResult] = []
        for model in selected_models(args):
            for backend in selected_backends(args):
                results.extend(
                    run_backend_sweep(
                        args,
                        model,
                        backend,
                        vllm_command,
                        aiter_jit_dir,
                    )
                )
                write_summaries(args, results, vllm_command)
        print_summary(results)
        return 0
    except KeyboardInterrupt:
        print("Benchmark interrupted.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
