#!/usr/bin/env -S uv run
"""
Unified experiment runner.

Usage:
    uv run python run.py experiments/mnist/dense.py              # auto-detect
    uv run python run.py --mode torchrun experiments/mnist/dense.py
    uv run python run.py --mode mps experiments/mnist/dense.py
    uv run python run.py --mode cpu experiments/mnist/dense.py
    uv run python run.py --mode debug experiments/mnist/dense.py
"""
import argparse
import os
import sys
import subprocess
import resource
from dataclasses import dataclass


@dataclass
class RunConfig:
    """Configuration for a run mode"""
    device: str
    runner: list[str]  # Command prefix
    env_vars: dict[str, str]


def get_mode_config(mode: str, nproc: int) -> RunConfig:
    """Get configuration for each mode"""
    configs = {
        "torchrun": RunConfig(
            device="cuda",
            runner=[
                "torchrun",
                "--nnodes=1",
                f"--nproc_per_node={nproc}",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=localhost:0",
            ],
            env_vars={"EP_TORCHRUN": "1"},
        ),
        "cuda": RunConfig(
            device="cuda",
            runner=[sys.executable],
            env_vars={},
        ),
        "mps": RunConfig(
            device="mps",
            runner=[sys.executable],
            env_vars={},
        ),
        "cpu": RunConfig(
            device="cpu",
            runner=[sys.executable],
            env_vars={},
        ),
        "debug": RunConfig(
            device="cuda",
            runner=[sys.executable, "-m", "ipdb"],
            env_vars={},
        ),
    }
    return configs[mode]


def detect_default_mode() -> str:
    """Auto-detect the best mode based on available hardware"""
    try:
        import torch
        if torch.cuda.is_available():
            return "torchrun"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments",
        usage="%(prog)s [--mode MODE] script.py [script_args...]",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "torchrun", "cuda", "mps", "cpu", "debug"],
        default="auto",
        help="Run mode (default: auto-detect)",
    )
    parser.add_argument(
        "--nproc", "-n",
        type=int,
        default=1,
        help="Number of processes for torchrun (default: 1)",
    )
    parser.add_argument(
        "script",
        help="Python script to run",
    )
    parser.add_argument(
        "script_args",
        nargs="*",
        help="Arguments to pass to the script",
    )

    args = parser.parse_args()

    # Resolve auto mode
    mode = args.mode if args.mode != "auto" else detect_default_mode()
    config = get_mode_config(mode, args.nproc)

    # Set file descriptor limit
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(64000, hard), hard))
    except (ValueError, resource.error):
        pass

    # Build environment
    env = os.environ.copy()
    env["TORCH_DEVICE"] = config.device
    env["PYTHONBREAKPOINT"] = "ipdb.set_trace"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(config.env_vars)

    # Build command
    cmd = config.runner + [args.script] + args.script_args

    # Print info
    print(f"[run] Mode: {mode}, Device: {config.device}")
    print(f"[run] {' '.join(cmd)}")
    print()

    # Run
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
