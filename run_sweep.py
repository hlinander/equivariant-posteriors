"""
SLURM sweep runner.

Submit mode (default): generate a SLURM batch script and submit via sbatch.
Worker mode (--worker): execute a single config by array index.

Usage:
    python run_sweep.py experiments/some_sweep.py              # submit
    python run_sweep.py --dry-run experiments/some_sweep.py    # print script only
    python run_sweep.py --run-local experiments/some_sweep.py  # run all locally
    python run_sweep.py --worker experiments/some_sweep.py 3   # run config 3
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from lib.slurm import SlurmConfig


def load_sweep_module(sweep_path: str):
    """Import a sweep file as a module."""
    spec = importlib.util.spec_from_file_location("sweep", sweep_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_slurm_config() -> SlurmConfig:
    """Load SlurmConfig from env.py, falling back to defaults."""
    try:
        import env

        if hasattr(env, "get_slurm_config"):
            return env.get_slurm_config()
    except ImportError:
        pass
    return SlurmConfig()


def generate_batch_script(
    sweep_path: str,
    num_configs: int,
    slurm: SlurmConfig,
    max_concurrent: int | None = None,
) -> str:
    """Generate a SLURM batch script string."""
    sweep_name = Path(sweep_path).stem

    array_spec = f"0-{num_configs - 1}"
    if max_concurrent is not None:
        array_spec += f"%{max_concurrent}"

    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={sweep_name}")
    lines.append(f"#SBATCH --array={array_spec}")
    lines.append(f"#SBATCH --time={slurm.time}")
    lines.append(f"#SBATCH --mem={slurm.mem}")
    lines.append(f"#SBATCH --cpus-per-task={slurm.cpus_per_task}")
    if slurm.gpus:
        lines.append(f"#SBATCH --gpus={slurm.gpus}")
    if slurm.partition:
        lines.append(f"#SBATCH --partition={slurm.partition}")
    if slurm.account:
        lines.append(f"#SBATCH --account={slurm.account}")
    lines.append(f"#SBATCH --output={slurm.output}")
    lines.append(f"#SBATCH --error={slurm.error}")
    for extra in slurm.extra_sbatch:
        lines.append(f"#SBATCH {extra}")

    lines.append("")

    for mod in slurm.modules:
        lines.append(f"module load {mod}")
    if slurm.modules:
        lines.append("")

    for cmd in slurm.setup_commands:
        lines.append(cmd)
    if slurm.setup_commands:
        lines.append("")

    lines.append(
        f"uv run python run_sweep.py --worker {sweep_path} $SLURM_ARRAY_TASK_ID"
    )

    return "\n".join(lines) + "\n"


def cmd_submit(sweep_path: str, dry_run: bool, max_concurrent: int | None):
    """Submit mode: generate batch script and submit via sbatch."""
    module = load_sweep_module(sweep_path)
    configs = module.create_configs()
    num_configs = len(configs)
    print(f"[sweep] Found {num_configs} configs in {sweep_path}")

    slurm = load_slurm_config()

    script = generate_batch_script(sweep_path, num_configs, slurm, max_concurrent)

    if dry_run:
        print("[sweep] Dry run — generated batch script:")
        print()
        print(script)
        return

    # Ensure log directory exists
    log_dir = Path(slurm.output).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    err_dir = Path(slurm.error).parent
    err_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["sbatch"],
        input=script,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


def cmd_worker(sweep_path: str, task_index: int):
    """Worker mode: run a single config by index."""
    module = load_sweep_module(sweep_path)
    configs = module.create_configs()
    if task_index < 0 or task_index >= len(configs):
        print(
            f"[sweep] Error: task index {task_index} out of range "
            f"(0-{len(configs) - 1})",
            file=sys.stderr,
        )
        sys.exit(1)
    config = configs[task_index]()
    module.run(config)


def cmd_run_local(sweep_path: str):
    """Run all configs sequentially without SLURM."""
    module = load_sweep_module(sweep_path)
    configs = module.create_configs()
    print(f"[sweep] Running {len(configs)} configs locally")
    for i, factory in enumerate(configs):
        print(f"[sweep] Config {i}/{len(configs) - 1}")
        config = factory()
        module.run(config)
    print("[sweep] Done")


def main():
    parser = argparse.ArgumentParser(
        description="SLURM sweep runner",
        usage=textwrap.dedent("""\
            %(prog)s [options] sweep_file.py
            %(prog)s --worker sweep_file.py INDEX"""),
    )
    parser.add_argument("--worker", action="store_true", help="Worker mode: run a single config by index")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated batch script without submitting")
    parser.add_argument("--run-local", action="store_true", help="Run all configs sequentially without SLURM")
    parser.add_argument("--max-concurrent", type=int, default=None, help="Limit SLURM array concurrency")
    parser.add_argument("sweep_file", help="Path to sweep file")
    parser.add_argument("task_index", nargs="?", type=int, help="Task index (worker mode only)")

    args = parser.parse_args()

    if args.worker:
        if args.task_index is None:
            parser.error("worker mode requires a task index")
        cmd_worker(args.sweep_file, args.task_index)
    elif args.run_local:
        cmd_run_local(args.sweep_file)
    else:
        cmd_submit(args.sweep_file, args.dry_run, args.max_concurrent)


if __name__ == "__main__":
    main()
