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
import sys
import textwrap
from pathlib import Path

from lib.slurm import (
    SlurmConfig,
    generate_batch_script,
    submit_batch_script,
    load_slurm_config_from_env,
)


def load_sweep_module(sweep_path: str):
    """Import a sweep file as a module."""
    spec = importlib.util.spec_from_file_location("sweep", sweep_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_slurm_config(sweep_module) -> SlurmConfig:
    """Load SlurmConfig from sweep module, falling back to env.py, then defaults."""
    if hasattr(sweep_module, "get_slurm_config"):
        return sweep_module.get_slurm_config()
    return load_slurm_config_from_env()


def cmd_submit(sweep_path: str, dry_run: bool, max_concurrent: int | None):
    """Submit mode: generate batch script and submit via sbatch."""
    module = load_sweep_module(sweep_path)
    configs = module.create_configs()
    num_configs = len(configs)
    print(f"[sweep] Found {num_configs} configs in {sweep_path}")

    slurm = load_slurm_config(module)

    array_spec = f"0-{num_configs - 1}"
    if max_concurrent is not None:
        array_spec += f"%{max_concurrent}"

    script = generate_batch_script(
        job_name=Path(sweep_path).stem,
        slurm=slurm,
        run_command=f"uv run python run.py run_slurm_sweep.py --worker {sweep_path} $SLURM_ARRAY_TASK_ID",
        array_spec=array_spec,
    )

    submit_batch_script(script, slurm, dry_run)


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
    os.environ["EP_DEBUG"] = "1"
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
        usage=textwrap.dedent(
            """\
            %(prog)s [options] sweep_file.py
            %(prog)s --worker sweep_file.py INDEX"""
        ),
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Worker mode: run a single config by index",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated batch script without submitting",
    )
    parser.add_argument(
        "--run-local",
        action="store_true",
        help="Run all configs sequentially without SLURM",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None, help="Limit SLURM array concurrency"
    )
    parser.add_argument("sweep_file", help="Path to sweep file")
    parser.add_argument(
        "task_index", nargs="?", type=int, help="Task index (worker mode only)"
    )

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
