"""
Submit a single experiment to SLURM via sbatch.

Usage:
    python run_slurm.py experiments/some_experiment.py           # submit
    python run_slurm.py --dry-run experiments/some_experiment.py # print script only
    python run_slurm.py experiments/some_experiment.py --extra-arg value
"""

import argparse
import sys
from pathlib import Path

from lib.slurm import generate_batch_script, submit_batch_script, load_slurm_config_from_env


def main():
    parser = argparse.ArgumentParser(
        description="Submit a single experiment to SLURM",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated batch script without submitting",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="SLURM job name (default: script stem)",
    )
    parser.add_argument("script", help="Python script to run")

    args, remaining = parser.parse_known_args()

    slurm = load_slurm_config_from_env()
    job_name = args.job_name or Path(args.script).stem

    run_parts = ["uv run python run.py", args.script] + remaining
    run_command = " ".join(run_parts)

    script = generate_batch_script(
        job_name=job_name,
        slurm=slurm,
        run_command=run_command,
    )

    submit_batch_script(script, slurm, args.dry_run)


if __name__ == "__main__":
    main()
