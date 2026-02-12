import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SlurmConfig:
    time: str = "2-00:00:00"
    mem: Optional[str] = None
    cpus_per_task: Optional[int] = None
    gpus: int = 1
    partition: Optional[str] = None
    account: Optional[str] = None
    output: str = "slurm/logs/%x.%A_%a.out"
    error: str = "slurm/logs/%x.%A_%a.err"
    extra_sbatch: list = field(default_factory=list)
    modules: list = field(default_factory=list)
    setup_commands: list = field(default_factory=list)


def generate_batch_script(
    job_name: str,
    slurm: SlurmConfig,
    run_command: str,
    array_spec: Optional[str] = None,
) -> str:
    """Generate a SLURM batch script string."""
    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    if array_spec:
        lines.append(f"#SBATCH --array={array_spec}")
    lines.append(f"#SBATCH --time={slurm.time}")
    if slurm.mem:
        lines.append(f"#SBATCH --mem={slurm.mem}")
    if slurm.cpus_per_task:
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

    lines.append(run_command)

    return "\n".join(lines) + "\n"


def submit_batch_script(script: str, slurm: SlurmConfig, dry_run: bool = False):
    """Submit a batch script via sbatch, or print it if dry_run."""
    import subprocess
    import sys
    from pathlib import Path

    if dry_run:
        print("[slurm] Dry run — generated batch script:")
        print()
        print(script)
        return

    # Ensure log directories exist
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


def load_slurm_config_from_env() -> "SlurmConfig":
    """Load SlurmConfig from env.py, falling back to defaults."""
    try:
        import env

        if hasattr(env, "get_slurm_config"):
            return env.get_slurm_config()
    except ImportError:
        pass
    return SlurmConfig()


def get_task_id() -> Optional[int]:
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        try:
            return int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        except ValueError:
            return None
    else:
        return None
        # raise Exception("SLURM_ARRAY_TASK_ID not set!")
