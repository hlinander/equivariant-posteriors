import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SlurmConfig:
    time: str = "24:00:00"
    mem: str = "40G"
    cpus_per_task: int = 4
    gpus: int = 1
    partition: Optional[str] = None
    account: Optional[str] = None
    output: str = "slurm/logs/%x.%A_%a.out"
    error: str = "slurm/logs/%x.%A_%a.err"
    extra_sbatch: list = field(default_factory=list)
    modules: list = field(default_factory=list)
    setup_commands: list = field(default_factory=list)


def get_task_id() -> Optional[int]:
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        try:
            return int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        except ValueError:
            return None
    else:
        return None
        # raise Exception("SLURM_ARRAY_TASK_ID not set!")
