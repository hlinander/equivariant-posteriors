import os


def get_task_id():
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    else:
        return None
        # raise Exception("SLURM_ARRAY_TASK_ID not set!")
