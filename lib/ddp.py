from datetime import timedelta
import os
import torch


def ddp_setup() -> str:
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=10)
    )
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    return device_id
