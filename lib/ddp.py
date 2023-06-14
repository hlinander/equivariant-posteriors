from datetime import timedelta
import os
import torch


def ddp_setup(backend=None) -> str:
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    if "TORCH_DEVICE" in os.environ:
        device = os.environ["TORCH_DEVICE"]
        print(f"Using device {device}")
        return device

    if backend is None:
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"

    torch.distributed.init_process_group(
        backend=backend, init_method="env://", timeout=timedelta(seconds=10)
    )
    device_id = int(os.environ["LOCAL_RANK"])
    if backend == "nccl":
        torch.cuda.set_device(device_id)
        return device_id
    if backend == "gloo":
        return "cpu"

    raise Exception("Unsupported backend")
