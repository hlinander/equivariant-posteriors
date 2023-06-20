import torch
from lib.train_dataclasses import TrainRun


def get_sampler(
    train_run: TrainRun, ds: torch.utils.data.DataLoader, shuffle: bool
) -> (torch.utils.data.Sampler, bool):
    """Get device compatible sampler.

    Distributed data parallell dataloader need distributed sampler,
    """
    if train_run.compute_config.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(ds)
        shuffle = False
    else:
        sampler = None
    return sampler, shuffle
