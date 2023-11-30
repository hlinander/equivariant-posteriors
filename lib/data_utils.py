from typing import Dict
import torch
from lib.train_dataclasses import ComputeConfig
from lib.metric import MetricSample
from lib.train_dataclasses import TrainEpochState


def get_sampler(
    compute_config: ComputeConfig, ds: torch.utils.data.DataLoader, shuffle: bool
) -> (torch.utils.data.Sampler, bool):
    """Get device compatible sampler.

    Distributed data parallell dataloader need distributed sampler,
    """
    if compute_config.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(ds)
        shuffle = False
    else:
        sampler = None
    return sampler, shuffle


def create_sample_legacy(input, target, sample_id):
    return dict(input=input, target=target, sample_id=sample_id)


def create_metric_sample_legacy(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    train_epoch_state: TrainEpochState,
):
    return MetricSample(
        output=output["logits"].detach(),
        prediction=output["predictions"].detach(),
        target=batch["target"].detach(),
        sample_id=batch["sample_id"].detach(),
        epoch=train_epoch_state.epoch,
    )
