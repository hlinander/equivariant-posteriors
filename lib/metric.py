from dataclasses import dataclass
from collections.abc import Callable
from typing import Dict
import torch

# import lib.render_duck as duck


@dataclass
class MetricSample:
    # output: torch.Tensor
    # prediction: torch.Tensor
    # target: torch.Tensor
    output: Dict[str, torch.Tensor]
    batch: Dict[str, torch.Tensor]
    batch: int
    model_id: int


@dataclass(frozen=True)
class MetricSampleKey:
    batch_idx: int
    sample_id: int
    epoch: int


class Metric:
    def __init__(
        self,
        metric_fn: Callable[[torch.Tensor, torch.Tensor], object],
        metric_kwargs=None,
        name=None,
    ):
        self.metric_fn = metric_fn
        if name is None:
            self.metric_name = metric_fn.__name__
        else:
            self.metric_name = name
        self.metric_kwargs = metric_kwargs if metric_kwargs is not None else dict()

    def __call__(self, metric_sample: MetricSample):
        values = (
            self.metric_fn(
                output=metric_sample.output,
                batch=metric_sample.batch,
                **self.metric_kwargs
            )
            .detach()
            .cpu()
        )
        return values.mean().item()

    def per_sample(self, metric_sample: MetricSample):
        values = (
            self.metric_fn(
                output=metric_sample.output,
                batch=metric_sample.batch,
                **self.metric_kwargs
            )
            .detach()
            .cpu()
        )
        return values

    def name(self):
        return self.metric_name


def detach_tensors(output, batch):
    output = {k: v.detach() for k, v in output.items() if torch.is_tensor(v)}
    batch = {k: v.detach() for k, v in batch.items() if torch.is_tensor(v)}
    return dict(output=output, batch=batch)


def create_metric(metric_fn, name=None):
    if name is None:
        name = metric_fn.__name__

    def detached_fn(output, batch):
        return metric_fn(**detach_tensors(output, batch))

    return lambda: Metric(detached_fn, name=name)
