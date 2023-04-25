from dataclasses import dataclass
from copy import deepcopy
from collections.abc import Callable
import torch


@dataclass
class MetricSample:
    prediction: torch.Tensor
    target: torch.Tensor
    sample_id: torch.Tensor
    epoch: int


@dataclass(frozen=True)
class MetricSampleKey:
    sample_id: int
    epoch: int


class Metric:
    # TODO: This should be tracked together with state and serialized/deserialized
    def __init__(
        self,
        metric_fn: Callable[[torch.Tensor, torch.Tensor], object],
        metric_kwargs=None,
    ):
        self.values = dict()
        self.metric_fn = metric_fn
        self.metric_name = metric_fn.__name__
        self.metric_kwargs = metric_kwargs if metric_kwargs is not None else dict()
        self.mean_mem = dict()

    def __call__(self, metric_sample: MetricSample):
        prediction = metric_sample.prediction
        target = metric_sample.target
        sample_id = metric_sample.sample_id
        values = (
            self.metric_fn(preds=prediction, target=target, **self.metric_kwargs)
            .detach()
            .cpu()
        )
        key = MetricSampleKey(sample_id=sample_id, epoch=metric_sample.epoch)
        self.values[key] = values.mean()

    def mean(self, epoch):
        if epoch not in self.mean_mem:
            keys = list(filter(lambda key: key.epoch == epoch, self.values.keys()))
            mean = None
            if len(keys) > 0:
                vals = torch.tensor([self.values[key] for key in keys])
                mean = torch.mean(vals).item()

            self.mean_mem[epoch] = mean
        return self.mean_mem[epoch]

    def serialize(self):
        return self.values

    def deserialize(self, values):
        self.values = values

    def name(self):
        return self.metric_name
