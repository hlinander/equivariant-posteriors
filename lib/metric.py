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
    def __init__(self, metric_fn: Callable[[torch.Tensor, torch.Tensor], object]):
        self.values = dict()
        self.metric_fn = metric_fn
        self.metric_name = metric_fn.__name__

    def __call__(self, metric_sample: MetricSample):
        for idx in range(metric_sample.sample_id.shape[0]):
            prediction = metric_sample.prediction[idx]
            target = metric_sample.target[idx]
            sample_id = metric_sample.sample_id[idx]
            value = self.metric_fn(preds=prediction, target=target).detach().cpu()
            key = MetricSampleKey(sample_id=sample_id, epoch=metric_sample.epoch)
            self.values[key] = value

    def mean(self, epoch):
        keys = filter(lambda key: key.epoch == epoch, self.values.keys())
        vals = torch.tensor([self.values[key] for key in keys])
        return torch.mean(vals)

    def serialize(self):
        return self.values

    def deserialize(self, values):
        self.values = deepcopy(values)

    def name(self):
        return self.metric_name
