from dataclasses import dataclass
from collections.abc import Callable
import torch


@dataclass
class MetricSample:
    batch: torch.Tensor
    output: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    sample_id: torch.Tensor
    epoch: int


@dataclass(frozen=True)
class MetricSampleKey:
    sample_id: int
    epoch: int


def key_from_metric_sample(metric_sample: MetricSample):
    return MetricSampleKey(sample_id=metric_sample.sample_id, epoch=metric_sample.epoch)


def epoch_filter(epoch):
    return lambda key: key.epoch == epoch


class MetricMeanStdStore:
    def __init__(self):
        self.means = dict()
        self.stds = dict()
        self.mean_mem = dict()

    def add(self, key, values):
        self.means[key] = values.mean()
        self.stds[key] = values.std()

    def mean(self, key_filter):
        keys = tuple(filter(key_filter, self.means.keys()))
        if keys not in self.mean_mem:
            mean = None
            if len(keys) > 0:
                vals = torch.tensor([self.means[key] for key in keys])
                mean = torch.mean(vals).item()

            self.mean_mem[keys] = mean
        return self.mean_mem[keys]

    def serialize(self):
        return (self.means, self.stds)

    def deserialize(self, values):
        self.means, self.stds = values

    def name(self):
        return self.metric_name


class MetricFn:
    def __init__(
        self,
        metric_fn: Callable[[torch.Tensor, torch.Tensor], object],
        metric_kwargs=None,
        raw_output=False,
    ):
        self.raw_output = raw_output
        self.metric_fn = metric_fn
        self.metric_name = metric_fn.__name__
        self.metric_kwargs = metric_kwargs if metric_kwargs is not None else dict()

    def __call__(
        self,
        metric_sample: MetricSample,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        output = metric_sample.output.detach()
        prediction = metric_sample.prediction.detach()
        target = metric_sample.target.detach()
        # sample_id = metric_sample.sample_id.detach()
        if self.raw_output:
            values = (
                self.metric_fn(preds=output, target=target, **self.metric_kwargs)
                .detach()
                .cpu()
            )
        else:
            values = (
                self.metric_fn(preds=prediction, target=target, **self.metric_kwargs)
                .detach()
                .cpu()
            )
        # key = MetricSampleKey(sample_id=sample_id, epoch=metric_sample.epoch)
        # self.values[key] = values.mean()
        return values

    def name(self):
        return self.metric_name


class Metric:
    def __init__(
        self,
        store,
        metric,
    ):
        self.store = store
        self.metric = metric

    def serialize(self):
        return self.store.serialize()

    def deserialize(self, values):
        self.store.deserialize(values)

    def name(self):
        return self.metric.name()

    def __call__(self, metric_sample: MetricSample, model: torch.nn.Module):
        values = self.metric(metric_sample=metric_sample, model=model)
        self.store.add(key=key_from_metric_sample(metric_sample), values=values)
