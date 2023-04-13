#!/usr/bin/env python
import os
import torch
import torchmetrics as tm
import time
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable
from typing import List
from copy import deepcopy


@dataclass
class DataConfig:
    input_shape: torch.Size
    output_shape: torch.Size
    batch_size: int


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_config: DataConfig):
        self.n_samples = 100
        self.config = data_config
        self.x = torch.rand(torch.Size((self.n_samples, *data_config.input_shape)))
        self.y = torch.sin(self.x)
        self.sample_ids = torch.range(start=0, end=self.n_samples)

    def __getitem__(self, idx):
        input = self.x[idx]
        target = self.y[idx]
        sample_id = self.sample_ids[idx]
        return input, target, sample_id

    def __len__(self):
        return self.x.shape[0]


@dataclass
class DenseConfig:
    d_hidden: int = 300


class Dense(torch.nn.Module):
    def __init__(self, model_config: DenseConfig, data_config: DataConfig):
        super().__init__()
        self.config = model_config
        self.l1 = torch.nn.Linear(
            data_config.input_shape.numel(), model_config.d_hidden
        )
        self.l2 = torch.nn.Linear(
            model_config.d_hidden, data_config.output_shape.numel()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.l1(x)
        x = self.l2(x)
        return x


class ModelFactory:
    def __init__(self):
        self.models = dict()
        self.models[DenseConfig] = Dense

    def register(self, config_class, model_class):
        self.models[config_class] = model_class

    def create(self, model_config, data_config) -> torch.nn.Module:
        return self.models[model_config.__class__](model_config, data_config)


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


@dataclass
class TrainEpochState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    metrics: List[Metric]
    epoch: int


@dataclass
class TrainEpochSpec:
    loss: object
    dataloader: torch.utils.data.DataLoader
    device_id: int


def train(train_epoch_state: TrainEpochState, train_epoch_spec: TrainEpochSpec):
    model = train_epoch_state.model
    dataloader = train_epoch_spec.dataloader
    loss = train_epoch_spec.loss
    optimizer = train_epoch_state.optimizer
    device = train_epoch_spec.device_id

    model.train()

    for i, (input, target, sample_id) in enumerate(dataloader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input)

        loss_val = loss(output, target)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        metric_sample = MetricSample(
            prediction=output,
            target=target,
            sample_id=sample_id,
            epoch=train_epoch_state.epoch,
        )
        for metric in train_epoch_state.metrics:
            metric(metric_sample)

    train_epoch_state.epoch += 1


@dataclass
class TrainConfig:
    model_config: object
    data_config: object
    epochs: int
    metrics: List[Callable[[], Metric]]
    _version: int = 0


def serialize(train_epoch_state: TrainEpochState):
    serialized_metrics = [
        (metric.name(), metric.serialize()) for metric in train_epoch_state.metrics
    ]
    serialized_metrics = {name: value for name, value in serialized_metrics}
    data_dict = dict(
        model=train_epoch_state.model.state_dict(),
        optimizer=train_epoch_state.optimizer.state_dict(),
        epoch=train_epoch_state.epoch,
        metrics=serialized_metrics,
    )
    torch.save(data_dict, "checkpoint.pt")


def deserialize(
    model_factory: ModelFactory, train_config: TrainConfig, checkpoint_path, device_id
):
    data_dict = torch.load(checkpoint_path)

    model = model_factory.create(train_config.model_config, train_config.data_config)
    model = model.to(torch.device(device_id))
    model.load_state_dict(data_dict["model"])

    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(data_dict["optimizer"])

    metrics = []
    for metric in train_config.metrics:
        metric_instance = metric()
        metric_instance.deserialize(data_dict["metrics"][metric_instance.name()])
        metrics.append(metric_instance)

    epoch = data_dict["epoch"]

    return TrainEpochState(
        model=model, optimizer=optimizer, epoch=epoch, metrics=metrics
    )


def create_initial_state(train_config: TrainConfig, device_id):
    models = ModelFactory()
    model = models.create(train_config.model_config, train_config.data_config)
    model = model.to(torch.device(device_id))
    opt = torch.optim.AdamW(model.parameters())
    metrics = [metric() for metric in train_config.metrics]

    return TrainEpochState(
        model=model,
        optimizer=opt,
        metrics=metrics,
        epoch=0,
    )


def load_state(train_config: TrainConfig, checkpoint_path, device_id):
    models = ModelFactory()
    return deserialize(models, train_config, checkpoint_path, device_id)


def do_training(train_config: TrainConfig, state: TrainEpochState, device_id):
    ds = Dataset(train_config.data_config)
    dl = torch.utils.data.DataLoader(ds, batch_size=train_config.data_config.batch_size)

    train_epoch_spec = TrainEpochSpec(
        loss=torch.nn.MSELoss(),
        dataloader=dl,
        device_id=device_id,
    )

    while state.epoch < train_config.epochs:
        train(state, train_epoch_spec)
        serialize(state)
        time.sleep(1)
        print(f"Epoch {state.epoch} done")

        for metric in state.metrics:
            print(f"{metric.name()}: {metric.mean(state.epoch - 1):.04E}")


def main():
    if torch.cuda.is_available():
        device_id = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    else:
        device_id = "cpu"

    print(f"Using device {device_id}")

    train_config = TrainConfig(
        model_config=DenseConfig(d_hidden=100),
        data_config=DataConfig(
            torch.Size([1]), output_shape=torch.Size([1]), batch_size=2
        ),
        epochs=10,
        metrics=[
            lambda: Metric(tm.functional.mean_absolute_error),
            lambda: Metric(tm.functional.mean_squared_error),
        ],
    )
    checkpoint_path = Path("checkpoint.pt")

    if checkpoint_path.is_file():
        state = load_state(train_config, checkpoint_path, device_id)
    else:
        state = create_initial_state(train_config, device_id)

    do_training(train_config, state, device_id)


if __name__ == "__main__":
    main()
