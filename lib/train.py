from collections.abc import Callable
from dataclasses import dataclass
from typing import List
import time
import torch

from lib.metric import Metric
from lib.metric import MetricSample
from lib.model import ModelFactory
from lib.data import Dataset


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
