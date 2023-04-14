from collections.abc import Callable
from dataclasses import dataclass
from typing import List
from pathlib import Path
import torch
import plotext as plt
import shutil

from lib.metric import Metric
from lib.metric import MetricSample
from lib.model import ModelFactory
from lib.data import DataFactory
from lib.stable_hash import stable_hash


@dataclass
class TrainEpochState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    metrics: List[Metric]
    dataloader: torch.utils.data.DataLoader
    epoch: int


@dataclass
class TrainEpochSpec:
    loss: object
    device_id: int


def train(train_epoch_state: TrainEpochState, train_epoch_spec: TrainEpochSpec):
    model = train_epoch_state.model
    dataloader = train_epoch_state.dataloader
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
        # print(f"{loss_val.detach().cpu()}")

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
    loss: torch.nn.Module
    batch_size: int
    ensemble_id: int = 0
    _version: int = 0


@dataclass
class TrainEval:
    metrics: List[Callable[[], Metric]]


@dataclass
class TrainRun:
    train_config: TrainConfig
    train_eval: TrainEval
    epochs: int


def get_checkpoint_path(train_config):
    config_hash = stable_hash(train_config)
    checkpoint_dir = Path("checkpoints/")
    checkpoint_dir.mkdir(exist_ok=True)
    tmp_checkpoint = checkpoint_dir / f"_checkpoint_{config_hash}.pt"
    checkpoint = checkpoint_dir / f"checkpoint_{config_hash}.pt"
    return checkpoint, tmp_checkpoint


def serialize(train_config: TrainConfig, train_epoch_state: TrainEpochState):
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

    checkpoint, tmp_checkpoint = get_checkpoint_path(train_config)
    torch.save(data_dict, tmp_checkpoint)
    shutil.move(tmp_checkpoint, checkpoint)


@dataclass
class DeserializeConfig:
    model_factory: ModelFactory
    data_factory: DataFactory
    train_run: TrainRun
    device_id: torch.device


def deserialize(config: DeserializeConfig):
    train_config = config.train_run.train_config
    checkpoint_path, _ = get_checkpoint_path(train_config)
    if not checkpoint_path.is_file():
        return None
    else:
        print(f"{checkpoint_path}")

    data_dict = torch.load(checkpoint_path)

    ds = config.data_factory.create(train_config.data_config)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = config.model_factory.create(
        train_config.model_config,
        train_config.data_config,
    )
    model = model.to(torch.device(config.device_id))
    model.load_state_dict(data_dict["model"])

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(data_dict["optimizer"])

    metrics = []
    for metric in config.train_run.train_eval.metrics:
        metric_instance = metric()
        metric_instance.deserialize(data_dict["metrics"][metric_instance.name()])
        metrics.append(metric_instance)

    epoch = data_dict["epoch"]

    return TrainEpochState(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics=metrics,
        dataloader=dataloader,
    )


def create_initial_state(
    models: ModelFactory, datasets: DataFactory, train_run: TrainRun, device_id
):
    train_config = train_run.train_config
    ds = datasets.create(train_config.data_config)
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=train_config.batch_size, shuffle=True, drop_last=True
    )

    model = models.create(train_config.model_config, train_config.data_config)
    model = model.to(torch.device(device_id))
    opt = torch.optim.Adam(model.parameters())
    metrics = [metric() for metric in train_run.train_eval.metrics]

    return TrainEpochState(
        model=model, optimizer=opt, metrics=metrics, epoch=0, dataloader=dataloader
    )


def load_or_create_state(train_run: TrainRun, device_id):
    models = ModelFactory()
    datasets = DataFactory()

    config = DeserializeConfig(
        model_factory=models,
        data_factory=datasets,
        train_run=train_run,
        device_id=device_id,
    )

    state = deserialize(config)

    if state is None:
        state = create_initial_state(
            models=models,
            datasets=datasets,
            train_run=config.train_run,
            device_id=config.device_id,
        )

    return state


def do_training(train_run: TrainRun, state: TrainEpochState, device_id):
    train_epoch_spec = TrainEpochSpec(
        loss=train_run.train_config.loss,
        device_id=device_id,
    )

    plt.title(state.metrics[0].name())
    while state.epoch < train_run.epochs:
        train(state, train_epoch_spec)
        serialize(train_run.train_config, state)
        # time.sleep(1)
        # print(f"Epoch {state.epoch} done")

        # for metric in state.metrics:
        #     print(f"{metric.name()}: {metric.mean(state.epoch - 1):.04E}")
        plt.clt()
        plt.cld()
        epochs = list(range(state.epoch))
        means = [state.metrics[0].mean(epoch) for epoch in epochs]
        plt.plot(epochs, means)
        plt.show()
