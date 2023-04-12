#!/usr/bin/env python
import os
import torch
import time
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataConfig:
    input_shape: torch.Size
    output_shape: torch.Size
    batch_size: int


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_config: DataConfig):
        self.config = data_config
        self.x = torch.rand(torch.Size((1000, *data_config.input_shape)))
        self.y = torch.sin(self.x)

    def __getitem__(self, idx):
        input = self.x[idx]
        target = self.y[idx]
        return input, target

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
class Metric:
    pass


@dataclass
class TrainEpochState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
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

    for i, (input, target) in enumerate(dataloader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input)

        loss_val = loss(output, target)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    train_epoch_state.epoch += 1


@dataclass
class TrainConfig:
    model_config: object
    data_config: object
    epochs: int
    _version: int = 0


def serialize(train_epoch_state: TrainEpochState):
    data_dict = dict(
        model=train_epoch_state.model.state_dict(),
        optimizer=train_epoch_state.optimizer.state_dict(),
        epoch=train_epoch_state.epoch,
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

    epoch = data_dict["epoch"]

    return TrainEpochState(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
    )


def create_initial_state(train_config: TrainConfig, device_id):
    models = ModelFactory()
    model = models.create(train_config.model_config, train_config.data_config)
    model = model.to(torch.device(device_id))
    opt = torch.optim.AdamW(model.parameters())

    return TrainEpochState(
        model=model,
        optimizer=opt,
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
        epochs=2,
    )
    checkpoint_path = Path("checkpoint.pt")

    if checkpoint_path.is_file():
        state = load_state(train_config, checkpoint_path, device_id)
    else:
        state = create_initial_state(train_config, device_id)

    do_training(train_config, state, device_id)


if __name__ == "__main__":
    main()
