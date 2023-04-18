import torch
from dataclasses import dataclass
from typing import List
from typing import Callable

from lib.metric import Metric
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


@dataclass
class OptimizerConfig:
    optimizer: torch.optim.Optimizer
    kwargs: dict


@dataclass
class Factories:
    model_factory: ModelFactory = ModelFactory()
    data_factory: DataFactory = DataFactory()


@dataclass
class TrainConfig:
    model_config: object
    data_config: object
    loss: torch.nn.Module
    optimizer: OptimizerConfig
    batch_size: int
    ensemble_id: int = 0
    _version: int = 0

    def serialize_human(self, factories: Factories):
        return dict(
            model=dict(
                config=self.model_config.serialize_human(factories),
                name=factories.model_factory.get_class(self.model_config).__name__,
            ),
            data=dict(
                config=self.data_config.serialize_human(factories),
                name=factories.data_factory.get_class(self.data_config).__name__,
            ),
            loss=self.loss.__class__.__name__,
            optimizer=dict(
                name=self.optimizer.optimizer.__name__,
                kwargs=self.optimizer.kwargs,
            ),
        )

    def ensemble_dict(self):
        ensemble_config = self.__dict__.copy()
        ensemble_config.pop("ensemble_id")
        return ensemble_config


@dataclass
class TrainEval:
    metrics: List[Callable[[], Metric]]

    def serialize_human(self, factories: Factories):
        return [metric().name() for metric in self.metrics]


@dataclass
class TrainRun:
    train_config: TrainConfig
    train_eval: TrainEval
    epochs: int
    save_nth_epoch: int

    def serialize_human(self, factories: Factories):
        return dict(
            train_config=self.train_config.serialize_human(factories),
            train_id=stable_hash(self.train_config),
            ensemble_id=stable_hash(self.train_config.ensemble_dict()),
            train_eval=self.train_eval.serialize_human(factories),
            epochs=self.epochs,
            save_nth_epoch=self.save_nth_epoch,
        )
