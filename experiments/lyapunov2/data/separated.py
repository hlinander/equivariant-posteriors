# from typing import Dict
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy
import pandas as pd
from pathlib import Path

# from lib.train_dataclasses import TrainEpochState


@dataclass(frozen=True)
class DataSeparatedConfig:
    validation: bool = False

    def serialize_human(self):
        return dict()


@dataclass
class Separated:
    xs: torch.Tensor
    ys: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    sample_ids: torch.Tensor


def load_separated():
    base = Path(__file__).parent / "separated"
    xs_df = pd.read_csv(base / "X.csv")
    ys_df = pd.read_csv(base / "y.csv")
    xs = torch.tensor(xs_df.to_numpy(), dtype=torch.float32).reshape(-1, 2, 1)
    ys = torch.tensor(ys_df.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    mean = xs.mean(dim=0)
    std = xs.std(dim=0)
    xs = (xs - mean) / std
    sample_ids = torch.arange(0, xs.shape[0], 1, dtype=torch.int32)
    return Separated(xs=xs, ys=ys, sample_ids=sample_ids, mean=mean, std=std)


def generate_validation():
    train_data = load_separated()
    xgrid, ygrid = torch.meshgrid(
        torch.linspace(-2, 2, 500), torch.linspace(-2, 2, 500)
    )
    xy = torch.stack([xgrid, ygrid], 2)
    xy = xy.reshape(-1, 2, 1).float()
    xy = (xy - train_data.mean) / train_data.std
    y = torch.zeros(xy.shape[0]).float().reshape(-1, 1)
    sample_ids = torch.arange(0, xy.shape[0], 1, dtype=torch.int32)
    return Separated(
        xs=xy, ys=y, sample_ids=sample_ids, mean=train_data.mean, std=train_data.std
    )


class DataSeparated(torch.utils.data.Dataset):
    def __init__(self, data_config: DataSeparatedConfig):
        if data_config.validation:
            self.data = generate_validation()
        else:
            self.data = load_separated()

        self.n_classes = 2
        self.config = data_config

    @staticmethod
    def data_spec(config):
        return DataSpec(
            input_shape=torch.Size([2, 1]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([1]),
        )

    @staticmethod
    def sample_id_spec(config):
        return ["idx"]

    def __getitem__(self, idx):
        return create_sample_legacy(
            self.data.xs[idx], self.data.ys[idx], self.data.sample_ids[idx]
        )

    def __len__(self):
        return self.data.xs.shape[0]


@dataclass(frozen=True)
class DataSeparatedClassConfig:
    validation: bool = False

    def serialize_human(self):
        return dict()


def load_separated_class():
    base = Path(__file__).parent / "separated"
    xs_df = pd.read_csv(base / "X.csv")
    ys_df = pd.read_csv(base / "y.csv")
    xs = torch.tensor(xs_df.to_numpy(), dtype=torch.float32).reshape(-1, 2, 1)
    ys = torch.tensor(ys_df.to_numpy(), dtype=torch.float32).reshape(-1)
    ys = torch.where(ys == 1.0, 0, 1)
    # ys = torch.where(ys == 1.0, torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))
    sample_ids = torch.arange(0, xs.shape[0], 1, dtype=torch.int32)
    return Separated(xs, ys, sample_ids)


def generate_validation_class():
    xgrid, ygrid = torch.meshgrid(
        torch.linspace(-2, 2, 500), torch.linspace(-2, 2, 500)
    )
    xy = torch.stack([xgrid, ygrid], 2)
    xy = xy.reshape(-1, 2, 1).float()
    y = torch.zeros((xy.shape[0], 1)).long().reshape(-1)
    sample_ids = torch.arange(0, xy.shape[0], 1, dtype=torch.int32)
    return Separated(xy, y, sample_ids)


class DataSeparatedClass(torch.utils.data.Dataset):
    def __init__(self, data_config: DataSeparatedClassConfig):
        if data_config.validation:
            self.data = generate_validation_class()
        else:
            self.data = load_separated_class()

        self.n_classes = 2
        self.config = data_config

    @staticmethod
    def data_spec(config):
        return DataSpec(
            input_shape=torch.Size([2, 1]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([2]),
        )

    @staticmethod
    def sample_id_spec(config):
        return ["idx"]

    def __getitem__(self, idx):
        return create_sample_legacy(
            self.data.xs[idx], self.data.ys[idx], self.data.sample_ids[idx]
        )

    def __len__(self):
        return self.data.xs.shape[0]
