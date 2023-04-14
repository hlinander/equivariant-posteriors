import torch
from dataclasses import dataclass


@dataclass
class DataSpec:
    input_shape: torch.Size
    output_shape: torch.Size


@dataclass
class DataSineConfig:
    input_shape: torch.Size
    output_shape: torch.Size


class DataSine(torch.utils.data.Dataset):
    def __init__(self, data_config: DataSineConfig):
        self.n_samples = 100
        self.config = data_config
        self.x = torch.rand(torch.Size((self.n_samples, *data_config.input_shape)))
        self.y = torch.sin(self.x)
        self.sample_ids = torch.range(start=0, end=self.n_samples)

    def data_spec(self):
        return DataSpec(input_shape=self.x.shape[1:], output_shape=self.y.shape[1:])

    def __getitem__(self, idx):
        input = self.x[idx]
        target = self.y[idx]
        sample_id = self.sample_ids[idx]
        return input, target, sample_id

    def __len__(self):
        return self.x.shape[0]


@dataclass(frozen=True)
class DataSpiralsConfig:
    pass


class DataSpirals(torch.utils.data.Dataset):
    def __init__(self, _data_config: DataSpiralsConfig):
        N = 1000
        angles = 4 * 3 * torch.rand(N, 1)
        r = 1.0 + 0.1 * torch.randn(N, 1)
        xs1 = torch.stack(
            [
                r * angles / (4 * 3) * torch.cos(angles),
                r * angles / (4 * 3) * torch.sin(angles),
            ],
            dim=1,
        )
        ys1 = torch.zeros(N, 1)

        xs2 = torch.stack(
            [
                r * angles / (4 * 3) * torch.cos(angles + 3.14),
                r * angles / (4 * 3) * torch.sin(angles + 3.14),
            ],
            dim=1,
        )
        ys2 = torch.ones(N, 1)
        self.xs = torch.concat([xs1, xs2], dim=0)
        self.ys = torch.concat([ys1, ys2], dim=0)
        self.sample_ids = torch.arange(0, self.xs.shape[0], 1, dtype=torch.int32)

    def data_spec(self):
        return DataSpec(input_shape=self.xs.shape[1:], output_shape=self.ys.shape[1:])

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], self.sample_ids[idx]

    def __len__(self):
        return self.xs.shape[0]


class DataFactory:
    def __init__(self):
        self.datasets = dict()
        self.datasets[DataSpiralsConfig] = DataSpirals
        self.datasets[DataSineConfig] = DataSine

    def register(self, config_class, data_class):
        self.datasets[config_class] = data_class

    def create(self, data_config) -> torch.utils.data.Dataset:
        return self.datasets[data_config.__class__](data_config)
