import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass
class DataSineConfig:
    input_shape: torch.Size
    output_shape: torch.Size

    def serialize_human(self, factories):
        return self.__dict__


class DataSine(torch.utils.data.Dataset):
    def __init__(self, data_config: DataSineConfig):
        self.n_samples = 100
        self.config = data_config
        self.x = torch.rand(torch.Size((self.n_samples, *data_config.input_shape)))
        self.y = torch.sin(self.x)
        self.sample_ids = torch.range(start=0, end=self.n_samples)

    def data_spec(self):
        return DataSpec(
            input_shape=self.x.shape[1:],
            output_shape=self.y.shape[1:],
            target_shape=self.y.shape[1:],
        )

    def __getitem__(self, idx):
        input = self.x[idx]
        target = self.y[idx]
        sample_id = self.sample_ids[idx]
        return input, target, sample_id

    def __len__(self):
        return self.x.shape[0]
