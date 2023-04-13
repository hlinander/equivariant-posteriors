import torch
from dataclasses import dataclass


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
