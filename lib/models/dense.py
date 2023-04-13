import torch
from dataclasses import dataclass

from lib.data import DataConfig


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
