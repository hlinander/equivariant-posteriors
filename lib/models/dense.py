import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass
class DenseConfig:
    d_hidden: int = 300


class Dense(torch.nn.Module):
    def __init__(self, model_config: DenseConfig, data_spec: DataSpec):
        super().__init__()
        self.config = model_config
        self.l1 = torch.nn.Linear(data_spec.input_shape.numel(), model_config.d_hidden)
        self.l2 = torch.nn.Linear(model_config.d_hidden, data_spec.output_shape.numel())

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.l1(x)
        x = self.l2(x)
        return x
