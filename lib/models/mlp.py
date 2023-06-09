import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass(frozen=True)
class MLPClassConfig:
    width: int

    def serialize_human(self, factories):
        return self.__dict__


class MLPClass(torch.nn.Module):
    def __init__(self, config: MLPClassConfig, data_spec: DataSpec):
        super().__init__()
        width = config.width
        self.mlp1 = torch.nn.Linear(2, width, bias=True)
        self.mlp2 = torch.nn.Linear(width, width, bias=True)
        self.mlp3 = torch.nn.Linear(width, 1, bias=True)

    def forward(self, x):
        y = x.reshape(x.shape[0], -1)
        y = self.mlp1(y)
        y = torch.nn.functional.gelu(y)
        y = self.mlp2(y)
        y = torch.nn.functional.gelu(y)
        y = self.mlp3(y)
        tout = y
        # tout = tout.reshape(x.shape[0], 2, -1)
        return torch.sigmoid(tout)
