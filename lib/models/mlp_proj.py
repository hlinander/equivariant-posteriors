import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from typing import List


@dataclass(frozen=True)
class MLPProjClassConfig:
    widths: List[int]
    store_layers: bool = False

    def serialize_human(self):
        return self.__dict__


class MLPProjClass(torch.nn.Module):
    def __init__(self, config: MLPProjClassConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.mlp_in = torch.nn.Linear(
            data_spec.input_shape.numel(), config.widths[0], bias=True
        )
        in_out = zip(config.widths[0:], config.widths[1:])
        self.mlps = torch.nn.ModuleList(
            [torch.nn.Linear(in_dim, out_dim, bias=True) for in_dim, out_dim in in_out]
        )
        self.mlp_out = torch.nn.Linear(
            config.widths[-1], data_spec.output_shape[-1], bias=True
        )

    def forward(self, x):
        y = x.reshape(x.shape[0], -1)
        y = self.mlp_in(y)
        y = torch.nn.functional.gelu(y)

        layers = []
        for idx, mlp in enumerate(self.mlps):
            y = mlp(y)
            if idx < len(self.mlps) - 1:
                y = torch.nn.functional.gelu(y)
            if self.config.store_layers:
                layers.append(y.detach())

        y = self.mlp_out(y)
        tout = y
        return dict(
            logits=tout, predictions=torch.softmax(tout.detach(), dim=-1), layers=layers
        )
