import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from typing import List
import math


@dataclass(frozen=True)
class MLPClassConfig:
    widths: List[int]

    def serialize_human(self):
        return self.__dict__


class MLPClass(torch.nn.Module):
    def __init__(self, config: MLPClassConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.mlp_in = torch.nn.Linear(
            data_spec.input_shape.numel(), config.widths[0], bias=True
        )
        in_out = zip(config.widths[0:], config.widths[1:])
        self.mlps = torch.nn.ModuleList(
            [torch.nn.Linear(in_dim, out_dim) for in_dim, out_dim in in_out]
        )
        self.mlp_out = torch.nn.Linear(
            config.widths[-1], data_spec.output_shape[-1], bias=True
        )
        for mlp in self.mlps:
            torch.nn.init.normal_(mlp.weight, 0.0, std=math.sqrt(1.0 / 20.0))
            torch.nn.init.normal_(mlp.bias, 0.0, std=math.sqrt(1e-7))

        torch.nn.init.normal_(self.mlp_in.weight, 0.0, math.sqrt(1.0 / (28 * 28)))
        torch.nn.init.normal_(self.mlp_in.bias, 0.0, std=math.sqrt(1e-7))
        torch.nn.init.normal_(self.mlp_out.weight, 0.0, math.sqrt(1.0 / 20.0))
        torch.nn.init.normal_(self.mlp_out.bias, 0.0, std=math.sqrt(1e-7))

    def forward(self, x):
        y = x.reshape(x.shape[0], -1)
        y = self.mlp_in(y)
        y = torch.nn.functional.tanh(y)

        for idx, mlp in enumerate(self.mlps):
            y = mlp(y)
            y = torch.nn.functional.tanh(y)

        y = self.mlp_out(y)
        tout = y
        return dict(logits=tout, predictions=torch.softmax(tout.detach(), dim=-1))
