import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from typing import List
from lib.models.mlp import MLPClass, MLPClassConfig


@dataclass(frozen=True)
class MLPProjClassConfig:
    mlp_config: MLPClassConfig
    n_proj: int

    def serialize_human(self):
        return self.__dict__


class MLPProjClass(torch.nn.Module):
    def __init__(self, config: MLPProjClassConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.mlp = MLPClass(config.mlp_config, data_spec)
        self.mlp_proj = torch.nn.Linear(10, config.n_proj, bias=True)
        self.mlp_out = torch.nn.Linear(config.n_proj, 10, bias=True)

    def forward(self, x):
        out = self.mlp(x)
        logits = self.mlp_out(self.mlp_proj(out["logits"]))
        return dict(logits=logits, predictions=torch.softmax(logits.detach(), dim=-1))
