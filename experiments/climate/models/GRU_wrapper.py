# Input: (B, T=12, C=4, N)
# For each t, run the existing model, collect outputs, then aggregate
import math
from dataclasses import dataclass, field
from typing import Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import healpix as hp

from lib.dataspec import DataSpec

#from experiments.weather.models import hp_shifting
from experiments.climate.models import hp_shifting # EDITED! FOR TESTING WITH D=1

from experiments.weather.models.hp_windowing import (
    window_partition,
    window_reverse,
    get_nest_win_idcs,
)

from experiments.climate.data.climateset_data_hp import ClimatesetDataSpec

from lib.serialize_human import serialize_human
from experiments.climate.models.swin_hp_climateset import SwinHPClimatesetConfig, SwinHPClimateset

@dataclass
class GRUTemporalWrapperConfig:
    backbone_config: SwinHPClimatesetConfig
    hidden_size: int = 64
    num_layers: int = 1
    bidirectional: bool = True

    def serialize_human(self):
        return serialize_human(self.__dict__)


class GRUTemporalWrapper(nn.Module):
    def __init__(self, config: GRUTemporalWrapperConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.backbone = SwinHPClimateset(config.backbone_config, data_spec)
        
        self.n_out = data_spec.n_output_channels  # 2
        directions = 2 if config.bidirectional else 1

        self.gru = nn.GRU(
            input_size=self.n_out,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        self.proj = nn.Linear(config.hidden_size * directions, self.n_out)

    def forward(self, batch):
        x = batch["input"]        # (B, T, C, N)
        B, T, C, N = x.shape

        outputs = []
        for t in range(T):
            out = self.backbone({"input": x[:, t]})["logits_output"]
            outputs.append(out)   # (B, N, 2)

        x = torch.stack(outputs, dim=2)   # (B, N, T, 2)
        x = x.reshape(B * N, T, 2)
        x, _ = self.gru(x)                # (B*N, T, hidden*directions)
        x = self.proj(x)                  # (B*N, T, 2)
        x = x.reshape(B, N, T, 2)        # (B, N, T, 2)
        x = x.permute(0, 2, 3, 1)        # (B, T, 2, N)
        return {"logits_output": x}