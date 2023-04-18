import torch
from dataclasses import dataclass


@dataclass
class DataSpec:
    input_shape: torch.Size
    output_shape: torch.Size
