import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.models.mlp import get_activation


@dataclass(frozen=True)
class GrokMLPConfig:
    width: int
    depth: int
    activation: str = "relu"

    def serialize_human(self):
        return self.__dict__


class GrokMLP(torch.nn.Module):
    def __init__(self, config: GrokMLPConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        in_dim = data_spec.input_shape.numel()
        out_dim = data_spec.output_shape[-1]
        activation = get_activation(config.activation)
        self.activation = activation

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_dim, config.width))
        for _ in range(config.depth - 1):
            self.layers.append(torch.nn.Linear(config.width, config.width))
        self.out = torch.nn.Linear(config.width, out_dim)

    def forward(self, batch):
        x = batch["input"]
        y = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            y = self.activation(layer(y))
        y = self.out(y)
        return dict(logits=y, predictions=torch.softmax(y.detach(), dim=-1))
