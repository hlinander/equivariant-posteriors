import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.models.transformer import PositionalEncoding
import math


@dataclass(frozen=True)
class TransformerEncoderConfig:
    embed_d: int
    mlp_dim: int
    num_layers: int
    num_heads: int
    softmax: bool
    activation: str = "gelu"
    # norm_first=False + legacy_embed_scale=True reproduce
    # lib.models.transformer.Transformer (post-LN, sqrt(32) embedding scale
    # instead of sqrt(embed_d)).
    norm_first: bool = True
    legacy_embed_scale: bool = False

    def serialize_human(self):
        return self.__dict__


class TransformerEncoder(torch.nn.Module):
    """Standard self-attention transformer: stacked self-attention + MLP blocks.

    Unlike lib.models.transformer.Transformer (a TransformerDecoder that
    cross-attends to the raw input embedding in every layer), this is a plain
    nn.TransformerEncoder stack. Everything else (embedding scale, positional
    encoding, readout at position 0) matches Transformer.
    """

    def __init__(self, config: TransformerEncoderConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        embed_d = config.embed_d
        self.embed = torch.nn.Linear(data_spec.input_shape[-1], embed_d, bias=True)
        self.embed_scale = math.sqrt(32 if config.legacy_embed_scale else embed_d)
        self.pos_embed = PositionalEncoding(embed_d, dropout=0.0)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_d,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=0.0,
            batch_first=True,
            activation=config.activation,
            norm_first=config.norm_first,
        )
        self.transformer = torch.nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            norm=torch.nn.LayerNorm(embed_d, eps=1e-5),
            enable_nested_tensor=False,
        )
        self.debed = torch.nn.Linear(embed_d, data_spec.output_shape[-1])

    def forward(self, batch):
        return self.forward_tensor(batch["input"])

    def forward_tensor(self, x):
        embed = self.embed(x) * self.embed_scale
        embed = self.pos_embed(embed)
        tout = self.transformer(embed)
        output = self.debed(tout[:, 0, :])
        return dict(logits=output, predictions=self.output_to_value_detached(output))

    def output_to_value_detached(self, output):
        if self.config.softmax:
            return torch.softmax(output.detach(), dim=-1)
        else:
            return output.detach()
