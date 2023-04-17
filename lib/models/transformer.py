import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass(frozen=True)
class TransformerConfig:
    embed_d: int
    mlp_dim: int
    n_seq: int
    batch_size: int


class Transformer(torch.nn.Module):
    def __init__(self, config: TransformerConfig, data_spec: DataSpec):
        super().__init__()
        embed_d = config.embed_d
        self.embed = torch.nn.Linear(1, embed_d, bias=True)
        lnorm = torch.nn.LayerNorm(embed_d, eps=1e-5)

        self.layer = torch.nn.TransformerDecoderLayer(
            d_model=embed_d,
            nhead=1,
            dim_feedforward=config.mlp_dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = torch.nn.TransformerDecoder(
            self.layer, num_layers=2, norm=lnorm
        )

        self.debed = torch.nn.Linear(embed_d, 1)
        self.register_buffer(
            "mem",
            torch.randn(
                (config.batch_size, config.n_seq, embed_d), requires_grad=False
            ),
        )

    def forward(self, x):
        embed = self.embed(x)
        tout = self.transformer(embed, self.mem)
        return torch.sigmoid(self.debed(tout))[:, 0, :]
