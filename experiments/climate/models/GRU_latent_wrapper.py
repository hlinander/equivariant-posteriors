# Level 2 temporal wrapper: GRU on per-pixel latent features (d-dim) instead of
# the backbone's 2-channel output predictions.
#
# Data flow:
#   x (B, T, C, N)
#     → for each t: FeatureExtractingSwinHP.forward_features(x[:,t])
#       → full Swin UNet encoder-decoder, but final projection is to latent_dim
#         instead of n_out=2
#       → (B, N_pix, latent_dim)
#     → stack  (B, N, T, latent_dim)
#     → reshape (B*N, T, latent_dim)
#     → GRU    (B*N, T, hidden*dirs)
#     → proj   (B*N, T, n_out)
#     → (B, T, n_out, N)

from dataclasses import dataclass
import torch
import torch.nn as nn

from lib.dataspec import DataSpec
from lib.serialize_human import serialize_human
from experiments.climate.models.swin_hp_climateset import (
    SwinHPClimatesetConfig,
    SwinHPClimateset,
)


class FeatureExtractingSwinHP(SwinHPClimateset):
    """Subclass that adds forward_features() — runs the same Swin UNet
    encoder-decoder but replaces the final ConvTranspose output head
    (which maps 384 → n_out at pixel resolution) with one that maps
    384 → latent_dim, exposing richer per-pixel representations."""

    def __init__(
        self,
        config: SwinHPClimatesetConfig,
        data_spec: DataSpec,
        latent_dim: int,
    ):
        super().__init__(config, data_spec)
        # 2 * embed_dims[-1] because the skip-connection is concatenated just
        # before final_up in _forward (see swin_hp_climateset.py).
        in_channels = 2 * config.embed_dims[-1]
        self.latent_proj = nn.ConvTranspose1d(
            in_channels,
            latent_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward_features(self, x_surface: torch.Tensor) -> torch.Tensor:
        """Return per-pixel latent features.

        Args:
            x_surface: (B, C_in, N_pix)

        Returns:
            (B, N_pix, latent_dim)
        """
        # Replicate _forward from SwinHPClimateset but stop before final_up.
        x = self.patch_embed(x_surface)   # (B, 1, N_patches, embed_dims[0])
        x = self.layers[0](x)
        skip = x
        x = self.downsample(x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.norm(x)
        x = self.upsample(x)
        x = self.layers[3](x)
        x = torch.concatenate([skip, x], dim=-1)  # (B, 1, N_patches, 2*embed_dims[-1])

        # Apply latent projection back to pixel resolution.
        x = x.permute(0, 3, 1, 2)                # (B, 2*embed_dims[-1], 1, N_patches)
        x = self.latent_proj(x[:, :, 0, :])      # (B, latent_dim, N_pix)
        x = x.permute(0, 2, 1)                   # (B, N_pix, latent_dim)
        return x


@dataclass
class GRULatentWrapperConfig:
    backbone_config: SwinHPClimatesetConfig
    # Dimensionality of per-pixel latent features fed into the GRU.
    # 192 matches embed_dims[0] (encoder output resolution).
    # 384 uses the full bottleneck width.
    latent_dim: int = 192
    hidden_size: int = 64
    num_layers: int = 1
    bidirectional: bool = True

    def serialize_human(self):
        return serialize_human(self.__dict__)


class GRULatentWrapper(nn.Module):
    def __init__(self, config: GRULatentWrapperConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.backbone = FeatureExtractingSwinHP(
            config.backbone_config, data_spec, config.latent_dim
        )

        self.n_out = data_spec.n_output_channels
        directions = 2 if config.bidirectional else 1

        self.gru = nn.GRU(
            input_size=config.latent_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        self.proj = nn.Linear(config.hidden_size * directions, self.n_out)

    def forward(self, batch):
        x = batch["input"]        # (B, T, C, N)
        B, T, C, N = x.shape

        features = []
        for t in range(T):
            feat = self.backbone.forward_features(x[:, t])  # (B, N, latent_dim)
            features.append(feat)

        x = torch.stack(features, dim=2)              # (B, N, T, latent_dim)
        x = x.reshape(B * N, T, self.config.latent_dim)
        x, _ = self.gru(x)                            # (B*N, T, hidden*dirs)
        x = self.proj(x)                              # (B*N, T, n_out)
        x = x.reshape(B, N, T, self.n_out)            # (B, N, T, n_out)
        x = x.permute(0, 2, 3, 1)                     # (B, T, n_out, N)
        return {"logits_output": x}
