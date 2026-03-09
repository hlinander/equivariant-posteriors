"""
Seq-to-seq variant of SwinHPClimateset for 12-month (annual) climate prediction.

Input:  batch["input"]  → (B, T, C_in,  N_pix)   T = seq_len (e.g. 12)
Output: logits_output   → (B, T, C_out, N_pix)

Architecture
------------
1. Flatten T into the batch dimension: (B*T, C_in, N_pix)
2. Run the shared SwinHP-UNet spatial backbone (same structure as baseline)
3. Restore temporal structure: (B, T, N_pix, C_out)
4. Apply a lightweight residual Temporal Mixing MLP over the T axis
5. Permute to target shape: (B, T, C_out, N_pix)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import healpix as hp

from lib.dataspec import DataSpec
from lib.serialize_human import serialize_human

from experiments.climate.models.swin_hp_climateset import (
    PatchEmbed,
    BasicLayer,
    PatchMerging,
    PatchExpand,
    FinalPatchExpand_Transpose,
)
from experiments.climate.climateset_data_hp import ClimatesetDataSpec


# ---------------------------------------------------------------------------
# Temporal mixing module
# ---------------------------------------------------------------------------

class TemporalMixingMLP(nn.Module):
    """
    Residual MLP that mixes information across the T (temporal) axis.

    Operates on tensors of shape (B, T, N_pix, C_out) by treating T as the
    feature dimension to be mixed via a small 2-layer MLP.
    """

    def __init__(self, seq_len: int, n_channels: int, hidden_mult: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(n_channels)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len * hidden_mult),
            nn.GELU(),
            nn.Linear(seq_len * hidden_mult, seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N_pix, C_out)
        residual = x
        x = self.norm(x)               # layer-norm over C_out
        x = x.permute(0, 2, 3, 1)     # (B, N_pix, C_out, T)
        x = self.mlp(x)               # mix over T
        x = x.permute(0, 3, 1, 2)     # (B, T, N_pix, C_out)
        return residual + x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SwinHPClimatesetSeqConfig:
    """Spatial backbone config for the seq-to-seq model (same knobs as baseline)."""
    base_pix: int = 12
    nside: int = 64
    patch_size: int = 16
    window_size: int = 36
    shift_size: int = 2
    shift_strategy: Literal["nest_roll", "nest_grid_shift", "ring_shift"] = "nest_roll"
    rel_pos_bias: Optional[Literal["flat"]] = None
    patch_embed_norm_layer: Optional[Literal[nn.LayerNorm]] = None
    depths: List[int] = field(default_factory=lambda: [2, 6, 6, 2])
    num_heads: List[int] = field(default_factory=lambda: [6, 12, 12, 6])
    embed_dims: List[int] = field(default_factory=lambda: [192, 384, 384, 192])
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    use_cos_attn: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Literal[nn.LayerNorm] = nn.LayerNorm
    use_v2_norm_placement: bool = False
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    dev_mode: bool = False

    def serialize_human(self):
        return serialize_human(self.__dict__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SwinHPClimatesetSeq(nn.Module):
    """
    Seq-to-seq SwinHP model for annual (12-month) climate prediction.

    The spatial SwinHP-UNet backbone is shared across all T timesteps
    (processed in parallel by flattening B*T into the batch dimension).
    A residual Temporal Mixing MLP is applied after decoding to allow
    information exchange across months before the final output.

    Data shapes
    -----------
    Input:  batch["input"]  (B, T, C_in,  N_pix)
    Output: logits_output   (B, T, C_out, N_pix)
    """

    def __init__(
        self,
        config: SwinHPClimatesetSeqConfig,
        data_spec: ClimatesetDataSpec,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.data_spec = data_spec
        self.seq_len = data_spec.seq_len
        self.num_layers = len(config.depths)

        # --- Spatial backbone (identical structure to SwinHPClimateset) ---
        self.patch_embed = PatchEmbed(config, data_spec)
        num_hp_patches = self.patch_embed.num_hp_patches

        self.input_resolutions = [
            [1, num_hp_patches],
            [1, num_hp_patches // 4],
            [1, num_hp_patches // 4],
            [1, num_hp_patches],
        ]

        self.pos_drop = nn.Dropout(p=config.drop_rate)

        dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))
        ]

        self.downsample = PatchMerging(config.embed_dims[0], dim_scale=2)
        self.upsample   = PatchExpand(config.embed_dims[1],  dim_scale=2)

        self.final_up = FinalPatchExpand_Transpose(
            patch_size=config.patch_size,
            dim=2 * config.embed_dims[-1],
            data_spec_hp=data_spec,
        )

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=config.embed_dims[i_layer],
                input_resolution=self.input_resolutions[i_layer],
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                window_size=config.window_size,
                base_pix=config.base_pix,
                shift_size=config.shift_size,
                shift_strategy=config.shift_strategy,
                rel_pos_bias=config.rel_pos_bias,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale,
                use_cos_attn=config.use_cos_attn,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[
                    sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])
                ],
                norm_layer=config.norm_layer,
                use_v2_norm_placement=config.use_v2_norm_placement,
                use_checkpoint=config.use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = config.norm_layer(config.embed_dims[1])

        # --- Temporal mixing (applied after spatial decode, before output) ---
        self.temporal_mix = TemporalMixingMLP(
            seq_len=self.seq_len,
            n_channels=data_spec.n_output_channels,
        )

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # Spatial pass (shared backbone)
    # ------------------------------------------------------------------

    def _forward_spatial(self, x_surface: torch.Tensor) -> torch.Tensor:
        """
        Run the SwinHP-UNet on a batch of single-timestep inputs.

        x_surface : (B, C_in, N_pix)   — B may equal original_B * T
        returns   : (B, N_pix, C_out)
        """
        x    = self.patch_embed(x_surface)   # (B, 1, N_patches, embed_dim)
        x    = self.layers[0](x)
        skip = x
        x    = self.downsample(x)
        x    = self.layers[1](x)
        x    = self.layers[2](x)
        x    = self.norm(x)
        x    = self.upsample(x)
        x    = self.layers[3](x)
        x    = torch.cat([skip, x], dim=-1)  # (B, 1, N_patches, 2*embed_dim)
        x    = self.final_up(x)              # (B, N_pix, C_out)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch):
        x = batch["input"]           # (B, T, C_in, N_pix)
        B, T, C, N = x.shape

        # Flatten T into batch for parallel spatial processing
        x_flat   = x.reshape(B * T, C, N)
        out_flat = self._forward_spatial(x_flat)   # (B*T, N_pix, C_out)

        N_pix = out_flat.shape[1]
        C_out = out_flat.shape[2]

        # Restore temporal dimension
        out = out_flat.reshape(B, T, N_pix, C_out)  # (B, T, N_pix, C_out)

        # Temporal mixing (residual)
        out = self.temporal_mix(out)                 # (B, T, N_pix, C_out)

        # Permute to match target layout (B, T, C_out, N_pix)
        out = out.permute(0, 1, 3, 2)

        return {"logits_output": out}
