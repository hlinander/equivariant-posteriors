"""
Decoupled tas/pr variants of SwinHPClimateset.

Three variants controlled by SwinHPClimatesetDecoupledConfig.decoupling:

  "dual_head"          (Variant A) — shared trunk + two independent final ConvTranspose1d heads.
                                     Least coupling reduction; makes variable separation explicit.

  "split_decoder"      (Variant B) — shared encoder + two independent (upsample, layer[3], head)
                                     branches. Recommended: each variable has its own upsampling
                                     path and decoder transformer, preventing gradient interference
                                     in the decoder.

  "split_decoder_norm" (Variant C) — same as B, but also two independent bottleneck LayerNorms.
                                     Prevents large activations in one variable's channels from
                                     rescaling the other variable's representation through shared
                                     mean/var statistics.

All variants return dict(logits_tas, logits_pr) with shapes (B, 1, N_pix) each, plus a
stacked logits_output=(B, 2, N_pix) for compatibility with existing loss functions.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from lib.dataspec import DataSpec
from experiments.climate.models.swin_hp_climateset import (
    SwinHPClimatesetConfig,
    PatchMerging,
    PatchExpand,
    BasicLayer,
    PatchEmbed,
)
from lib.serialize_human import serialize_human


class FinalPatchExpand_Transpose(nn.Module):
    """Single-variable output head: ConvTranspose1d mapping patch embeddings → pixel space."""

    def __init__(self, patch_size: int, dim: int, n_out_channels: int):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.conv_surface = nn.ConvTranspose1d(
            dim,
            n_out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, D, N_patches, C  →  B, N_pix, n_out_channels
        x = x.permute(0, 3, 1, 2)
        x_surface = self.conv_surface(x[:, :, 0, :])
        return x_surface.permute(0, 2, 1)


@dataclass
class SwinHPClimatesetDecoupledConfig(SwinHPClimatesetConfig):
    decoupling: Literal["dual_head", "split_decoder", "split_decoder_norm"] = "split_decoder"

    def serialize_human(self):
        return serialize_human(self.__dict__)


class SwinHPClimatesetDecoupled(nn.Module):
    """
    SwinHPClimateset with configurable tas/pr decoupling in the decoder.

    See module docstring for variant descriptions.
    """

    def __init__(self, config: SwinHPClimatesetDecoupledConfig, data_spec: DataSpec, **kwargs):
        super().__init__()
        self.config = config
        self.data_spec = data_spec

        self.patch_embed = PatchEmbed(config, data_spec)
        num_hp_patches = self.patch_embed.num_hp_patches

        input_resolutions = [
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

        # Shared encoder: layers 0, 1, 2
        self.encoder_layers = nn.ModuleList()
        for i in range(3):
            self.encoder_layers.append(
                BasicLayer(
                    dim=config.embed_dims[i],
                    input_resolution=input_resolutions[i],
                    depth=config.depths[i],
                    num_heads=config.num_heads[i],
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
                    drop_path=dpr[sum(config.depths[:i]) : sum(config.depths[: i + 1])],
                    norm_layer=config.norm_layer,
                    use_v2_norm_placement=config.use_v2_norm_placement,
                    use_checkpoint=config.use_checkpoint,
                )
            )

        # Decoder layer shared kwargs (layer index 3)
        dec_drop_path = dpr[sum(config.depths[:3]) : sum(config.depths[:4])]
        dec_kwargs = dict(
            dim=config.embed_dims[3],
            input_resolution=input_resolutions[3],
            depth=config.depths[3],
            num_heads=config.num_heads[3],
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
            drop_path=dec_drop_path,
            norm_layer=config.norm_layer,
            use_v2_norm_placement=config.use_v2_norm_placement,
            use_checkpoint=config.use_checkpoint,
        )

        final_up_dim = 2 * config.embed_dims[-1]

        if config.decoupling == "dual_head":
            # Variant A: shared decoder, two separate output convolutions
            self.norm = config.norm_layer(config.embed_dims[1])
            self.upsample = PatchExpand(config.embed_dims[1], dim_scale=2)
            self.decoder_layer = BasicLayer(**dec_kwargs)
            self.final_up_tas = FinalPatchExpand_Transpose(config.patch_size, final_up_dim, n_out_channels=1)
            self.final_up_pr = FinalPatchExpand_Transpose(config.patch_size, final_up_dim, n_out_channels=1)

        elif config.decoupling == "split_decoder":
            # Variant B: shared norm, independent (upsample, decoder layer, head) per variable
            self.norm = config.norm_layer(config.embed_dims[1])
            self.upsample_tas = PatchExpand(config.embed_dims[1], dim_scale=2)
            self.upsample_pr = PatchExpand(config.embed_dims[1], dim_scale=2)
            self.decoder_layer_tas = BasicLayer(**dec_kwargs)
            self.decoder_layer_pr = BasicLayer(**dec_kwargs)
            self.final_up_tas = FinalPatchExpand_Transpose(config.patch_size, final_up_dim, n_out_channels=1)
            self.final_up_pr = FinalPatchExpand_Transpose(config.patch_size, final_up_dim, n_out_channels=1)

        elif config.decoupling == "split_decoder_norm":
            # Variant C: independent bottleneck norms + independent (upsample, decoder layer, head)
            self.norm_tas = config.norm_layer(config.embed_dims[1])
            self.norm_pr = config.norm_layer(config.embed_dims[1])
            self.upsample_tas = PatchExpand(config.embed_dims[1], dim_scale=2)
            self.upsample_pr = PatchExpand(config.embed_dims[1], dim_scale=2)
            self.decoder_layer_tas = BasicLayer(**dec_kwargs)
            self.decoder_layer_pr = BasicLayer(**dec_kwargs)
            self.final_up_tas = FinalPatchExpand_Transpose(config.patch_size, final_up_dim, n_out_channels=1)
            self.final_up_pr = FinalPatchExpand_Transpose(config.patch_size, final_up_dim, n_out_channels=1)

        else:
            raise ValueError(f"Unknown decoupling mode: {config.decoupling!r}")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _forward(self, x_surface):
        # Shared encoder
        x = self.patch_embed(x_surface)   # B, 1, N_patches, C0
        x = self.encoder_layers[0](x)
        skip = x
        x = self.downsample(x)            # B, 1, N_patches//4, C1
        x = self.encoder_layers[1](x)
        x = self.encoder_layers[2](x)     # B, 1, N_patches//4, C1

        if self.config.decoupling == "dual_head":
            x = self.norm(x)
            x = self.upsample(x)          # B, 1, N_patches, C0
            x = self.decoder_layer(x)
            x = torch.concatenate([skip, x], dim=-1)   # B, 1, N_patches, 2*C0
            x_tas = self.final_up_tas(x)  # B, N_pix, 1
            x_pr = self.final_up_pr(x)    # B, N_pix, 1

        elif self.config.decoupling == "split_decoder":
            x = self.norm(x)
            # tas branch
            x_tas = self.upsample_tas(x)
            x_tas = self.decoder_layer_tas(x_tas)
            x_tas = torch.concatenate([skip, x_tas], dim=-1)
            x_tas = self.final_up_tas(x_tas)   # B, N_pix, 1
            # pr branch
            x_pr = self.upsample_pr(x)
            x_pr = self.decoder_layer_pr(x_pr)
            x_pr = torch.concatenate([skip, x_pr], dim=-1)
            x_pr = self.final_up_pr(x_pr)      # B, N_pix, 1

        else:  # split_decoder_norm
            # tas branch
            x_tas = self.norm_tas(x)
            x_tas = self.upsample_tas(x_tas)
            x_tas = self.decoder_layer_tas(x_tas)
            x_tas = torch.concatenate([skip, x_tas], dim=-1)
            x_tas = self.final_up_tas(x_tas)   # B, N_pix, 1
            # pr branch
            x_pr = self.norm_pr(x)
            x_pr = self.upsample_pr(x_pr)
            x_pr = self.decoder_layer_pr(x_pr)
            x_pr = torch.concatenate([skip, x_pr], dim=-1)
            x_pr = self.final_up_pr(x_pr)      # B, N_pix, 1

        return x_tas, x_pr

    def forward(self, batch):
        x_tas, x_pr = self._forward(batch["input"])
        # x_tas, x_pr: B, N_pix, 1  →  B, 1, N_pix
        x_tas = x_tas.permute(0, 2, 1)
        x_pr = x_pr.permute(0, 2, 1)
        return dict(
            logits_tas=x_tas,
            logits_pr=x_pr,
            # stacked for compatibility with loss fns that use output["logits_output"]
            logits_output=torch.cat([x_tas, x_pr], dim=1),
        )
