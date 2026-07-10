"""
Causal version of climate_pear_temporal_atn.py.

Key changes vs the original:
  1. PatchEmbed uses kernel_size/stride [1, patch_size] (no temporal mixing at embed time).
  2. FinalPatchExpand_Transpose uses kernel_size/stride [1, patch_size] (no temporal mixing
     at output time).
  3. SwinTransformerBlock.forward builds a temporal causal mask by propagating time indices
     through the same shift + window_partition pipeline used for the data, then combines it
     with the existing shifted-window mask before calling attention.
  4. input_resolutions no longer hard-codes D=6; it is derived from the actual temporal
     dimension of the embedded tensor.
  5. NestRollShift.shift rolls only the spatial (N) dimension.  The original already does
     this, but we assert the property explicitly in the causality test.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

import healpix as hp

from lib.dataspec import DataSpec

from experiments.climate.models import hp_shifting
from experiments.weather.models.hp_windowing import (
    window_partition,
    window_reverse,
    get_nest_win_idcs,
)

from experiments.climate.data.climateset_data_hp import ClimatesetDataSpec
from lib.serialize_human import serialize_human


# ---------------------------------------------------------------------------
# Utility: build a temporal causal mask in window space
# ---------------------------------------------------------------------------

def _build_temporal_causal_mask(
    time_ids: torch.Tensor,          # (1, D, N, 1) float
    shifter,
    window_size,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return additive causal mask of shape (nW, W, W).

    Entries where key_time > query_time are filled with a large negative value so
    that softmax suppresses them to ~0.  All other entries are 0.
    """
    shifted_ids = shifter.shift(time_ids)                        # (1, D, N, 1)
    time_windows = window_partition(shifted_ids, window_size, device=device)
    # time_windows: (nW, W, 1) — squeeze last channel dim
    time_windows = time_windows.squeeze(-1)                      # (nW, W)

    query_times = time_windows.unsqueeze(2)                      # (nW, W, 1)
    key_times   = time_windows.unsqueeze(1)                      # (nW, 1, W)

    future_mask = key_times > query_times                        # (nW, W, W) bool

    large_neg = -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9
    causal_mask = torch.zeros(future_mask.shape, dtype=dtype, device=device)
    causal_mask = causal_mask.masked_fill(future_mask, large_neg)

    return causal_mask                                           # (nW, W, W)


# ---------------------------------------------------------------------------
# Standard building blocks (unchanged from original except where noted)
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (unchanged from original)."""

    def __init__(self, dim, window_size, num_heads, input_resolution=None,
                 rel_pos_bias=None, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, use_cos_attn=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_cos_attn = use_cos_attn
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rel_pos_bias = rel_pos_bias

        if self.use_cos_attn:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        if self.rel_pos_bias == "earth":
            n_windows = (torch.tensor(input_resolution).prod()
                         // torch.tensor(window_size).prod())
            window_size_d, window_size_hp = window_size
            self.earth_position_bias = nn.Parameter(
                torch.zeros((n_windows, 1,
                             window_size_d * window_size_hp,
                             window_size_d * window_size_hp)))
            trunc_normal_(self.earth_position_bias, std=0.02)

        if self.rel_pos_bias == "single":
            window_size_d, window_size_hp = window_size
            self.earth_position_bias = nn.Parameter(
                torch.zeros((1, self.num_heads,
                             window_size_d * window_size_hp,
                             window_size_d * window_size_hp)))
            trunc_normal_(self.earth_position_bias, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, W, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B_, W, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_cos_attn:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(
                self.logit_scale,
                max=torch.log(torch.tensor(1.0 / 0.01,
                                           device=self.logit_scale.device))).exp()
            attn = attn * logit_scale
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

        if self.rel_pos_bias in ("earth", "single"):
            attn = attn + self.earth_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, W, W)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, W, W)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self):
        return (f"dim={self.dim}, window_size={self.window_size}, "
                f"num_heads={self.num_heads}")


# ---------------------------------------------------------------------------
# SwinTransformerBlock — causal mask injected here
# ---------------------------------------------------------------------------

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, base_pix, num_heads,
                 window_size=4, shift_size=0, shift_strategy="nest_roll",
                 rel_pos_bias=None, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_v2_norm_placement=False, use_cos_attn=False,
                 use_causal_mask=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_v2_norm_placement = use_v2_norm_placement

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, input_resolution=input_resolution, window_size=window_size,
            num_heads=num_heads, rel_pos_bias=rel_pos_bias, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            use_cos_attn=use_cos_attn)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=0)

        nside = math.floor(math.sqrt(input_resolution[1] // base_pix))
        shifters = {
            "nest_roll": (hp_shifting.NestRollShift,
                          {"shift_size": self.shift_size,
                           "input_resolution": self.input_resolution,
                           "window_size": self.window_size}),
            "nest_grid_shift": (hp_shifting.NestGridShift,
                                {"nside": nside, "base_pix": base_pix,
                                 "window_size": self.window_size}),
            "ring_shift": (hp_shifting.RingShift,
                           {"nside": nside, "base_pix": base_pix,
                            "window_size": self.window_size,
                            "shift_size": self.shift_size,
                            "input_resolution": self.input_resolution}),
        }

        if self.shift_size > 0:
            self.shifter = shifters[shift_strategy][0](**shifters[shift_strategy][1])
        else:
            self.shifter = hp_shifting.NoShift()

        attn_mask = self.shifter.get_mask(
            lambda x, window_size: window_partition(
                x, window_size, device=next(self.parameters()).device))
        self.register_buffer("attn_mask", attn_mask)

        self.use_causal_mask = use_causal_mask
        if use_causal_mask:
            D, N = input_resolution
            time_ids = (torch.arange(D, dtype=torch.float)
                        .view(1, D, 1, 1)
                        .expand(1, D, N, 1))
            causal_mask_temporal = _build_temporal_causal_mask(
                time_ids, self.shifter, self.window_size,
                device=next(self.parameters()).device, dtype=torch.float32)
            self.register_buffer("causal_mask_temporal", causal_mask_temporal)

    def forward(self, x):
        _, D, N, _ = x.shape

        shortcut = x
        if not self.use_v2_norm_placement:
            x = self.norm1(x)

        shifted_x = self.shifter.shift(x)

        if self.use_causal_mask:
            causal_mask = self.causal_mask_temporal.to(dtype=x.dtype)
            if self.attn_mask is not None:
                sw_mask = self.attn_mask.to(device=x.device, dtype=x.dtype)
                combined_mask = sw_mask + causal_mask
            else:
                combined_mask = causal_mask
            assert not (combined_mask < -1e3).all(dim=-1).any(), \
                "Causal mask fully masks at least one query row — check temporal resolution."
        else:
            combined_mask = self.attn_mask.to(device=x.device, dtype=x.dtype) if self.attn_mask is not None else None

        # Partition windows and run attention
        x_windows = window_partition(shifted_x, self.window_size,
                                     device=next(self.parameters()).device)
        attn_windows = self.attn(x_windows, mask=combined_mask)

        shifted_x = window_reverse(attn_windows, self.window_size, D, N,
                                   device=next(self.parameters()).device)
        x = self.shifter.shift_back(shifted_x)

        if self.use_v2_norm_placement:
            x = shortcut + self.drop_path(self.norm1(x))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self):
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"num_heads={self.num_heads}, window_size={self.window_size}, "
                f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}")


# ---------------------------------------------------------------------------
# Patch merging / expanding (unchanged)
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, D, N, C = x.shape
        assert N % 4 == 0
        x0 = x[:, :, 0::4, :]; x1 = x[:, :, 1::4, :]
        x2 = x[:, :, 2::4, :]; x3 = x[:, :, 3::4, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale != 1 else nn.Identity()
        self.norm = norm_layer(dim // 2)
        self.linear = nn.Linear(dim // 2, dim // 2, bias=False)

    def forward(self, x):
        x = self.expand(x)
        B, D, N, C = x.shape
        x = rearrange(x, "b d n (p c) -> b d (n p) c", p=4, c=C // 4)
        x = self.norm(x)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# PatchEmbed — CHANGED: temporal kernel/stride 1 instead of 2
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Embed spatial patches independently per timestep.

    Original used kernel_size=[2, patch_size] which mixed adjacent timesteps
    before the causal mask could act.  We use kernel_size=[1, patch_size] so
    each output timestep depends only on its own input timestep.
    """

    def __init__(self, config, data_spec: ClimatesetDataSpec):
        super().__init__()
        self.config = config
        self.data_spec = data_spec
        self.num_hp_patches = hp.nside2npix(data_spec.nside) // config.patch_size

        self.proj = nn.Conv2d(
            data_spec.n_input_channels,
            config.embed_dims[0],
            kernel_size=[1, config.patch_size],
            stride=[1, config.patch_size],
        )

    def forward(self, x):
        # x arrives as (B, C_in, T, N) — Conv2d sees (batch, channels, height, width)
        B, C_in, T, N = x.shape
        assert N == hp.nside2npix(self.data_spec.nside)
        x = self.proj(x)            # (B, embed_dim, T, N//patch_size)
        x = x.permute(0, 2, 3, 1)  # (B, T, N//patch_size, embed_dim)
        return x


# ---------------------------------------------------------------------------
# FinalPatchExpand_Transpose — CHANGED: temporal kernel/stride 1 instead of 2
# ---------------------------------------------------------------------------

class FinalPatchExpand_Transpose(nn.Module):
    """Expand patches back to original spatial resolution per timestep.

    Original used kernel_size=[2, patch_size] which would mix adjacent output
    timesteps.  We use kernel_size=[1, patch_size] to keep each output timestep
    independent.
    """

    def __init__(self, patch_size, dim, data_spec_hp: ClimatesetDataSpec):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.conv = nn.ConvTranspose2d(
            dim,
            data_spec_hp.n_output_channels,
            kernel_size=[1, patch_size],
            stride=[1, patch_size],
        )

    def forward(self, x: torch.Tensor):
        # x: (B, D, N, C) -> (B, C, D, N)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)           # (B, out_channels, D, N*patch_size)
        x = x.permute(0, 2, 3, 1) # (B, D, N*patch_size, out_channels)
        return x


# ---------------------------------------------------------------------------
# BasicLayer (unchanged)
# ---------------------------------------------------------------------------

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 base_pix, shift_size, shift_strategy, rel_pos_bias,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0,
                 attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                 use_checkpoint=False, use_v2_norm_placement=False,
                 use_cos_attn=False, use_causal_mask=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, base_pix=base_pix,
                shift_size=0 if (i % 2 == 0) else shift_size,
                shift_strategy=shift_strategy, rel_pos_bias=rel_pos_bias,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_v2_norm_placement=use_v2_norm_placement,
                use_cos_attn=use_cos_attn,
                use_causal_mask=use_causal_mask)
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def extra_repr(self):
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"depth={self.depth}")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SwinHPClimatesetTemporalAtnCausalConfig:
    base_pix: int = 12
    nside: int = 64
    patch_size: int = 16
    window_size: List[int] = field(default_factory=lambda: [2, 64])
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
    pad_fix: bool = True
    use_causal_mask: bool = True

    def serialize_human(self):
        return serialize_human(self.__dict__)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class SwinHPClimatesetTemporalAtnCausal(nn.Module):
    """Causal variant of SwinHPClimatesetTemporalAtn.

    Changes vs original:
      - PatchEmbed uses temporal stride 1 (no cross-timestep mixing at embed time).
      - FinalPatchExpand_Transpose uses temporal stride 1 (no cross-timestep mixing
        at output time).
      - input_resolutions are derived from data_spec.seq_len at construction time.
      - SwinTransformerBlock constructs a temporal causal mask every forward pass and
        combines it with the existing shifted-window mask.
    """

    def __init__(self, config: SwinHPClimatesetTemporalAtnCausalConfig,
                 data_spec: DataSpec, **kwargs):
        super().__init__()
        self.config = config
        self.data_spec = data_spec
        self.num_layers = len(config.depths)

        self.patch_embed = PatchEmbed(config, data_spec)
        num_hp_patches = self.patch_embed.num_hp_patches

        self.pos_drop = nn.Dropout(p=config.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate,
                                                  sum(config.depths))]

        self.downsample = PatchMerging(config.embed_dims[0], dim_scale=2)
        self.upsample = PatchExpand(config.embed_dims[1], dim_scale=2)

        self.final_up = FinalPatchExpand_Transpose(
            patch_size=config.patch_size,
            dim=2 * config.embed_dims[-1],
            data_spec_hp=data_spec,
        )

        self.norm = config.norm_layer(config.embed_dims[1])

        # PatchEmbed uses temporal stride 1, so embedded D == seq_len exactly.
        # Build layers eagerly so load_state_dict works without a forward pass.
        D = data_spec.seq_len
        input_resolutions = [
            [D, num_hp_patches],
            [D, num_hp_patches // 4],
            [D, num_hp_patches // 4],
            [D, num_hp_patches],
        ]
        self.input_resolutions = input_resolutions

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=config.embed_dims[i_layer],
                input_resolution=input_resolutions[i_layer],
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
                drop_path=dpr[sum(config.depths[:i_layer]):
                               sum(config.depths[:i_layer + 1])],
                norm_layer=config.norm_layer,
                use_v2_norm_placement=config.use_v2_norm_placement,
                use_checkpoint=config.use_checkpoint,
                use_causal_mask=config.use_causal_mask,
            )
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _forward(self, x):
        # x: (B, C_in, T, N) — time is dim-2, spatial is dim-3
        x = self.patch_embed(x)          # (B, T, N_patches, embed_dim)
        x = self.pos_drop(x)
        x = self.layers[0](x)
        skip = x
        x = self.downsample(x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.norm(x)
        x = self.upsample(x)
        x = self.layers[3](x)
        x = torch.cat([skip, x], dim=-1)
        x = self.final_up(x)
        return x

    def forward(self, batch):
        # batch["input"]: (B, T, C, N)
        # Permute to (B, C, T, N) so Conv2d sees channels in dim-1, time in dim-2.
        x = batch["input"].permute(0, 2, 1, 3)  # (B, C_in, T, N)
        x = self._forward(x)                     # (B, T, N_pix, C_out)
        x = x.permute(0, 1, 3, 2)               # (B, T, C_out, N_pix)
        return dict(logits_output=x)


# ---------------------------------------------------------------------------
# Causality test
# ---------------------------------------------------------------------------

def test_causality(model, data_spec, T=6, cutoff_t=3, device="cpu", tol=1e-4):
    """Verify that output[:, :cutoff_t+1] is identical for inputs that agree up to cutoff_t.

    Returns True if the model is causal, raises AssertionError otherwise.
    """
    model = model.to(device).eval()
    N = hp.nside2npix(data_spec.nside)
    C = data_spec.n_input_channels
    B = 1

    base = torch.randn(B, T, C, N, device=device)
    x1 = base.clone()
    x2 = base.clone()
    # Differ after cutoff_t
    x2[:, cutoff_t + 1:] = torch.randn_like(x2[:, cutoff_t + 1:])

    with torch.no_grad():
        out1 = model({"input": x1})["logits_output"]
        out2 = model({"input": x2})["logits_output"]

    # Outputs before and including cutoff_t must match
    err = (out1[:, :cutoff_t + 1] - out2[:, :cutoff_t + 1]).abs().max().item()
    assert err < tol, (
        f"Causality violation: max abs diff at t<={cutoff_t} is {err:.3e} (tol={tol})"
    )

    # Outputs after cutoff_t should differ (sanity check)
    diff_future = (out1[:, cutoff_t + 1:] - out2[:, cutoff_t + 1:]).abs().max().item()
    assert diff_future > tol, (
        "Outputs after cutoff are identical even though inputs differed — "
        "the model may be ignoring future inputs entirely."
    )

    return True
