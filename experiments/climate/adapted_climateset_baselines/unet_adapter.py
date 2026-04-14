"""
UNetAdapter — framework adapter for the ClimateSet UNet baseline.

The original UNet in baselines.py uses segmentation_models_pytorch (smp) which
is not available in this environment.  This file replaces smp.Unet with a
standard 4-level encoder-decoder UNet built from plain torch.nn primitives, so
no extra packages beyond PyTorch are required.

Architecture mirrors the smp.Unet(vgg11) contract:
  - Input : (B, C_in,  H, W)  — spatial dims must be divisible by 32
  - Output: (B, C_out, H, W)

Spatial compatibility (padding to multiples of 32) and the seq-length
time-distribution logic are handled in UNetAdapter, identical to the original
UNet in baselines.py.

To adapt a different baseline from baselines.py, copy this file, rename the
classes, and adjust __init__ / forward to match the target model's logic.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from experiments.climate.climateset_data_no_hp import ClimatesetDataSpec


# ---------------------------------------------------------------------------
# Primitive UNet blocks (pure torch.nn, no extra dependencies)
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Two consecutive Conv2d → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    """MaxPool2d(2) followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    """Transposed conv upsample, concatenate skip, then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # in_ch comes from the previous decoder level; skip has in_ch//2 channels
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)  # in_ch after concat with skip

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class _UNet(nn.Module):
    """
    Standard 4-level encoder-decoder UNet.

    Encoder feature widths: 64 → 128 → 256 → 512 → 1024 (bottleneck)
    Input spatial dims must be divisible by 32 (2^5 levels including bottleneck).
    """

    def __init__(self, in_channels: int, out_channels: int, base_features: int = 64):
        super().__init__()
        f = base_features
        # Encoder
        self.inc   = _DoubleConv(in_channels, f)
        self.down1 = _Down(f,     f * 2)
        self.down2 = _Down(f * 2, f * 4)
        self.down3 = _Down(f * 4, f * 8)
        self.down4 = _Down(f * 8, f * 16)
        # Decoder
        self.up1   = _Up(f * 16, f * 8)
        self.up2   = _Up(f * 8,  f * 4)
        self.up3   = _Up(f * 4,  f * 2)
        self.up4   = _Up(f * 2,  f)
        # Output projection
        self.outc  = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# TimeDistributed helper  (same logic as in baselines.py, no emulator import)
# ---------------------------------------------------------------------------

class _TimeDistributed(nn.Module):
    """Applies a module over the time (sequence) dimension identically per step.

    Input shape:  (batch, seq_len, *spatial_dims)
    Output shape: (batch, seq_len, *output_dims)
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len = x.shape[0], x.shape[1]
        out = self.module(x.reshape(bs * seq_len, *x.shape[2:]))
        return out.reshape(bs, seq_len, *out.shape[1:])

    def __repr__(self):
        return f"_TimeDistributed({self.module})"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class UNetAdapterConfig:
    """Configuration for UNetAdapter.

    Mirrors the constructor parameters of UNet in baselines.py so that the
    config is easy to read alongside the original.
    """
    readout: Literal["pooling", "linear"] = "pooling"
    # Base feature width of the UNet encoder (doubles at each level)
    base_features: int = 64
    # Spatial grid dimensions — 250 km ClimateSet grid: lat=96, lon=144
    longitude: int = 96
    latitude: int = 144
    seq_len: int = 1
    seq_to_seq: bool = True

    def serialize_human(self):
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class UNetAdapter(nn.Module):
    """
    Adapter wrapping a pure-PyTorch UNet for the equivariant-posteriors
    training framework.

    Implements the same spatial-padding / TimeDistributed / readout pipeline
    as UNet in models_climatesetrepo/baselines.py, but uses a built-in UNet
    instead of segmentation_models_pytorch (which is not installed).

    Expected batch dict (ClimatesetData with seq_len == 1):
        batch["input"]  : (B, C_in, H, W)
        batch["target"] : (B, C_out, H, W)

    For seq_len > 1:
        batch["input"]  : (B, T, C_in, H, W)
        batch["target"] : (B, T, C_out, H, W)

    Returns:
        {"logits_output": tensor matching batch["target"] shape}
    """

    def __init__(self, config: UNetAdapterConfig, data_spec: ClimatesetDataSpec):
        super().__init__()
        self.config = config
        self.lon = config.longitude
        self.lat = config.latitude
        self.seq_len = config.seq_len
        self.num_input_vars  = data_spec.n_input_channels
        self.num_output_vars = data_spec.n_output_channels

        # Padding so spatial dims are divisible by 32 (required by 4-level UNet).
        # Mirrors baselines.py UNet.__init__ padding logic exactly.
        pad_lon = int((np.ceil(self.lon / 32) * 32) - (self.lon / 32) * 32)
        pad_lat = int((np.ceil(self.lat / 32)) * 32 - (self.lat / 32) * 32)

        unet = _UNet(
            in_channels=self.num_input_vars,
            out_channels=self.num_output_vars,
            base_features=config.base_features,
        )

        if config.readout == "pooling":
            # ConstantPad2d pads last two dims: (left, right, top, bottom)
            # AdaptiveAvgPool3d pools (C_out, H+pad, W+pad) → (C_out, lon, lat)
            # treating the seq dimension as the "N channels" dim of Pool3d
            # (same trick as baselines.py).
            self.model = nn.Sequential(
                nn.ConstantPad2d((pad_lat, 0, pad_lon, 0), 0),
                _TimeDistributed(unet),
                nn.AdaptiveAvgPool3d(
                    output_size=(self.num_output_vars, self.lon, self.lat)
                ),
            )
        elif config.readout == "linear":
            padded_lon = self.lon + pad_lon
            padded_lat = self.lat + pad_lat
            self.model = nn.Sequential(
                nn.ConstantPad2d((pad_lat, 0, pad_lon, 0), 0),
                _TimeDistributed(unet),
                nn.Flatten(),
                nn.Linear(
                    in_features=self.num_output_vars * padded_lon * padded_lat * self.seq_len,
                    out_features=self.num_output_vars * self.lon * self.lat * self.seq_len,
                ),
            )
        else:
            raise ValueError(
                f"readout must be 'pooling' or 'linear', got '{config.readout}'"
            )

    def forward(self, batch: dict) -> dict:
        x = batch["input"]   # (B, C_in, H, W) or (B, T, C_in, H, W)

        # Add sequence dimension for single-timestep inputs so the UNet always
        # receives (B, T, C, H, W) through _TimeDistributed.
        squeeze_out = x.dim() == 4
        if squeeze_out:
            x = x.unsqueeze(1)   # (B, 1, C_in, H, W)

        x = self.model(x)   # (B, T, C_out, H, W)  [pooling] or flattened [linear]
        x = x.reshape(-1, self.seq_len, self.num_output_vars, self.lon, self.lat)
        x = x.nan_to_num()

        if not self.config.seq_to_seq:
            x = x[:, -1:, ...]   # keep only last timestep → (B, 1, C_out, H, W)

        if squeeze_out:
            x = x.squeeze(1)   # (B, C_out, H, W)

        return dict(logits_output=x)
