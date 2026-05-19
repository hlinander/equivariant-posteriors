
"""
Evaluate approximate rotational equivariance of individual building blocks
in isolation — no PatchEmbed, no FinalPatchExpand, no real data.

Each block receives random Gaussian noise in token space [B, D, N_patches, C]
directly.  N_patches = hp.nside2npix(nside_eff) = 3072 (nside_eff = 16), so
healpy can rotate the token map at nside_eff just like a normal HEALPix map.

This completely removes the confound from the encoder/decoder layers and
measures only the architectural equivariance of the block itself.

Variants tested:
  - identity               : x → x  (should give ~0, sanity check)
  - isolatitude (no shift) : SwinTransformerBlock from swin_hp_pangu_isolatitude
  - pad (no shift)         : SwinTransformerBlock from swin_hp_pangu_pad
  - ConvNeXt block         : physicsnemo ConvNeXtBlock with HEALPixLayer geometry
  - HEALPix conv           : single HEALPixLayer(Conv2d) + GELU

Usage:
    uv run python run.py experiments/weather/evaluate_equivariance_swin_block.py
"""

import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import healpy as hp_lib

from lib.ddp import ddp_setup
from experiments.weather.models.swin_hp_pangu_isolatitude import (
    SwinTransformerBlock as SwinTransformerBlockIsolatitude,
    SwinHPPanguIsolatitudeConfig,
)
from experiments.weather.models.swin_hp_pangu_equivariant import (
    SwinTransformerBlockEquivariant,
)
from experiments.weather.models.swin_hp_pangu_pad import (
    SwinTransformerBlock as SwinTransformerBlockPad,
    SwinHPPanguPadConfig,
)
from physicsnemo.models.dlwp_healpix.layers.healpix_blocks import ConvNeXtBlock
from physicsnemo.nn.module.hpx.layers import HEALPixLayer
from experiments.weather.metrics import equivariance_error

# ── shared token dimensions ───────────────────────────────────────────────────

NSIDE       = 64
BASE_PIX    = 12
PATCH_SIZE  = 16
WINDOW_SIZE = [2, 64]
EMBED_DIM   = 192
NUM_HEADS   = 6
DEPTH       = 8   # D dimension expected by the Swin blocks

# nside_eff: HEALPix resolution of the token map after patching
N_PATCHES   = hp_lib.nside2npix(NSIDE) // PATCH_SIZE   # = 3072
NSIDE_EFF   = hp_lib.npix2nside(N_PATCHES)              # = 16

SENSITIVITY  = 120   # 3° steps  (360/120 = 3°)
N_SYNTH      = 20    # synthetic noise samples per seed
N_SEEDS      = 2     # random initialisations to average over

# ── token-space block wrappers ────────────────────────────────────────────────
# Input:  batch["input_surface"] = [B, D*C, N_patches]  (Gaussian noise)
#         batch["input_upper"]   = [B, 1, 1, N_patches]  (dummy zeros)
# Output: logits_surface         = [B, D*C, N_patches]
#         logits_upper           = [B, 1, 1, N_patches]  (zeros, ignored)
#
# equivariance_error() auto-detects nside_eff=16 from N_patches=3072 and
# precomputes rotations at that resolution — no extra changes needed.


class SwinTokenModel(nn.Module):
    """Wraps a SwinTransformerBlock for direct token-space EE measurement."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.D = DEPTH
        self.C = EMBED_DIM

    def forward(self, batch):
        x = batch["input_surface"]               # [B, D*C, N]
        B, DC, N = x.shape
        D, C = self.D, self.C
        x = x.reshape(B, D, C, N).permute(0, 1, 3, 2)   # [B, D, N, C]
        x = self.block(x)                                  # [B, D, N, C]
        x = x.permute(0, 1, 3, 2).reshape(B, D * C, N)   # [B, D*C, N]
        dummy = torch.zeros(B, 1, 1, N, device=x.device, dtype=x.dtype)
        return dict(logits_surface=x, logits_upper=dummy)


class ConvTokenModel(nn.Module):
    """Wraps a HEALPix conv block for direct token-space EE measurement.

    Uses the same _conv_pass reshape as BasicLayer so the block sees
    per-face spatial tensors [B*D*12, C, nside_eff, nside_eff].
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.D = DEPTH
        self.C = EMBED_DIM
        self.nside_eff = NSIDE_EFF

    def forward(self, batch):
        x = batch["input_surface"]               # [B, D*C, N]
        B, DC, N = x.shape
        D, C, nside = self.D, self.C, self.nside_eff
        x = x.reshape(B, D, C, N).permute(0, 1, 3, 2)   # [B, D, N, C]

        # _conv_pass: reshape to per-face spatial tensors, apply block, reshape back
        x_c = x.permute(0, 1, 3, 2).reshape(B, D, C, 12, nside, nside)
        x_c = x_c.permute(0, 1, 3, 2, 4, 5).reshape(B * D * 12, C, nside, nside)
        x_c = self.block(x_c)
        x_c = x_c.reshape(B, D, 12, C, nside, nside)
        x_c = x_c.permute(0, 1, 3, 2, 4, 5).reshape(B, D, C, N)
        x = x_c.permute(0, 1, 3, 2)                      # [B, D, N, C]

        x = x.permute(0, 1, 3, 2).reshape(B, D * C, N)   # [B, D*C, N]
        dummy = torch.zeros(B, 1, 1, N, device=x.device, dtype=x.dtype)
        return dict(logits_surface=x, logits_upper=dummy)


# ── block factory functions ───────────────────────────────────────────────────

def _input_res():
    return [DEPTH, N_PATCHES]


def make_identity() -> SwinTokenModel:
    return SwinTokenModel(nn.Identity())


def make_isolatitude() -> SwinTokenModel:
    block = SwinTransformerBlockIsolatitude(
        dim=EMBED_DIM, input_resolution=_input_res(), base_pix=BASE_PIX,
        num_heads=NUM_HEADS, window_size=WINDOW_SIZE, shift_size=0,
        shift_strategy="nest_roll", rel_pos_bias=None,
        drop=0.0, attn_drop=0.0, drop_path=0.0,
    )
    return SwinTokenModel(block)


def make_pad() -> SwinTokenModel:
    block = SwinTransformerBlockPad(
        dim=EMBED_DIM, input_resolution=_input_res(), base_pix=BASE_PIX,
        num_heads=NUM_HEADS, window_size=WINDOW_SIZE, shift_size=0,
        shift_strategy="nest_roll", rel_pos_bias=None,
        drop=0.0, attn_drop=0.0, drop_path=0.0,
    )
    return SwinTokenModel(block)


def make_equivariant() -> SwinTokenModel:
    block = SwinTransformerBlockEquivariant(
        dim=EMBED_DIM, input_resolution=_input_res(), base_pix=BASE_PIX,
        num_heads=NUM_HEADS, window_size=WINDOW_SIZE, shift_size=0,
        rel_pos_bias=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
    )
    return SwinTokenModel(block)


def make_convnext() -> ConvTokenModel:
    block = ConvNeXtBlock(
        geometry_layer=HEALPixLayer,
        in_channels=EMBED_DIM, latent_channels=EMBED_DIM, out_channels=EMBED_DIM,
        kernel_size=3, dilation=1, upscale_factor=4, activation=nn.GELU(),
    )
    return ConvTokenModel(block)


def make_healpix_conv() -> ConvTokenModel:
    block = nn.Sequential(
        HEALPixLayer(nn.Conv2d, in_channels=EMBED_DIM, out_channels=EMBED_DIM, kernel_size=3),
        nn.GELU(),
    )
    return ConvTokenModel(block)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = ddp_setup()

    # Synthetic token-space dataset — pure Gaussian noise, no real data needed.
    # shape: [1, D*C, N_patches] = [1, 1536, 3072]
    # equivariance_error detects nside_eff=16 from N_patches=3072 automatically.
    torch.manual_seed(999)
    synthetic_dl = [
        {
            "input_surface": torch.randn(1, DEPTH * EMBED_DIM, N_PATCHES),
            "input_upper":   torch.zeros(1, 1, 1, N_PATCHES),
        }
        for _ in range(N_SYNTH)
    ]

    variants = [
        ("identity",          make_identity),
        ("isolatitude",       make_isolatitude),
        ("pad",               make_pad),
        ("ConvNeXt block",    make_convnext),
        ("HEALPix conv",      make_healpix_conv),
    ]

    def _mean_ee_over_angles(ee):
        """Mean EE over all token channels → {angle: scalar}."""
        return {a: ee.surface[a].mean().item() for a in ee.surface}

    results = {}  # label -> list of {angle: scalar} (one per seed)
    for label, factory in variants:
        print(f"\n=== {label} ===")
        angle_vals = {}   # angle -> list of per-seed mean EE
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            model = factory().to(device)
            model.eval()
            ee = equivariance_error(
                model, synthetic_dl, device=device,
                sensitivity=SENSITIVITY, max_batches=N_SYNTH,
            )
            for a, v in _mean_ee_over_angles(ee).items():
                angle_vals.setdefault(a, []).append(v)
        results[label] = {a: (sum(vs)/len(vs), (sum((v - sum(vs)/len(vs))**2 for v in vs)/len(vs))**0.5)
                          for a, vs in angle_vals.items()}   # angle -> (mean, std)

    # ── plot ──────────────────────────────────────────────────────────────────
    angles = sorted(next(iter(results.values())).keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"Token-space block equivariance (random Gaussian input, {N_SEEDS} seeds)\n"
        f"nside_eff={NSIDE_EFF}, D={DEPTH}, C={EMBED_DIM}, window={WINDOW_SIZE}, "
        f"sensitivity={SENSITIVITY}°, n_samples={N_SYNTH}",
        fontsize=10, fontweight="bold",
    )

    for label, res in results.items():
        means = [res[a][0] for a in angles]
        stds  = [res[a][1] for a in angles]
        line, = ax.plot(angles, means, marker="o", markersize=4, linewidth=1.8, label=label)
        ax.fill_between(angles,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=line.get_color())

    ax.set_xlabel("Rotation angle (°)", fontsize=10)
    ax.set_ylabel("Mean equivariance error (token space)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = "equivariance_error_swin_block.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
