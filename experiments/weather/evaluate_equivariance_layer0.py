"""
Evaluate approximate rotational equivariance of the first BasicLayer (layers[0])
extracted from trained model configs, using pretrained weights when available
(falls back to random init with a warning if no checkpoint is found).

Pipeline: raw pixel input → model.patch_embed → layers[0] → ConvTranspose decode
→ pixel-space output.  I/O has the same shape as the real model input (nside=64).

All configs share identical layer-0 dimensions:
  embed_dims[0] = 48,  D = 8,  N_patches = 3072 (nside_eff = 16)

Variants:
  1. pear_equiv_ds              → SwinHPPanguPad        (pad windowing, no conv)
  2. pear_isolatitude_ring_shift → SwinHPPanguIsolatitude (iso-lat windowing, no conv)
  3. conv_pear_isolatitude       → SimpleConvPearIsolatitude (iso-lat + simple HEALPix conv)
  4. convnext_pear_isolatitude   → CvTPearIsolatitudeV2      (iso-lat + ConvNeXt before attn)

Usage:
    uv run python run.py experiments/weather/evaluate_equivariance_layer0.py
    uv run python run.py experiments/weather/evaluate_equivariance_layer0.py --use-synthetic
"""

import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import healpy as hp_lib
from dataclasses import replace
from pathlib import Path

from lib.ddp import ddp_setup
from lib.serialization import deserialize_model, instantiate_model, DeserializeConfig
from experiments.weather.metrics import equivariance_error

# ── shared dims (all configs) ────────────────────────────────────────────────

NSIDE      = 64
PATCH_SIZE = 16
NPIX       = hp_lib.nside2npix(NSIDE)                  # 49152
N_SURFACE  = 4
N_UPPER    = 5
N_LEVELS   = 13
C          = 48    # embed_dims[0] = 192 // 4
D          = 8     # input_resolutions[0][0]

SENSITIVITY  = 120   # 3° steps
N_SYNTH      = 1
N_SEEDS      = 3
OPTIMISED    = True
TARGET_EPOCH = 0


# ── wrapper ───────────────────────────────────────────────────────────────────

class PatchEmbedLayerModel(nn.Module):
    """Raw pixel input → patch_embed → layers[0] → ConvTranspose → pixel output.

    Input:
      batch["input_surface"]: [B, N_SURFACE, Npix]
      batch["input_upper"]:   [B, N_UPPER, L, Npix]
    Output:
      logits_surface:         [B, N_SURFACE, Npix]
      logits_upper:           [B, N_UPPER, L, Npix]

    The decode head (ConvTranspose layers) is freshly initialised and is
    equivariant by construction — equivariance error comes from patch_embed
    and layers[0].
    """

    def __init__(
        self,
        patch_embed: nn.Module,
        layer: nn.Module,
        embed_dim: int,
        patch_size: int,
        n_surface: int,
        n_upper: int,
        n_levels: int,
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.layer = layer
        self.n_levels = n_levels
        self.conv_surface = nn.ConvTranspose1d(
            embed_dim, n_surface, kernel_size=patch_size, stride=patch_size
        )
        self.conv_upper = nn.ConvTranspose2d(
            embed_dim, n_upper, kernel_size=[2, patch_size], stride=[2, patch_size]
        )

    def forward(self, batch):
        x_s = batch["input_surface"]            # [B, C_s, Npix]
        x_u = batch["input_upper"]              # [B, C_u, L, Npix]
        x = self.patch_embed(x_s, x_u)          # [B, D, N_patches, C]
        x = self.layer(x)                       # [B, D, N_patches, C]
        x = x.permute(0, 3, 1, 2)              # [B, C, D, N_patches]
        out_surface = self.conv_surface(x[:, :, 0, :])      # [B, n_surface, Npix]
        out_upper   = self.conv_upper(x[:, :, 1:, :])       # [B, n_upper, ?, Npix]
        out_upper   = out_upper[:, :, :self.n_levels, :]    # crop to n_levels
        return dict(logits_surface=out_surface, logits_upper=out_upper)


# ── model loading helpers ─────────────────────────────────────────────────────

def _load_layer0(train_run, device: torch.device, label: str, only_transformer: bool = False) -> PatchEmbedLayerModel:
    """Load full model (pretrained or random) and return wrapped patch_embed + layers[0]."""
    train_run = replace(train_run, epochs=TARGET_EPOCH)
    cfg = DeserializeConfig(train_run=train_run, device_id=str(device))
    deserialized = deserialize_model(cfg, latest_ok=False)
    if deserialized is not None:
        model = deserialized.model
        print(f"  [{label}] loaded checkpoint  epoch={deserialized.epoch}")
    else:
        model = instantiate_model(train_run.train_config).to(device)
        print(f"  [{label}] WARNING: no checkpoint found — using random init")
    patch_embed = model.patch_embed.to(device)
    layer = model.layers[0].to(device)
    if only_transformer:
        layer = layer.blocks[0]
    return PatchEmbedLayerModel(
        patch_embed, layer,
        embed_dim=C, patch_size=PATCH_SIZE,
        n_surface=N_SURFACE, n_upper=N_UPPER, n_levels=N_LEVELS,
    )





# ── config factories ──────────────────────────────────────────────────────────

def _pear_equiv(device):
    from experiments.weather.persisted_configs.equivariant_ds.pear_equiv_ds import (
        create_config,
    )
    return _load_layer0(create_config(0), device, "pear_equiv_ds (ring_shift)")


def _pear_isolatitude(device):
    from experiments.weather.persisted_configs.equivariant_ds.pear_isolatitude_ring_shift_equiv_ds import (
        create_config,
    )
    return _load_layer0(create_config(0), device, "pear_isolatitude (ring shift)")


def _conv_pear(device):
    from experiments.weather.persisted_configs.equivariant_ds.conv_pear_isolatitude_1block_3x3_ds import (
        create_config,
    )
    return _load_layer0(create_config(0), device, "conv_pear (simple conv + isolat)")

def _conv_pear_no_conv_embedding(device):
    from experiments.weather.persisted_configs.equivariant_ds.conv_pear_isolatitude_1block_3x3_ds_no_conv_embedding import (
        create_config,
    )
    return _load_layer0(create_config(0), device, "conv_pear_no_conv_embedding (simple conv + isolat)")


def _convnext_pear(device):
    from experiments.weather.persisted_configs.equivariant_ds.convnext_pear_isolatitude_1block_3x3_ds import (
        create_config,
    )
    return _load_layer0(create_config(0), device, "convnext_pear (ConvNeXt + isolat)")

def _convnext_pear_no_conv_embedding(device):
    from experiments.weather.persisted_configs.equivariant_ds.convnext_pear_isolatitude_1block_3x3_ds_no_conv_embedding import (
        create_config,
    )
    return _load_layer0(create_config(0), device, "convnext_pear_no_conv_embedding (ConvNeXt + isolat)")

def _pear_equiv_only_transformer(device):
    from experiments.weather.persisted_configs.equivariant_ds.pear_equiv_ds import (
        create_config,
    )
    return _load_layer0(create_config(0), device, "pear_equiv_ds (ring_shift, only transformer)", only_transformer=True)

def _pear_isolatitude_only_transformer(device):
    from experiments.weather.persisted_configs.equivariant_ds.pear_isolatitude_ring_shift_equiv_ds import (
        create_config,
    )
    return _load_layer0(create_config(0), device, "pear_isolatitude (ring shift, only transformer)", only_transformer=True)

def _identity(device):
    """Identity mapping (for sanity check). Not a real config."""
    class IdentityModel(nn.Module):
        def forward(self, batch):
            return dict(
                logits_surface=batch["input_surface"],
                logits_upper=batch["input_upper"],
            )
    return IdentityModel().to(device)


# ── main ──────────────────────────────────────────────────────────────────────

def _make_val_dl(n_synth: int, seed: int = 42):
    """Sample n_synth items from the validation set with a fixed seed."""
    from experiments.weather.persisted_configs.equivariant_ds.pear_equiv_ds import create_config
    from experiments.weather.data import DataHP
    train_run = create_config(0)
    ds_config = train_run.train_config.train_data_config.validation()
    hp_config = getattr(ds_config, "base", ds_config)
    ds = DataHP(hp_config)
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(len(ds), generator=rng)[:n_synth].tolist()
    subset = torch.utils.data.Subset(ds, indices)
    return torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False, collate_fn=DataHP.collate_fn
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=TARGET_EPOCH)
    parser.add_argument("--optimised", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-synthetic", action="store_true", default=False,
                        help="Use random Gaussian input instead of real validation samples.")
    parser.add_argument("--only-transformer", action="store_true", default=False,
                        help="Evaluate only the transformer block (bypass patch_embed) to isolate its contribution to equivariance error.")

    args = parser.parse_args()

    TARGET_EPOCH = args.epoch
    OPTIMISED = args.optimised
    device = ddp_setup()

    if args.use_synthetic:
        torch.manual_seed(999)
        eval_dl = [
            {
                "input_surface": torch.randn(1, N_SURFACE, NPIX),
                "input_upper":   torch.randn(1, N_UPPER, N_LEVELS, NPIX),
            }
            for _ in range(N_SYNTH)
        ]
        data_label = "random Gaussian input"
    else:
        eval_dl = _make_val_dl(N_SYNTH, seed=42)
        data_label = "validation samples (fixed seed=42)"

    if args.only_transformer:
        print("Evaluating only the transformer block (bypassing patch_embed)...")
        variants = [
            ("pear (only transformer)",               _pear_equiv_only_transformer),
            ("pear isolatitude (only transformer)",   _pear_isolatitude_only_transformer),
            ("identity (sanity check)",               _identity),
        ]
    else:
        variants = [
            ("pear (pad)",               _pear_equiv),
            ("pear isolatitude",         _pear_isolatitude),
            ("isolatitude + Conv",       _conv_pear),
            ("isolatitude + ConvNeXt",   _convnext_pear),
            ("isolatitude + Conv (no conv embedding)", _conv_pear_no_conv_embedding),
            ("isolatitude + ConvNeXt (no conv embedding)", _convnext_pear_no_conv_embedding),
        ]

    def _mean_ee(ee):
        return {a: ee.surface[a].mean().item() for a in ee.surface}

    print("\nLoading models...")
    results = {}
    for label, factory in variants:
        print(f"\n=== {label} ===")
        angle_vals = {}
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            model = factory(device).to(device)
            model.eval()
            ee = equivariance_error(
                model, eval_dl, device=device,
                sensitivity=SENSITIVITY, max_batches=N_SYNTH, optimised=OPTIMISED,
            )
            for a, v in _mean_ee(ee).items():
                angle_vals.setdefault(a, []).append(v)
        results[label] = {
            a: (
                sum(vs) / len(vs),
                (sum((v - sum(vs) / len(vs)) ** 2 for v in vs) / len(vs)) ** 0.5,
            )
            for a, vs in angle_vals.items()
        }

    # ── plot ──────────────────────────────────────────────────────────────────
    angles = sorted(next(iter(results.values())).keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"patch_embed + BasicLayer-0 equivariance  ({data_label}, {N_SEEDS} seeds)\n"
        f"nside={NSIDE},  D={D},  C={C},  sensitivity={SENSITIVITY}°,  "
        f"n_samples={N_SYNTH}",
        fontsize=10, fontweight="bold",
    )

    for label, res in results.items():
        means = [res[a][0] for a in angles]
        stds  = [res[a][1] for a in angles]
        line, = ax.plot(angles, means, marker="o", markersize=4, linewidth=1.8, label=label)
        ax.fill_between(
            angles,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=line.get_color(),
        )

    ax.set_xlabel("Rotation angle (°)", fontsize=10)
    ax.set_ylabel("Mean equivariance error (pixel space)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    plot_path = Path("experiments/weather/plots/equivariance_error_layer0/")
    plot_path.mkdir(parents=True, exist_ok=True)
    if args.only_transformer:
        out_path = plot_path / f"equivariance_error_layer0_transformer_only_ep_{TARGET_EPOCH}_optimised_{OPTIMISED}_synthetic_{args.use_synthetic}.png"
    else:
        out_path = plot_path / f"equivariance_error_layer0_ep_{TARGET_EPOCH}_optimised_{OPTIMISED}_synthetic_{args.use_synthetic}.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
