#!/usr/bin/env python3
"""
Verifies RingShift.shift() from the actual model code and visualises it.

Two modes:

  --mode synthetic  (default)
      Feeds synthetic pixel-index data (and optionally window / delta / disp)
      through the real RingShift.shift() object — same as synthetic_healpix_viewer
      but using the model's own shift code instead of the standalone helper.
      Shows 2 rows (post-patch | after-downsample) × 2 columns (original | shifted).

  --mode real  (requires --checkpoint)
      Loads the model, hooks the first shifted SwinTransformerBlock inside
      layers[0], and captures the activations BEFORE and AFTER the ring shift
      during a real forward pass.  Shows the same 2×2 layout on actual features.

In both modes the layout is:
  [original]  [after shift]
  [original]  [after shift]
  top row = post-patch nside, bottom row = after-downsample nside

Usage
-----
  # Synthetic data, climate defaults
  python climate_shift_verifier.py

  # Synthetic + cartesian projection
  python climate_shift_verifier.py --projection cart

  # All synthetic pixel modes
  python climate_shift_verifier.py --mode synthetic --pixel-mode window
  python climate_shift_verifier.py --mode synthetic --pixel-mode delta
  python climate_shift_verifier.py --mode synthetic --pixel-mode disp

  # Real model features
  python climate_shift_verifier.py --mode real \\
      --checkpoint /proj/heal_pangu/eqp_climate/checkpoints/checkpoint_9153213b2c7536b1/model_epoch_0200
"""

import sys
import json
import dataclasses
import argparse
from pathlib import Path

_ROOT  = Path(__file__).resolve().parents[2]
_LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LOCAL))

import numpy as np
import torch
import healpix as healpix_pkg
import healpy  as hp_lib
import chealpix as chp
import matplotlib.pyplot as plt

from models.hp_shifting import RingShift


# ── RingShift instantiation ───────────────────────────────────────────────────

def build_ring_shift(nside, window_size_hp, shift_size, D=1):
    N = 12 * nside ** 2
    return RingShift(
        nside=nside,
        base_pix=12,
        window_size=[D, window_size_hp],
        shift_size=shift_size,
        input_resolution=(D, N),
    )


# ── synthetic data makers ─────────────────────────────────────────────────────

def make_synthetic(npix, nside, mode, window_size):
    """Return a (npix,) float array for the requested pixel-value mode."""
    idx = np.arange(npix, dtype=float)
    if mode == "index":
        return idx
    if mode == "window":
        return (idx // window_size).astype(float)
    if mode == "delta":
        return idx   # reference; shift produces the delta later
    if mode == "disp":
        lon, _ = healpix_pkg.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
        return lon
    raise ValueError(f"Unknown pixel-mode {mode!r}")


def apply_model_shift(rs, data_1d):
    """
    Run data_1d (npix float array) through RingShift.shift() exactly as the
    model would.  Packs into (B=1, D=1, npix, C=1), calls shift(), unpacks.
    """
    x = torch.tensor(data_1d, dtype=torch.float32)[None, None, :, None]  # (1,1,N,1)
    xs = rs.shift(x)
    return x[0, 0, :, 0].numpy(), xs[0, 0, :, 0].numpy()


# ── projections ───────────────────────────────────────────────────────────────

def project_moll(data, nest=True, xsize=800):
    proj = hp_lib.mollview(data.astype(float), nest=nest,
                           return_projected_map=True, xsize=xsize)
    plt.close()
    return np.ma.filled(proj, np.nan)


def project_cart(data, nside, nest=True, n_lat=360, n_lon=720):
    lats = np.linspace(-90, 90, n_lat)
    lons = np.linspace(0, 360, n_lon, endpoint=False)
    lon_g, lat_g = np.meshgrid(lons, lats)
    pix = healpix_pkg.ang2pix(nside, lon_g.ravel(), lat_g.ravel(),
                               lonlat=True, nest=nest)
    return data.astype(float)[pix].reshape(n_lat, n_lon)


def to_proj(data, nside, projection):
    if projection == "moll":
        return project_moll(data)
    return project_cart(data, nside)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_panel(ax, proj, title, cmap="plasma", vmin=None, vmax=None):
    finite = proj[np.isfinite(proj)]
    if vmin is None:
        vmin = float(np.nanmin(finite)) if len(finite) else 0
    if vmax is None:
        vmax = float(np.nanmax(finite)) if len(finite) else 1
    im = ax.imshow(proj, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


def row_lims(*projs):
    vals = np.concatenate([p[np.isfinite(p)] for p in projs])
    return float(np.nanmin(vals)), float(np.nanmax(vals))


# ── checkpoint helpers (shared with inspect_forward_pass) ─────────────────────

def _unwrap(obj):
    if isinstance(obj, dict):
        if "__class__" in obj and "__data__" in obj:
            return {k: _unwrap(v) for k, v in obj["__data__"].items()}
        return {k: _unwrap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap(v) for v in obj]
    return obj


def read_cfg_from_ckpt(ckpt_path):
    search_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    json_path  = search_dir / "train_run.json"
    if not json_path.is_file():
        return {}
    try:
        raw  = json.loads(json_path.read_text())
        flat = _unwrap(raw)
        for v in flat.values():
            if isinstance(v, dict) and "model_config" in v:
                return v["model_config"]
    except Exception:
        pass
    return {}


def load_state_dict(path, device):
    if path.is_dir():
        f = path / "model"
    else:
        f = path
    raw = torch.load(f, map_location=device)
    if isinstance(raw, dict) and raw and all(k.startswith("module.") for k in raw):
        raw = {k[7:]: v for k, v in raw.items()}
    return raw


# ── real-model shift capture ──────────────────────────────────────────────────

class ShiftCapture:
    """
    Pre-forward hook on a SwinTransformerBlock: captures x before the ring
    shift, and computes block.shifter.shift(x) to get the shifted tensor.
    Only fires once (removes itself after the first call).
    """
    def __init__(self):
        self.before = None   # (B, D, N, C) cpu tensor
        self.after  = None
        self._handle = None

    def register(self, block):
        def _pre_hook(module, args):
            if self.before is not None:   # already captured
                return
            x = args[0].detach().cpu()
            self.before = x
            self.after  = module.shifter.shift(x).detach().cpu()
        self._handle = block.register_forward_pre_hook(_pre_hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Verify and visualise RingShift.shift() from the actual model code."
    )
    p.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    p.add_argument("--pixel-mode", default="index",
                   choices=["index", "window", "delta", "disp"],
                   help="(synthetic mode) What value to colour each pixel by.")
    p.add_argument("--nside",       type=int, default=32)
    p.add_argument("--patch-size",  type=int, default=16)
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--shift-size",  type=int, default=4)
    p.add_argument("--depth",       type=int, default=1,
                   help="Depth dimension D (1=climate, 2=weather).")
    p.add_argument("--projection",  default="moll", choices=["moll", "cart"])
    p.add_argument("--cmap",        default="plasma")
    p.add_argument("--feature-idx", type=int, default=0,
                   help="(real mode) Embedding feature index to visualise.")
    p.add_argument("--checkpoint",  default=None,
                   help="(real mode) Checkpoint file or directory.")
    p.add_argument("--sample-idx",  type=int, default=0,
                   help="(real mode) Dataset sample index.")
    p.add_argument("--split",       default="val",
                   choices=["train", "val", "test"])
    p.add_argument("--device",      default="cpu")
    p.add_argument("--out",         default=None)
    return p.parse_args()


# ── synthetic mode ────────────────────────────────────────────────────────────

def run_synthetic(args):
    sqrt_patch  = int(round(args.patch_size ** 0.5))
    assert sqrt_patch ** 2 == args.patch_size
    nside_patch = args.nside // sqrt_patch
    nside_down  = nside_patch // 2
    npix_patch  = 12 * nside_patch ** 2
    npix_down   = 12 * nside_down  ** 2

    print(f"Post-patch nside:       {nside_patch}  (npix={npix_patch})")
    print(f"After-downsample nside: {nside_down}  (npix={npix_down})")

    rs_p = build_ring_shift(nside_patch, args.window_size, args.shift_size, D=args.depth)
    rs_d = build_ring_shift(nside_down,  args.window_size, args.shift_size, D=args.depth)

    raw_p = make_synthetic(npix_patch, nside_patch, args.pixel_mode, args.window_size)
    raw_d = make_synthetic(npix_down,  nside_down,  args.pixel_mode, args.window_size)

    orig_p, shft_p = apply_model_shift(rs_p, raw_p)
    orig_d, shft_d = apply_model_shift(rs_d, raw_d)

    # For disp mode: convert to actual longitude displacement
    if args.pixel_mode == "disp":
        lon_p = make_synthetic(npix_patch, nside_patch, "disp", args.window_size)
        lon_d = make_synthetic(npix_down,  nside_down,  "disp", args.window_size)
        shft_p = ((shft_p - lon_p) + 180.0) % 360.0 - 180.0
        shft_d = ((shft_d - lon_d) + 180.0) % 360.0 - 180.0
        orig_p = np.zeros_like(orig_p)
        orig_d = np.zeros_like(orig_d)

    print("Projecting …")
    pp_orig = to_proj(orig_p, nside_patch, args.projection)
    pp_shft = to_proj(shft_p, nside_patch, args.projection)
    pd_orig = to_proj(orig_d, nside_down,  args.projection)
    pd_shft = to_proj(shft_d, nside_down,  args.projection)

    vmin_p, vmax_p = row_lims(pp_orig, pp_shft)
    vmin_d, vmax_d = row_lims(pd_orig, pd_shft)

    mode_label = {
        "index":  "pixel index (nested)",
        "window": "window ID",
        "delta":  "shift displacement (pixels)",
        "disp":   "longitude displacement (°)",
    }[args.pixel_mode]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    plot_panel(axes[0, 0], pp_orig,
               f"Post-patch nside={nside_patch}  |  Original — {mode_label}",
               args.cmap, vmin_p, vmax_p)
    plot_panel(axes[0, 1], pp_shft,
               f"Post-patch nside={nside_patch}  |  After RingShift.shift() — {mode_label}\n"
               f"(shift_size={args.shift_size})",
               args.cmap, vmin_p, vmax_p)
    plot_panel(axes[1, 0], pd_orig,
               f"After-downsample nside={nside_down}  |  Original — {mode_label}",
               args.cmap, vmin_d, vmax_d)
    plot_panel(axes[1, 1], pd_shft,
               f"After-downsample nside={nside_down}  |  After RingShift.shift() — {mode_label}\n"
               f"(shift_size={args.shift_size})",
               args.cmap, vmin_d, vmax_d)

    fig.suptitle(
        f"climate_shift_verifier  |  mode=synthetic  pixel-mode={args.pixel_mode}  "
        f"projection={args.projection}\n"
        f"Using actual RingShift.shift() from models/hp_shifting.py  |  "
        f"nside={args.nside}  patch={args.patch_size}  window={args.window_size}  "
        f"shift={args.shift_size}  D={args.depth}",
        fontsize=8,
    )
    fig.tight_layout()
    out = args.out or f"shift_verifier_synthetic_{args.pixel_mode}_{args.projection}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


# ── real mode ─────────────────────────────────────────────────────────────────

def run_real(args):
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --mode real")

    # ── imports that are only needed for real mode ────────────────────────────
    from experiments.climate.models.swin_hp_climateset_trunc_init import (
        SwinHPClimatesetTruncInit,
        SwinHPClimatesetConfig,
    )
    from experiments.climate.data.climateset_data_hp import (
        ClimatesetDataHP,
        ClimatesetHPConfig,
    )

    ckpt_path = Path(args.checkpoint)
    json_cfg  = read_cfg_from_ckpt(ckpt_path)

    # resolve nside / patch_size from json or CLI
    nside      = json_cfg.get("nside",      args.nside)
    patch_size = json_cfg.get("patch_size", args.patch_size)
    sqrt_patch = int(round(patch_size ** 0.5))
    nside_patch = nside // sqrt_patch
    nside_down  = nside_patch // 2
    npix_patch  = 12 * nside_patch ** 2

    print(f"nside={nside}  patch_size={patch_size}")
    print(f"Post-patch nside: {nside_patch}  (npix={npix_patch})")

    # ── dataset ───────────────────────────────────────────────────────────────
    data_cfg = ClimatesetHPConfig(nside=nside, split=args.split,
                                  normalized=True, seq_len=1)
    print(f"Loading dataset (split={args.split!r}) …")
    try:
        dataset = ClimatesetDataHP(data_cfg)
    except FileNotFoundError as e:
        if "Normalization stats not found" in str(e) and args.split != "train":
            print("  Computing train stats first …")
            ClimatesetDataHP(ClimatesetHPConfig(nside=nside, split="train",
                                                 normalized=True, seq_len=1))
            dataset = ClimatesetDataHP(data_cfg)
        else:
            raise
    data_spec = ClimatesetDataHP.data_spec(data_cfg)

    # ── model ─────────────────────────────────────────────────────────────────
    valid = {f.name for f in dataclasses.fields(SwinHPClimatesetConfig)}
    cfg_kwargs = {k: v for k, v in json_cfg.items() if k in valid}
    cfg_kwargs.pop("norm_layer", None)
    model_cfg = SwinHPClimatesetConfig(**cfg_kwargs)
    model = SwinHPClimatesetTruncInit(model_cfg, data_spec)

    sd = load_state_dict(ckpt_path, args.device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing: {missing[:3]}{'…' if len(missing)>3 else ''}")
    print("Checkpoint loaded.")
    model.to(args.device).eval()

    # ── find first shifted block in layers[0] ─────────────────────────────────
    # BasicLayer alternates: even blocks have shift_size=0, odd blocks have ring shift
    shifted_blocks = [blk for blk in model.layers[0].blocks
                      if blk.shift_size > 0]
    if not shifted_blocks:
        raise RuntimeError("No shifted blocks found in layers[0].")
    target_block = shifted_blocks[0]
    print(f"Hooking SwinTransformerBlock with shift_size={target_block.shift_size} "
          f"({type(target_block.shifter).__name__})")

    cap_patch = ShiftCapture()
    cap_patch.register(target_block)

    # Also hook the first shifted block in layers[1] (after downsample)
    shifted_down = [blk for blk in model.layers[1].blocks
                    if blk.shift_size > 0]
    cap_down = ShiftCapture()
    if shifted_down:
        cap_down.register(shifted_down[0])
        print(f"Hooking downsample block with shift_size={shifted_down[0].shift_size}")

    # ── forward pass ──────────────────────────────────────────────────────────
    sample = dataset[args.sample_idx]
    inp = torch.as_tensor(sample["input"], dtype=torch.float32).unsqueeze(0).to(args.device)
    with torch.no_grad():
        model({"input": inp})
    cap_patch.remove()
    cap_down.remove()

    if cap_patch.before is None:
        raise RuntimeError("Hook did not fire — model may not use ring_shift in layers[0].")

    fidx = args.feature_idx

    def extract(tensor):
        # tensor: (B, D, N, C) → (N,) for d=0, b=0, feature=fidx
        return tensor[0, 0, :, fidx].numpy()

    orig_p = extract(cap_patch.before)
    shft_p = extract(cap_patch.after)

    print(f"Projecting …  (feature_idx={fidx})")
    pp_orig = to_proj(orig_p, nside_patch, args.projection)
    pp_shft = to_proj(shft_p, nside_patch, args.projection)

    fig_rows, fig_cols = 1, 2
    has_down = cap_down.before is not None
    if has_down:
        orig_d = extract(cap_down.before)
        shft_d = extract(cap_down.after)
        nside_down_actual = int(round(np.sqrt(len(orig_d) / 12)))
        pd_orig = to_proj(orig_d, nside_down_actual, args.projection)
        pd_shft = to_proj(shft_d, nside_down_actual, args.projection)
        fig_rows = 2

    fig, axes = plt.subplots(fig_rows, 2, figsize=(14, 7 * fig_rows // 2 + 3))
    if fig_rows == 1:
        axes = axes[np.newaxis, :]

    vmin_p, vmax_p = row_lims(pp_orig, pp_shft)
    plot_panel(axes[0, 0], pp_orig,
               f"Post-patch nside={nside_patch}  |  Before RingShift.shift()\n"
               f"feature {fidx}  shape {tuple(cap_patch.before.shape)}",
               "RdBu_r", vmin_p, vmax_p)
    plot_panel(axes[0, 1], pp_shft,
               f"Post-patch nside={nside_patch}  |  After RingShift.shift()\n"
               f"feature {fidx}  shift_size={target_block.shift_size}",
               "RdBu_r", vmin_p, vmax_p)

    if has_down:
        vmin_d, vmax_d = row_lims(pd_orig, pd_shft)
        plot_panel(axes[1, 0], pd_orig,
                   f"After-downsample nside={nside_down_actual}  |  Before RingShift.shift()\n"
                   f"feature {fidx}",
                   "RdBu_r", vmin_d, vmax_d)
        plot_panel(axes[1, 1], pd_shft,
                   f"After-downsample nside={nside_down_actual}  |  After RingShift.shift()\n"
                   f"feature {fidx}",
                   "RdBu_r", vmin_d, vmax_d)

    fig.suptitle(
        f"climate_shift_verifier  |  mode=real  feature={fidx}  sample={args.sample_idx}\n"
        f"checkpoint: {args.checkpoint}",
        fontsize=8,
    )
    fig.tight_layout()
    out = args.out or f"shift_verifier_real_f{fidx}_{args.projection}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
