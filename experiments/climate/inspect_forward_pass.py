#!/usr/bin/env python3
"""
Forward-pass inspector for SwinHPClimatesetFixed.

Captures activations at 4 stages and plots them as HEALPix mollview maps:
  Stage 0 – Raw input  (before model)
  Stage 1 – After patch embedding  (one feature)
  Stage 2 – After layers[0] / attention, before downsampling
  Stage 3 – After downsampling  (PatchMerging)

Usage
-----
  # Random weights (sanity check):
  python inspect_forward_pass.py

  # Epoch-specific checkpoint file (raw state-dict):
  python inspect_forward_pass.py --checkpoint /path/to/checkpoint/model_epoch_42

  # Latest checkpoint directory (serialization.py format; contains a 'model' file):
  python inspect_forward_pass.py --checkpoint /path/to/checkpoint/

  # Custom sample / split / output:
  python inspect_forward_pass.py --checkpoint /path/to/ckpt --sample-idx 5 --split val --out diag.png

When a checkpoint directory is given, the script automatically reads the
architecture config from train_run.json so the model topology always matches
the saved weights.
"""

from __future__ import annotations
import sys
import json
import dataclasses
import argparse
from pathlib import Path

# Project root is two levels above this file (experiments/climate -> experiments -> repo root)
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import healpy as hp_lib          # the 'healpy' package for mollview
import healpix                   # the 'healpix' package used in the data pipeline
import matplotlib.pyplot as plt

from experiments.climate.models.swin_hp_climateset_fixed import (
    SwinHPClimatesetFixed,
    SwinHPClimatesetConfig,
)
from experiments.climate.data.climateset_data_hp import (
    ClimatesetDataHP,
    ClimatesetHPConfig,
)

# ── config reader ─────────────────────────────────────────────────────────────

def _unwrap_dataclass_json(obj):
    """Recursively strip __class__/__data__ wrappers from serialization.py JSON."""
    if isinstance(obj, dict):
        if "__class__" in obj and "__data__" in obj:
            return {k: _unwrap_dataclass_json(v) for k, v in obj["__data__"].items()}
        return {k: _unwrap_dataclass_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap_dataclass_json(v) for v in obj]
    return obj


def read_model_cfg_from_checkpoint(ckpt_path: Path) -> dict:
    """
    Parse train_run.json from a checkpoint directory (or the parent of a
    checkpoint file) and return the SwinHPClimatesetConfig fields as a plain dict.
    Returns empty dict if the file is missing or unparseable.
    """
    # If given a file (e.g. model_epoch_42), look for train_run.json alongside it
    search_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    json_path = search_dir / "train_run.json"
    if not json_path.is_file():
        return {}
    try:
        raw = json.loads(json_path.read_text())
        flat = _unwrap_dataclass_json(raw)

        # Walk the nested structure to find model_config
        model_cfg_raw: dict | None = None
        for train_cfg in flat.values():
            if isinstance(train_cfg, dict) and "model_config" in train_cfg:
                model_cfg_raw = train_cfg["model_config"]
                break

        if model_cfg_raw is None:
            return {}

        # Keep only fields that SwinHPClimatesetConfig actually accepts
        valid_fields = {f.name for f in dataclasses.fields(SwinHPClimatesetConfig)}
        cfg = {k: v for k, v in model_cfg_raw.items() if k in valid_fields}

        # norm_layer is serialized as a string repr; drop it (the default is fine)
        cfg.pop("norm_layer", None)

        print(f"  Read model config from train_run.json: {cfg}")
        return cfg
    except Exception as exc:
        print(f"  Warning: could not parse train_run.json ({exc}); using defaults.")
        return {}


# ── utilities ─────────────────────────────────────────────────────────────────

def next_path(path_str):
    """Return path_str if it does not exist, otherwise append _1, _2, … until free."""
    p = Path(path_str)
    if not p.exists():
        return str(p)
    i = 1
    while True:
        candidate = p.parent / f"{p.stem}_{i}{p.suffix}"
        if not candidate.exists():
            return str(candidate)
        i += 1


def npix_to_nside(npix: int) -> int:
    nside = int(round(np.sqrt(npix / 12)))
    assert 12 * nside * nside == npix, (
        f"{npix} is not a valid HEALPix pixel count (must equal 12 * nside^2)"
    )
    return nside


def project_to_array(data_1d: np.ndarray, nside: int, nest: bool = True,
                     projection: str = "moll", xsize: int = 800) -> np.ndarray:
    """Return a 2-D projected map (mollweide or equirectangular) as a numpy array."""
    if projection == "moll":
        proj = hp_lib.mollview(data_1d, nest=nest, return_projected_map=True, xsize=xsize)
        plt.close()  # close the figure healpy creates internally
        return np.ma.filled(proj, np.nan)
    else:  # cart — equirectangular via nearest-neighbour reprojection
        n_lat, n_lon = 360, 720
        lats = np.linspace(-90, 90, n_lat)
        lons = np.linspace(0, 360, n_lon, endpoint=False)
        lon_g, lat_g = np.meshgrid(lons, lats)
        pix = healpix.ang2pix(nside, lon_g.ravel(), lat_g.ravel(), lonlat=True, nest=nest)
        return data_1d[pix].reshape(n_lat, n_lon)


def plot_hp_stage(
    ax,
    title: str,
    data_1d: np.ndarray,
    nside: int,
    nest: bool = True,
    cmap: str = "RdBu_r",
    clip_pct: float = 99.0,
    projection: str = "moll",
):
    proj = project_to_array(data_1d, nside, nest=nest, projection=projection)
    finite = proj[np.isfinite(proj)]
    vmax = float(np.percentile(np.abs(finite), clip_pct)) if len(finite) else 1.0
    im = ax.imshow(
        proj,
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


# ── forward hooks ─────────────────────────────────────────────────────────────

class ForwardCapture:
    """Registers forward hooks on named modules and stores their last outputs."""

    def __init__(self):
        self.outputs: dict[str, torch.Tensor] = {}
        self._handles = []

    def register(self, name: str, module: torch.nn.Module):
        def _hook(_, __, output):
            self.outputs[name] = output.detach().cpu()
        self._handles.append(module.register_forward_hook(_hook))

    def register_pre(self, name: str, module: torch.nn.Module):
        """Capture the INPUT to a module (before it runs) rather than the output."""
        def _pre_hook(_, args):
            self.outputs[name] = args[0].detach().cpu()
        self._handles.append(module.register_forward_pre_hook(_pre_hook))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── checkpoint loader ─────────────────────────────────────────────────────────

def load_state_dict(path: Path | None, device: str) -> dict | None:
    """
    Accept either:
      - a file  (epoch checkpoint: raw state-dict saved by serialization.py)
      - a directory  (checkpoint dir containing a 'model' file)
    Returns None when path is None.
    """
    if path is None:
        return None

    if path.is_dir():
        model_file = path / "model"
        if not model_file.is_file():
            raise FileNotFoundError(
                f"Expected a 'model' file in checkpoint directory {path}"
            )
        print(f"Loading state dict from {model_file}")
        raw = torch.load(model_file, map_location=device)
    else:
        print(f"Loading state dict from {path}")
        raw = torch.load(path, map_location=device)

    # Unwrap DDP-wrapped state dicts (keys start with "module.")
    if isinstance(raw, dict) and raw and all(k.startswith("module.") for k in raw):
        raw = {k[7:]: v for k, v in raw.items()}

    return raw


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect SwinHPClimatesetFixed forward pass with HEALPix plots."
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint file or directory.  Omit for random weights.",
    )
    p.add_argument("--sample-idx", type=int, default=0,
                   help="Dataset index of the sample to use.")
    p.add_argument("--split", default="val",
                   choices=["train", "val", "test", "all"],
                   help="Dataset split to draw the sample from.")
    p.add_argument("--nside", type=int, default=32,
                   help="HEALPix nside used during training.")
    p.add_argument("--patch-size", type=int, default=16,
                   help="Patch size used in PatchEmbed.")
    p.add_argument("--input-channel", type=int, default=0,
                   help="Input channel index to show in stage 0.")
    p.add_argument("--feature-idx", type=int, default=0,
                   help="Embedding feature index to show in stages 1-3.")
    p.add_argument("--feature-idx-4", type=int, default=None,
                   help="Feature index for stage 4 (before final_up). "
                        "Defaults to --feature-idx. "
                        "Stage 4 has 2× channels: 0..(C-1) = skip from layers[0], "
                        "C..(2C-1) = upsample path.")
    p.add_argument("--pre-final-only", action="store_true",
                   help="Show only the 4 features specified by --pre-final-features "
                        "from the stage right before final_up.  Skips all other stages.")
    p.add_argument("--pre-final-features", type=int, nargs=4,
                   default=[0, 16, 48, 64],
                   metavar=("F0", "F1", "F2", "F3"),
                   help="Four feature indices to plot in --pre-final-only mode "
                        "(default: 0 16 48 64).  "
                        "Recall: 0..(C-1) = skip from layers[0], C..(2C-1) = upsample path.")
    p.add_argument("--cmap", default="RdBu_r",
                   help="Matplotlib colormap.")
    p.add_argument("--projection", default="moll", choices=["moll", "cart"],
                   help="Map projection: mollweide (default) or cartesian/equirectangular.")
    p.add_argument("--out", default="forward_pass_inspection.png",
                   help="Output file for the plot.")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── resolve effective nside/patch_size from checkpoint before anything else
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    json_cfg: dict = {}
    if ckpt_path is not None:
        json_cfg = read_model_cfg_from_checkpoint(ckpt_path)

    merged_cfg = {"nside": args.nside, "patch_size": args.patch_size}
    merged_cfg.update(json_cfg)  # train_run.json values override CLI defaults
    eff_nside = merged_cfg["nside"]

    # ── dataset ──────────────────────────────────────────────────────────────
    data_cfg = ClimatesetHPConfig(
        nside=eff_nside,
        split=args.split,
        normalized=True,
        seq_len=1,
    )
    print(f"Loading dataset (split={args.split!r}, nside={eff_nside}) …")
    try:
        dataset = ClimatesetDataHP(data_cfg)
    except FileNotFoundError as e:
        if "Normalization stats not found" in str(e) and args.split != "train":
            print(f"  Stats missing for split={args.split!r} — computing from train split first …")
            train_cfg = ClimatesetHPConfig(
                nside=eff_nside,
                split="train",
                normalized=True,
                seq_len=1,
            )
            ClimatesetDataHP(train_cfg)   # side-effect: computes and saves stats
            dataset = ClimatesetDataHP(data_cfg)
        else:
            raise

    data_spec = ClimatesetDataHP.data_spec(data_cfg)
    input_vars: list[str] = list(data_cfg.input_vars)

    print(f"  n_input_channels  = {data_spec.n_input_channels}")
    print(f"  n_output_channels = {data_spec.n_output_channels}")
    print(f"  input_vars        = {input_vars}")

    # ── model ─────────────────────────────────────────────────────────────────
    model_cfg = SwinHPClimatesetConfig(**merged_cfg)
    model = SwinHPClimatesetFixed(model_cfg, data_spec)

    state_dict = load_state_dict(ckpt_path, args.device)
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}):    {missing[:3]}{'…' if len(missing)>3 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'…' if len(unexpected)>3 else ''}")
        print("Checkpoint loaded.")
    else:
        print("No checkpoint — using random weights.")

    model.to(args.device).eval()

    # ── sample ────────────────────────────────────────────────────────────────
    sample = dataset[args.sample_idx]
    inp = (
        torch.as_tensor(sample["input"], dtype=torch.float32)
        .unsqueeze(0)           # add batch dim → (1, C, N_pix)
        .to(args.device)
    )
    batch = {"input": inp}

    # ── expected nside at each stage ──────────────────────────────────────────
    # Use merged_cfg (train_run.json overrides CLI) so patch_size=4 checkpoints
    # get the right labels instead of silently using the CLI defaults.
    eff_nside      = merged_cfg["nside"]
    eff_patch_size = merged_cfg["patch_size"]
    npix_input   = hp_lib.nside2npix(eff_nside)
    npix_patches = npix_input    // eff_patch_size
    npix_merged  = npix_patches  // 4

    nside_input   = eff_nside
    nside_patches = npix_to_nside(npix_patches)
    nside_merged  = npix_to_nside(npix_merged)

    print(f"\nExpected nside / npix at each stage:")
    print(f"  Stage 0 (raw input):           nside={nside_input}   npix={npix_input}")
    print(f"  Stage 1 (after patch_embed):   nside={nside_patches}   npix={npix_patches}")
    print(f"  Stage 2 (after layers[0]):     nside={nside_patches}   npix={npix_patches}")
    print(f"  Stage 3 (after downsample):    nside={nside_merged}    npix={npix_merged}")

    # ── hooks ─────────────────────────────────────────────────────────────────
    capture = ForwardCapture()
    if args.pre_final_only:
        capture.register_pre("pre_final_up", model.final_up)
    else:
        capture.register("patch_embed",  model.patch_embed)
        capture.register("layers_0",     model.layers[0])
        capture.register("downsample",   model.downsample)
        capture.register("norm",         model.norm)
        capture.register_pre("pre_final_up", model.final_up)

    # ── forward pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        model(batch)
    capture.remove()

    # ── extract 1-D slices for visualisation ──────────────────────────────────
    cidx = args.input_channel
    fidx = args.feature_idx
    ch_label = input_vars[cidx] if cidx < len(input_vars) else f"ch{cidx}"

    pfu = capture.outputs["pre_final_up"]  # input to final_up: (B,1,N_patches,2C)
    n_ch4 = pfu.shape[-1]
    half  = n_ch4 // 2

    # ── --pre-final-only: 2×2 grid of 4 chosen features ──────────────────────
    if args.pre_final_only:
        feats = args.pre_final_features
        for f in feats:
            if f < 0 or f >= n_ch4:
                raise ValueError(f"Feature {f} out of range (0–{n_ch4-1}; "
                                 f"skip=0–{half-1}, upsample={half}–{n_ch4-1})")
        fig, axes = plt.subplots(2, 2, figsize=(14, 7))
        axes = axes.ravel()
        for ax, f in zip(axes, feats):
            data = pfu[0, 0, :, f].numpy()
            half_lbl = "skip" if f < half else "upsample"
            plot_hp_stage(ax,
                          f"Before final_up  feature {f}  [{half_lbl}]\n"
                          f"shape {tuple(pfu.shape)}  nside={nside_patches}  "
                          f"(skip=0–{half-1} | upsample={half}–{n_ch4-1})",
                          data, nside_patches, cmap=args.cmap, projection=args.projection)
        ckpt_label = args.checkpoint or "random weights"
        fig.suptitle(
            f"Before final_up  |  sample_idx={args.sample_idx}  split={args.split!r}  "
            f"|  {ckpt_label}",
            fontsize=9,
        )
        fig.tight_layout()
        _base = args.out if args.out != "forward_pass_inspection.png" \
                else "pre_final_up_features.png"
        out = next_path(_base)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved → {out}")
        plt.show()
        return

    # Stage 0: (1, C, N_pix) → (N_pix,)
    s0 = inp[0, cidx].cpu().numpy()

    # Stages 1-5: (B, 1, N_tokens, embed_dim) → pick [b=0, d=0, :, fidx]
    pe  = capture.outputs["patch_embed"]    # after Conv1d + permute
    l0  = capture.outputs["layers_0"]      # after SwinTransformer attention blocks
    ds  = capture.outputs["downsample"]    # after PatchMerging
    nm  = capture.outputs["norm"]          # bottleneck: after layers[2]+norm

    s1 = pe [0, 0, :, fidx].numpy()
    s2 = l0 [0, 0, :, fidx].numpy()
    s3 = ds [0, 0, :, fidx].numpy()
    s5 = nm [0, 0, :, fidx].numpy()
    # Stage 4: 2× channels — 0..(C-1) = skip from layers[0], C..(2C-1) = upsample path
    n_ch4 = pfu.shape[-1]
    half  = n_ch4 // 2
    fidx4 = args.feature_idx_4 if args.feature_idx_4 is not None else fidx
    if fidx4 < 0 or fidx4 >= n_ch4:
        raise ValueError(f"--feature-idx-4 {fidx4} out of range for stage 4 "
                         f"(0–{n_ch4-1}; skip=0–{half-1}, upsample={half}–{n_ch4-1})")
    s4 = pfu[0, 0, :, fidx4].numpy()
    half_label = "skip from layers[0]" if fidx4 < half else "upsample path"

    print(f"\nActual captured tensor shapes:")
    print(f"  Stage 0 (input):              {tuple(inp.shape)}")
    print(f"  Stage 1 (patch_embed out):    {tuple(pe.shape)}")
    print(f"  Stage 2 (layers[0] out):      {tuple(l0.shape)}")
    print(f"  Stage 3 (downsample out):     {tuple(ds.shape)}")
    print(f"  Stage 5 (norm / bottleneck):  {tuple(nm.shape)}")
    print(f"  Stage 4 (pre final_up input): {tuple(pfu.shape)}  "
          f"(plotting feature {fidx4}  [{half_label}]  "
          f"skip=0–{half-1}, upsample={half}–{n_ch4-1})")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(21, 7))
    axes = axes.ravel()

    plot_hp_stage(
        axes[0],
        f"Stage 0 – Raw input  (channel: {ch_label}, idx={cidx})\n"
        f"shape {tuple(inp.shape)}  →  plotting  (npix={len(s0)}, nside={nside_input})",
        s0, nside_input, cmap=args.cmap, projection=args.projection,
    )
    plot_hp_stage(
        axes[1],
        f"Stage 1 – After patch_embed  (feature {fidx})\n"
        f"full shape {tuple(pe.shape)}  →  plotting  (npix={len(s1)}, nside={nside_patches})",
        s1, nside_patches, cmap=args.cmap, projection=args.projection,
    )
    plot_hp_stage(
        axes[2],
        f"Stage 2 – After layers[0] / attention  (feature {fidx})\n"
        f"full shape {tuple(l0.shape)}  →  plotting  (npix={len(s2)}, nside={nside_patches})",
        s2, nside_patches, cmap=args.cmap, projection=args.projection,
    )
    plot_hp_stage(
        axes[3],
        f"Stage 3 – After downsample / PatchMerging  (feature {fidx})\n"
        f"full shape {tuple(ds.shape)}  →  plotting  (npix={len(s3)}, nside={nside_merged})",
        s3, nside_merged, cmap=args.cmap, projection=args.projection,
    )
    plot_hp_stage(
        axes[4],
        f"Stage 4 – Before final_up  (feature {fidx4}  [{half_label}])\n"
        f"full shape {tuple(pfu.shape)}  →  plotting  (npix={len(s4)}, nside={nside_patches})\n"
        f"skip=feat 0–{half-1}  |  upsample path=feat {half}–{n_ch4-1}",
        s4, nside_patches, cmap=args.cmap, projection=args.projection,
    )
    plot_hp_stage(
        axes[5],
        f"Stage 5 – Bottleneck: after layers[2]+norm  (feature {fidx})\n"
        f"full shape {tuple(nm.shape)}  →  plotting  (npix={len(s5)}, nside={nside_merged})\n"
        f"[deepest compressed representation, before upsample]",
        s5, nside_merged, cmap=args.cmap, projection=args.projection,
    )

    ckpt_label = args.checkpoint or "random weights"
    fig.suptitle(
        f"Forward-pass inspection  |  sample_idx={args.sample_idx}  split={args.split!r}  |  {ckpt_label}",
        fontsize=9,
    )
    fig.tight_layout()
    out = next_path(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
