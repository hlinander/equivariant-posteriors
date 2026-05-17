#!/usr/bin/env python3
"""
Synthetic HEALPix viewer.

Generates synthetic maps (pixel index in NESTED ordering) at the two
resolutions the SwinHP model operates at, and shows the effect of the
ring shift that enables window-shifted attention.

  Post-patch:        nside = input_nside / sqrt(patch_size)   (e.g. 32/4 = 8)
  After downsample:  nside = post_patch_nside / 2             (e.g. 8/2  = 4)

Four display modes (--mode):
  index   – raw pixel index 0…N-1  (reveals NESTED Z-order structure)
  window  – attention window ID per pixel  (shows window tiling)
  delta   – (shifted index - original index) mod N  (pixel-index displacement)
  disp    – actual longitude displacement in degrees  (shows polar non-uniformity)

Layout: 2 rows (resolutions) × 2 columns (original | after ring shift).

Usage
-----
  # default: mollview, index mode, model defaults
  python synthetic_healpix_viewer.py

  # cartesian projection
  python synthetic_healpix_viewer.py --projection cart

  # window-tiling mode, save to file
  python synthetic_healpix_viewer.py --mode window --out windows.png

  # custom config (must match your checkpoint)
  python synthetic_healpix_viewer.py --nside 32 --patch-size 16 \\
      --window-size 64 --shift-size 4
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
import chealpix as chp
import healpy as hp_lib
import healpix as healpix_pkg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def npix_to_nside(npix: int) -> int:
    nside = int(round(np.sqrt(npix / 12)))
    assert 12 * nside * nside == npix
    return nside


# ── ring-shift permutation ────────────────────────────────────────────────────

def ring_shift_indices(nside: int, shift_size: int) -> np.ndarray:
    """
    Return the permutation array `shift_idcs` such that applying
    `data[shift_idcs]` replicates exactly what RingShift.shift() does
    to the HP dimension.

    Follows hp_shifting.RingShift._get_shifted_idcs_and_mask() exactly:
      1. Roll ring indices by -shift_size
      2. Convert to nested
      3. Compose with nest→ring map to get the full permutation in nested space
    """
    npix = 12 * nside ** 2
    ring_idcs = np.arange(npix)
    shifted_ring_idcs = np.roll(ring_idcs, -shift_size)
    shifted_ring_idcs_in_nest = chp.ring2nest(nside, shifted_ring_idcs)
    nest_idcs = np.arange(npix)
    nest_idcs_in_ring = chp.nest2ring(nside, nest_idcs)
    return shifted_ring_idcs_in_nest[nest_idcs_in_ring]


# ── projection helpers ────────────────────────────────────────────────────────

def project_moll(data: np.ndarray, nest: bool = True, xsize: int = 800) -> np.ndarray:
    proj = hp_lib.mollview(data.astype(float), nest=nest,
                           return_projected_map=True, xsize=xsize)
    plt.close()
    return np.ma.filled(proj, np.nan)


def project_cart(data: np.ndarray, nside: int, nest: bool = True,
                 n_lat: int = 360, n_lon: int = 720) -> np.ndarray:
    lats = np.linspace(-90, 90, n_lat)
    lons = np.linspace(0, 360, n_lon, endpoint=False)
    lon_g, lat_g = np.meshgrid(lons, lats)
    pix = healpix_pkg.ang2pix(nside, lon_g.ravel(), lat_g.ravel(),
                               lonlat=True, nest=nest)
    return data.astype(float)[pix].reshape(n_lat, n_lon)


def to_projected(data: np.ndarray, nside: int, projection: str,
                 nest: bool = True) -> np.ndarray:
    if projection == "moll":
        return project_moll(data, nest=nest)
    return project_cart(data, nside, nest=nest)


# ── coloring modes ────────────────────────────────────────────────────────────

def make_data(npix: int, shift_idcs: np.ndarray,
              mode: str, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (original_values, shifted_values) for the given mode."""
    idx = np.arange(npix)

    if mode == "index":
        orig = idx.astype(float)
        shft = shift_idcs.astype(float)   # shifted[i] = shift_idcs[i] (= original source pixel)

    elif mode == "window":
        # Colour by attention-window ID so window boundaries become visible
        orig = (idx // window_size).astype(float)
        shft = (shift_idcs // window_size).astype(float)

    elif mode == "delta":
        # Displacement: how far (in pixel-index space) each pixel moved
        orig = np.zeros(npix, dtype=float)  # reference: zero displacement
        shft = ((shift_idcs.astype(int) - idx) % npix).astype(float)

    elif mode == "disp":
        # Geographic longitude displacement in degrees.
        # For each pixel i, how many degrees east did its source come from?
        # source_lon - dest_lon, wrapped to [-180, 180].
        # Original panel shows zero everywhere (reference); shifted panel shows
        # the actual geographic shift per pixel — reveals polar non-uniformity.
        lon, _ = healpix_pkg.pix2ang(npix_to_nside(npix), np.arange(npix),
                                      nest=True, lonlat=True)
        lon_src = lon[shift_idcs]
        raw = lon_src - lon
        disp = (raw + 180.0) % 360.0 - 180.0   # wrap to [-180, 180]
        orig = np.zeros(npix, dtype=float)
        shft = disp.astype(float)

    else:
        raise ValueError(f"Unknown mode {mode!r}; choose index | window | delta | disp")

    return orig, shft


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_panel(ax, proj: np.ndarray, title: str, cmap: str,
               vmin=None, vmax=None, cyclic: bool = False):
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Synthetic HEALPix viewer for SwinHP ring-shift inspection."
    )
    p.add_argument("--nside", type=int, default=32,
                   help="Input nside (full-resolution data).")
    p.add_argument("--patch-size", type=int, default=16,
                   help="Patch size used in PatchEmbed.")
    p.add_argument("--window-size", type=int, default=64,
                   help="Attention window size in HP dimension.")
    p.add_argument("--shift-size", type=int, default=4,
                   help="Ring shift size (pixels in ring ordering).")
    p.add_argument("--mode", default="index",
                   choices=["index", "window", "delta", "disp"],
                   help="What value to colour each pixel by.")
    p.add_argument("--projection", default="moll", choices=["moll", "cart"],
                   help="Map projection: mollweide or cartesian/equirectangular.")
    p.add_argument("--cmap", default="plasma",
                   help="Matplotlib colormap.")
    p.add_argument("--out", default=None,
                   help="Output file. Defaults to synthetic_<mode>_<projection>.png")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Derived nsides
    sqrt_patch = int(round(args.patch_size ** 0.5))
    assert sqrt_patch ** 2 == args.patch_size, \
        f"patch_size must be a perfect square (got {args.patch_size})"

    nside_patch = args.nside // sqrt_patch       # post-patch embedding
    nside_down  = nside_patch // 2               # after PatchMerging

    npix_patch = 12 * nside_patch ** 2
    npix_down  = 12 * nside_down  ** 2

    print(f"Input nside:          {args.nside}  (npix={12*args.nside**2})")
    print(f"Post-patch nside:     {nside_patch}  (npix={npix_patch})")
    print(f"After-downsample nside: {nside_down}  (npix={npix_down})")
    print(f"Window size (HP):     {args.window_size}")
    print(f"Shift size:           {args.shift_size}")
    print(f"Mode:                 {args.mode}")
    print(f"Projection:           {args.projection}")

    # Ring-shift permutations at each resolution
    shift_patch = ring_shift_indices(nside_patch, args.shift_size)
    shift_down  = ring_shift_indices(nside_down,  args.shift_size)

    # Synthetic data
    orig_patch, shft_patch = make_data(npix_patch, shift_patch,
                                       args.mode, args.window_size)
    orig_down,  shft_down  = make_data(npix_down,  shift_down,
                                       args.mode, args.window_size)

    print("\nProjecting maps …")

    # Project all four panels
    pp_orig = to_projected(orig_patch, nside_patch, args.projection)
    pp_shft = to_projected(shft_patch, nside_patch, args.projection)
    pd_orig = to_projected(orig_down,  nside_down,  args.projection)
    pd_shft = to_projected(shft_down,  nside_down,  args.projection)

    # Shared colour scale within each row so comparisons are fair
    def row_lims(*arrs):
        finite = np.concatenate([a[np.isfinite(a)] for a in arrs])
        return float(np.nanmin(finite)), float(np.nanmax(finite))

    vmin_p, vmax_p = row_lims(pp_orig, pp_shft)
    vmin_d, vmax_d = row_lims(pd_orig, pd_shft)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    mode_label = {"index": "pixel index (nested)",
                  "window": "window ID",
                  "delta": "shift displacement (pixels)",
                  "disp":  "longitude displacement (°)"}[args.mode]

    plot_panel(axes[0, 0], pp_orig,
               f"Post-patch  nside={nside_patch}  npix={npix_patch}\n"
               f"Original — {mode_label}",
               args.cmap, vmin_p, vmax_p)

    plot_panel(axes[0, 1], pp_shft,
               f"Post-patch  nside={nside_patch}  npix={npix_patch}\n"
               f"After ring shift ({args.shift_size} px) — {mode_label}",
               args.cmap, vmin_p, vmax_p)

    plot_panel(axes[1, 0], pd_orig,
               f"After downsample  nside={nside_down}  npix={npix_down}\n"
               f"Original — {mode_label}",
               args.cmap, vmin_d, vmax_d)

    plot_panel(axes[1, 1], pd_shft,
               f"After downsample  nside={nside_down}  npix={npix_down}\n"
               f"After ring shift ({args.shift_size} px) — {mode_label}",
               args.cmap, vmin_d, vmax_d)

    fig.suptitle(
        f"Synthetic HEALPix inspection  |  mode={args.mode}  "
        f"projection={args.projection}  |  "
        f"input nside={args.nside}  patch_size={args.patch_size}  "
        f"window={args.window_size}  shift={args.shift_size}",
        fontsize=9,
    )
    fig.tight_layout()

    out = args.out or f"synthetic_{args.mode}_{args.projection}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
