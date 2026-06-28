#!/usr/bin/env python3
"""
RingShift mask viewer.

For each resolution (post-patch and after-downsample) shows two panels side by side:
  Left:  longitude-displacement map produced by the ring shift.
  Right: same map but with the pixels that carry a non-standard attention-mask
         label drawn in red on top (these are the pixels that will be masked
         against other-label pixels inside their attention window).

The mask labels come from RingShift._get_shifted_idcs_and_mask() and are based
on the RING position of each (nested-ordered) pixel:
  - majority label  ->  standard attention  (no -100 masking)
  - boundary label  ->  masked against majority pixels within the same window
  - seam label      ->  masked against everything else (global ring-wrap pixels)

Usage
-----
  python mask_viewer.py
  python mask_viewer.py --projection cart
  python mask_viewer.py --nside 64 --patch-size 16 --window-size 64 --shift-size 4
  python mask_viewer.py --depth 2 --out mask.png
"""

import sys
import argparse
from pathlib import Path

_ROOT  = Path(__file__).resolve().parents[2]
_LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LOCAL))

import numpy as np
import healpix as healpix_pkg
import healpy  as hp_lib
import chealpix as chp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from models.hp_shifting import RingShift


# ── ring-shift helpers ────────────────────────────────────────────────────────

def build_ring_shift(nside, window_size_hp, shift_size, D=1):
    N = 12 * nside ** 2
    return RingShift(
        nside=nside,
        base_pix=12,
        window_size=[D, window_size_hp],
        shift_size=shift_size,
        input_resolution=(D, N),
    )


def shift_disp(nside, shift_size):
    """
    Longitude displacement (degrees) for each nested pixel after the ring shift.
    disp[i] = lon(source_of_i) - lon(i), wrapped to [-180, 180].
    """
    npix = 12 * nside ** 2
    ring_idcs = np.arange(npix)
    shifted = np.roll(ring_idcs, -shift_size)
    shifted_nest = chp.ring2nest(nside, shifted)
    nest_idcs = np.arange(npix)
    ring_of_nest = chp.nest2ring(nside, nest_idcs)
    shift_idcs = shifted_nest[ring_of_nest]          # shift_idcs[i] = source of nested pixel i

    lon, _ = healpix_pkg.pix2ang(nside, nest_idcs, nest=True, lonlat=True)
    lon_src = lon[shift_idcs]
    raw = lon_src - lon
    return (raw + 180.0) % 360.0 - 180.0            # wrap to [-180, 180]


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


def to_proj(data, nside, projection, nest=True):
    if projection == "moll":
        return project_moll(data, nest=nest)
    return project_cart(data, nside, nest=nest)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_disp(ax, disp_proj, title, vabs=180):
    """Plot longitude-displacement map (symmetric colour scale)."""
    im = ax.imshow(disp_proj, origin="lower", cmap="RdBu_r",
                   vmin=-vabs, vmax=vabs,
                   interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="disp (°)")


def plot_disp_with_overlay(ax, disp_proj, boundary_proj, seam_proj, title):
    """
    Displacement map with two overlays:
      orange = boundary-label pixels (will be masked against majority)
      red    = seam-label pixels (global ring-wrap pixels)
    """
    vabs = 180
    im = ax.imshow(disp_proj, origin="lower", cmap="RdBu_r",
                   vmin=-vabs, vmax=vabs,
                   interpolation="nearest", aspect="auto")

    # boundary pixels: orange, semi-transparent
    boundary_rgba = np.zeros((*boundary_proj.shape, 4))
    mask_b = np.isfinite(boundary_proj) & (boundary_proj > 0)
    boundary_rgba[mask_b] = [1.0, 0.5, 0.0, 0.85]   # orange
    ax.imshow(boundary_rgba, origin="lower", interpolation="nearest", aspect="auto")

    # seam pixels: red, opaque
    seam_rgba = np.zeros((*seam_proj.shape, 4))
    mask_s = np.isfinite(seam_proj) & (seam_proj > 0)
    seam_rgba[mask_s] = [1.0, 0.0, 0.0, 1.0]         # red
    ax.imshow(seam_rgba, origin="lower", interpolation="nearest", aspect="auto")

    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="disp (°)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay RingShift attention-mask boundaries on the displacement map."
    )
    p.add_argument("--nside",       type=int, default=32)
    p.add_argument("--patch-size",  type=int, default=16)
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--shift-size",  type=int, default=4)
    p.add_argument("--depth",       type=int, default=1,
                   help="Depth dimension D (1=climate, 2=weather).")
    p.add_argument("--projection",  default="moll", choices=["moll", "cart"])
    p.add_argument("--out",         default=None)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    sqrt_patch = int(round(args.patch_size ** 0.5))
    assert sqrt_patch ** 2 == args.patch_size

    nside_patch = args.nside // sqrt_patch
    nside_down  = nside_patch // 2
    npix_patch  = 12 * nside_patch ** 2
    npix_down   = 12 * nside_down  ** 2

    print(f"Post-patch nside:        {nside_patch}  (npix={npix_patch})")
    print(f"After-downsample nside:  {nside_down}  (npix={npix_down})")
    print(f"window_size_hp={args.window_size}  shift_size={args.shift_size}  D={args.depth}")
    print()

    # ── build RingShift and extract masks ─────────────────────────────────────
    rs_p = build_ring_shift(nside_patch, args.window_size, args.shift_size, D=args.depth)
    rs_d = build_ring_shift(nside_down,  args.window_size, args.shift_size, D=args.depth)

    # mask[0] is shape (N,) in NESTED ordering; label based on ring position of each pixel
    mask_p = rs_p.mask[0].numpy()
    mask_d = rs_d.mask[0].numpy()

    majority_p = int(np.bincount(mask_p).argmax())
    majority_d = int(np.bincount(mask_d).argmax())

    labels_p = np.unique(mask_p)
    labels_d = np.unique(mask_d)

    # boundary = label just below the seam; seam = highest label
    seam_label_p = int(labels_p.max())
    seam_label_d = int(labels_d.max())
    bnd_label_p  = int(sorted(labels_p)[-2]) if len(labels_p) > 2 else seam_label_p
    bnd_label_d  = int(sorted(labels_d)[-2]) if len(labels_d) > 2 else seam_label_d

    boundary_p = (mask_p == bnd_label_p).astype(float)
    boundary_d = (mask_d == bnd_label_d).astype(float)
    seam_p     = (mask_p == seam_label_p).astype(float)
    seam_d     = (mask_d == seam_label_d).astype(float)

    n_bnd_p = int(boundary_p.sum());  n_seam_p = int(seam_p.sum())
    n_bnd_d = int(boundary_d.sum());  n_seam_d = int(seam_d.sum())
    n_win_p = npix_patch // args.window_size
    n_win_d = npix_down  // args.window_size

    print(f"Post-patch  — unique labels: {labels_p}  (majority={majority_p})")
    print(f"  boundary pixels (orange): {n_bnd_p} / {npix_patch}"
          f"  across {n_win_p} windows")
    print(f"  seam pixels (red):        {n_seam_p} / {npix_patch}")
    print()
    print(f"Downsample  — unique labels: {labels_d}  (majority={majority_d})")
    print(f"  boundary pixels (orange): {n_bnd_d} / {npix_down}"
          f"  across {n_win_d} windows")
    print(f"  seam pixels (red):        {n_seam_d} / {npix_down}")
    print()

    # ── compute displacement maps ─────────────────────────────────────────────
    print("Computing displacements …")
    disp_p = shift_disp(nside_patch, args.shift_size)
    disp_d = shift_disp(nside_down,  args.shift_size)

    print("Projecting …")
    proj_disp_p    = to_proj(disp_p,    nside_patch, args.projection)
    proj_disp_d    = to_proj(disp_d,    nside_down,  args.projection)
    proj_bnd_p     = to_proj(boundary_p, nside_patch, args.projection)
    proj_bnd_d     = to_proj(boundary_d, nside_down,  args.projection)
    proj_seam_p    = to_proj(seam_p,    nside_patch, args.projection)
    proj_seam_d    = to_proj(seam_d,    nside_down,  args.projection)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    plot_disp(axes[0, 0], proj_disp_p,
              f"Post-patch nside={nside_patch}  |  longitude displacement after ring shift")

    plot_disp_with_overlay(
        axes[0, 1], proj_disp_p, proj_bnd_p, proj_seam_p,
        f"Post-patch nside={nside_patch}  |  masked pixels overlaid\n"
        f"orange = boundary ({n_bnd_p} px, ring pos N-{args.window_size}..N-{args.shift_size+1})  "
        f"red = seam ({n_seam_p} px, ring pos N-{args.shift_size}..N-1)"
    )

    plot_disp(axes[1, 0], proj_disp_d,
              f"After-downsample nside={nside_down}  |  longitude displacement after ring shift")

    plot_disp_with_overlay(
        axes[1, 1], proj_disp_d, proj_bnd_d, proj_seam_d,
        f"After-downsample nside={nside_down}  |  masked pixels overlaid\n"
        f"orange = boundary ({n_bnd_d} px, ring pos N-{args.window_size}..N-{args.shift_size+1})  "
        f"red = seam ({n_seam_d} px, ring pos N-{args.shift_size}..N-1)"
    )

    fig.suptitle(
        f"RingShift mask overlay  |  projection={args.projection}  |  "
        f"input nside={args.nside}  patch={args.patch_size}  "
        f"window_hp={args.window_size}  shift={args.shift_size}  D={args.depth}\n"
        f"NOTE: mask labels are assigned by RING position of each (nested-ordered) pixel.\n"
        f"Pixels in the same attention window but with different labels get -100 attention weight.",
        fontsize=8,
    )
    fig.tight_layout()

    out = args.out or f"mask_viewer_{args.projection}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
