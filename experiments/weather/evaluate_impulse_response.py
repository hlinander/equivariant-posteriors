#!/usr/bin/env python
"""
Impulse response analysis for HEALPix weather forecasting models.

Feed a synthetic "impulse" input (a spatial pattern at high amplitude, zero elsewhere)
through a trained model and visualise the output to understand how information propagates
across the sphere.

Usage:
    uv run python experiments/weather/evaluate_impulse_response.py <config.py> \\
        [--epoch 100] \\
        [--impulse equator|poles|north_pole|south_pole|meridian|spot] \\
        [--amplitude 5.0] \\
        [--lat-width 5.0] \\
        [--lon 0.0] \\
        [--spot-lat 0.0 --spot-lon 0.0] \\
        [--input-channel all|<int>] \\
        [--device cuda|cpu] \\
        [--out-dir experiments/weather/plots/impulse_response]

Impulse types
-------------
equator    : pixels within ±lat_width degrees of the equator (default)
poles      : pixels within lat_width degrees of both poles
north_pole : pixels within lat_width degrees of the north pole only
south_pole : pixels within lat_width degrees of the south pole only
meridian   : pixels within lat_width degrees of longitude `--lon`
spot       : a single HEALPix pixel nearest to (--spot-lat, --spot-lon)

Output
------
One figure per surface channel (4 total) and one summary figure showing
all surface channels side-by-side, saved as PDF to <out-dir>/<run_name>/.
"""

import argparse
import importlib
from pathlib import Path

import healpy as hp
import healpix
import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.serialization import DeserializeConfig, deserialize_model

# ── helpers ───────────────────────────────────────────────────────────────────

SURFACE_NAMES = ["msl", "u10", "v10", "t2m"]
UPPER_NAMES   = ["z", "q", "t", "u", "v"]


def load_create_config(module_file_path: str):
    module_name = Path(module_file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    return cfg_mod.create_config


def get_nside(create_config, epoch: int) -> int:
    train_run = create_config(0, epoch)
    data_cfg = train_run.train_config.train_data_config
    hp_cfg = getattr(data_cfg, "base", data_cfg)
    return hp_cfg.nside


def get_model(create_config, epoch: int, device: str):
    train_run = create_config(0, epoch)
    deser_config = DeserializeConfig(train_run=train_run, device_id=device)
    deser = deserialize_model(deser_config, latest_ok=True)
    if deser is None:
        raise RuntimeError(f"No checkpoint found for epoch {epoch}.")
    if deser.epoch != epoch:
        print(f"  Warning: requested epoch {epoch}, loaded latest saved epoch {deser.epoch}.")
    return deser.model


# ── impulse construction ──────────────────────────────────────────────────────

def build_impulse_mask(nside: int, impulse: str, lat_width: float,
                       lon: float, spot_lat: float, spot_lon: float) -> np.ndarray:
    """Return a boolean mask over HEALPix pixels that are 'on' in the impulse."""
    npix  = healpix.nside2npix(nside)
    lons, lats = healpix.pix2ang(nside, np.arange(npix), lonlat=True, nest=True)

    if impulse == "equator":
        mask = np.abs(lats) <= lat_width

    elif impulse == "poles":
        mask = np.abs(lats) >= (90.0 - lat_width)

    elif impulse == "north_pole":
        mask = lats >= (90.0 - lat_width)

    elif impulse == "south_pole":
        mask = lats <= -(90.0 - lat_width)

    elif impulse == "meridian":
        # angular distance from the given meridian, accounting for wrap-around
        d = np.abs(((lons - lon + 180) % 360) - 180)
        mask = d <= lat_width

    elif impulse == "spot":
        # nearest single pixel
        pix = healpix.ang2pix(nside, spot_lon, spot_lat, lonlat=True, nest=True)
        mask = np.zeros(npix, dtype=bool)
        mask[pix] = True

    else:
        raise ValueError(f"Unknown impulse type: {impulse!r}")

    if not mask.any():
        raise ValueError(
            f"Impulse mask is empty (impulse={impulse!r}, lat_width={lat_width}). "
            "Try increasing --lat-width."
        )
    return mask


def build_batch(nside: int, mask: np.ndarray, amplitude: float,
                input_channel, device) -> dict:
    """
    Construct a synthetic batch with the impulse pattern.

    input_channel: 'all' → activate all surface + upper channels;
                   int   → activate only that surface channel index.
    """
    npix = healpix.nside2npix(nside)
    surf = np.zeros((1, 4, npix), dtype=np.float32)
    upper = np.zeros((1, 5, 13, npix), dtype=np.float32)

    if input_channel == "all":
        surf[:, :, mask]  = amplitude
        upper[:, :, :, mask] = amplitude
    else:
        ch = int(input_channel)
        surf[:, ch, mask] = amplitude

    return {
        "input_surface": torch.tensor(surf).to(device),
        "input_upper":   torch.tensor(upper).to(device),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def _to_ring(values: np.ndarray, nside: int) -> np.ndarray:
    """Convert a NESTED HEALPix map to RING ordering expected by healpy."""
    return hp.reorder(values, n2r=True)


def _safe_vlim(data: np.ndarray) -> tuple[float, float]:
    """Return finite symmetric (vmin, vmax), replacing NaN/Inf with 0."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.abs(finite).max())
    if vmax == 0.0:
        return -1.0, 1.0
    return -vmax, vmax


def _mollview(fig: plt.Figure, sub: tuple, values: np.ndarray, nside: int,
              title: str, vmin=None, vmax=None, cmap="RdBu_r"):
    """Draw a single Mollweide map into a subplot position using healpy.mollview."""
    clean = np.where(np.isfinite(values), values, 0.0)
    if vmin is None or vmax is None:
        vmin, vmax = _safe_vlim(clean)
    hp.mollview(
        _to_ring(clean, nside),
        fig=fig.number,
        sub=sub,
        title=title,
        min=vmin,
        max=vmax,
        cmap=cmap,
        notext=True,
        cbar=True,
    )
    hp.graticule(dpar=30, dmer=60, alpha=0.3, verbose=False)


def plot_impulse_input(nside: int, mask: np.ndarray, amplitude: float,
                       out_path: Path):
    """Plot the input impulse pattern."""
    values = np.where(mask, amplitude, 0.0).astype(np.float64)
    fig = plt.figure(figsize=(8, 4))
    hp.mollview(
        _to_ring(values, nside),
        fig=fig.number,
        title="Impulse input",
        min=0, max=amplitude,
        cmap="hot_r",
        notext=True,
        cbar=True,
    )
    hp.graticule(dpar=30, dmer=60, alpha=0.3, verbose=False)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_surface_response(nside: int, output: np.ndarray,
                          channel_names: list, out_path: Path):
    """Plot all 4 surface channels in a 2×2 grid."""
    n_ch = output.shape[0]
    ncols, nrows = 2, (n_ch + 1) // 2
    vmin, vmax = _safe_vlim(output)

    fig = plt.figure(figsize=(10, 4 * nrows))
    for ch_idx in range(n_ch):
        _mollview(fig, (nrows, ncols, ch_idx + 1),
                  output[ch_idx].astype(np.float64), nside,
                  channel_names[ch_idx], vmin=vmin, vmax=vmax)

    fig.suptitle("Surface response to impulse input", fontsize=12, y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_upper_response(nside: int, output: np.ndarray,
                        channel_names: list, levels: list,
                        selected_levels: list, out_path: Path):
    """
    Plot upper-level response for selected pressure levels.

    output shape: (n_channels, n_levels, npix)
    selected_levels: list of level indices to plot.
    """
    n_ch = output.shape[0]
    n_lev = len(selected_levels)
    vmin, vmax = _safe_vlim(output[:, selected_levels, :])

    fig = plt.figure(figsize=(4 * n_ch, 3.5 * n_lev))
    for row, lev_idx in enumerate(selected_levels):
        for col in range(n_ch):
            title = (
                f"{channel_names[col]} {int(levels[lev_idx])}hPa"
                if lev_idx < len(levels)
                else f"{channel_names[col]} L{lev_idx}"
            )
            _mollview(fig, (n_lev, n_ch, row * n_ch + col + 1),
                      output[col, lev_idx].astype(np.float64), nside,
                      title, vmin=vmin, vmax=vmax)

    fig.suptitle("Upper-level response to impulse input", fontsize=12, y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Impulse response analysis for HEALPix weather models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to experiment config .py file.")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Checkpoint epoch to load (default: 100).")
    parser.add_argument(
        "--impulse",
        default="equator",
        choices=["equator", "poles", "north_pole", "south_pole", "meridian", "spot"],
        help="Spatial pattern of the impulse (default: equator).",
    )
    parser.add_argument("--amplitude", type=float, default=5.0,
                        help="Amplitude of the impulse pixels (default: 5.0, in normalised units).")
    parser.add_argument("--lat-width", type=float, default=5.0,
                        help="Half-width in degrees for equator/poles/meridian impulses (default: 5.0).")
    parser.add_argument("--lon", type=float, default=0.0,
                        help="Longitude (degrees) for the meridian impulse (default: 0.0).")
    parser.add_argument("--spot-lat", type=float, default=0.0,
                        help="Latitude of the spot impulse (default: 0.0).")
    parser.add_argument("--spot-lon", type=float, default=0.0,
                        help="Longitude of the spot impulse (default: 0.0).")
    parser.add_argument(
        "--input-channel", default="all",
        help="Which input channel to activate: 'all' or a surface channel index 0-3 (default: all).",
    )
    parser.add_argument("--device", default=None,
                        help="Device: 'cuda', 'cpu', etc. Auto-detected if omitted.")
    parser.add_argument("--out-dir", default="experiments/weather/plots/impulse_response",
                        help="Output directory for plots.")
    parser.add_argument("--upper-levels", default="0,3,6,9,12",
                        help="Comma-separated upper-level indices to visualise (default: 0,3,6,9,12).")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ── load config + model ───────────────────────────────────────────────────
    print(f"Loading config from {args.config} (epoch {args.epoch})...")
    create_config = load_create_config(args.config)
    nside = get_nside(create_config, args.epoch)
    print(f"  nside = {nside}  (npix = {healpix.nside2npix(nside)})")

    model = get_model(create_config, args.epoch, device)
    model.eval()

    # ── build impulse ─────────────────────────────────────────────────────────
    print(f"Building impulse: {args.impulse!r}  amplitude={args.amplitude}  lat_width={args.lat_width}")
    mask = build_impulse_mask(
        nside, args.impulse,
        lat_width=args.lat_width,
        lon=args.lon,
        spot_lat=args.spot_lat,
        spot_lon=args.spot_lon,
    )
    print(f"  {mask.sum()} / {mask.size} pixels activated ({100*mask.mean():.2f}%)")

    batch = build_batch(nside, mask, args.amplitude, args.input_channel, device)

    # ── forward pass ──────────────────────────────────────────────────────────
    print("Running model forward pass...")
    with torch.no_grad():
        output = model(batch)

    surf_out  = output["logits_surface"][0].cpu().numpy()   # (4, npix)
    upper_out = output["logits_upper"][0].cpu().numpy()     # (5, 13, npix)

    # ── output directory ──────────────────────────────────────────────────────
    run_tag = f"{Path(args.config).stem}_ep{args.epoch}_{args.impulse}"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── plots ─────────────────────────────────────────────────────────────────
    print("Saving plots...")

    # 1. Input impulse pattern
    plot_impulse_input(nside, mask, args.amplitude, out_dir / "input_impulse.pdf")

    # 2. Surface output: all 4 channels
    plot_surface_response(nside, surf_out, SURFACE_NAMES, out_dir / "surface_response.pdf")
    plot_surface_response(nside, surf_out, SURFACE_NAMES, out_dir / "surface_response.png")

    # 3. Individual surface channels (larger, standalone)
    for ch_idx, ch_name in enumerate(SURFACE_NAMES):
        vmin, vmax = _safe_vlim(surf_out[ch_idx])
        fig = plt.figure(figsize=(8, 4))
        hp.mollview(
            _to_ring(surf_out[ch_idx].astype(np.float64), nside),
            fig=fig.number,
            title=f"Surface response: {ch_name}",
            min=vmin, max=vmax,
            cmap="RdBu_r",
            notext=True,
            cbar=True,
        )
        hp.graticule(dpar=30, dmer=60, alpha=0.3, verbose=False)
        path = out_dir / f"surface_{ch_name}.pdf"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

    # 4. Upper-level response at selected pressure levels
    upper_level_indices = [int(l) for l in args.upper_levels.split(",")]
    n_levels = upper_out.shape[1]
    valid_levels = [l for l in upper_level_indices if l < n_levels]
    if valid_levels:
        # Try to get pressure level labels from MeteorologicalData if available.
        try:
            from experiments.weather.metrics import MeteorologicalData
            levels = MeteorologicalData().upper.levels
        except Exception:
            levels = list(range(n_levels))
        plot_upper_response(
            nside, upper_out, UPPER_NAMES, levels, valid_levels,
            out_dir / "upper_response.pdf",
        )

    # 5. Summary: absolute max response per surface channel (scalar summary)
    print("\nSurface channel response summary (abs max, normalised units):")
    for ch_idx, ch_name in enumerate(SURFACE_NAMES):
        print(f"  {ch_name:6s}: max={np.abs(surf_out[ch_idx]).max():.4f}  "
              f"mean={surf_out[ch_idx].mean():.4f}")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
