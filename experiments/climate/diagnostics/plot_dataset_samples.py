#!/home/x_tagty/equivariant-posteriors/.venv/bin/python3
"""
Plot gas emission fluxes for January 2100, one figure per SSP scenario.
Each figure has 4 subplots (BC, CH4, SO2, CO2) with individual colorbars.

Unit note
---------
The netCDF files carry no 'units' variable attribute, but the global
'reporting_unit' reads "Mass flux of <gas>" and the dataset_category is
"emissions".  CMIP6/input4mips emissions are standardised to kg m⁻² s⁻¹
(see https://docs.google.com/document/d/1pU9IiJvPJwRvIgVaSDdJ4O0Jeorv_2ekEtced2DflvU).
No unit conversion is applied; all four gases are shown in kg m⁻² s⁻¹.

Examples
--------
python plot_gas_jan2100.py                        # viridis, linear
python plot_gas_jan2100.py --log                  # SymLogNorm
python plot_gas_jan2100.py --cmap plasma --log
python plot_gas_jan2100.py --scenarios ssp126 ssp585
python plot_gas_jan2100.py --gases BC CO2
"""

import argparse
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (mirror climateset_data_no_hp.py)
# ---------------------------------------------------------------------------
DATA_DIR     = "/proj/heal_pangu/users/x_tagty/climateset"
INPUT_DIR    = Path(DATA_DIR) / "inputs" / "input4mips"
SPATIAL_RES  = "250_km"
TEMPORAL_RES = "mon"
FIRE_TYPE    = "all-fires"
YEAR         = 2100
MONTH_IDX    = 0   # January = first timestep in the annual file

GAS_FOLDER_MAPPING = {"BC": "BC_sum", "CH4": "CH4_sum", "SO2": "SO2_sum", "CO2": "CO2_sum"}
NO_FIRE_VARS       = {"CO2", "CO2_sum"}

UNIT = "kg m⁻² s⁻¹"   # same for all gases; confirmed from global 'reporting_unit'

# Tiny offset added to exact-zero pixels so log scales don't break.
# Chosen to be ~10 orders of magnitude below the smallest real signal,
# so zeros appear as "near-zero" rather than NaN on log axes.
ZERO_EPS = 1e-20

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_january(scenario: str, gas: str):
    """Load January data. Returns (lat, lon, arr2d) or (None, None, None)."""
    folder   = GAS_FOLDER_MAPPING.get(gas, gas)
    year_dir = INPUT_DIR / scenario / folder / SPATIAL_RES / TEMPORAL_RES / str(YEAR)
    files    = glob.glob(str(year_dir / "*.nc"))
    if not files:
        return None, None, None
    if gas not in NO_FIRE_VARS:
        files = [f for f in files if FIRE_TYPE in f]
    if not files:
        return None, None, None

    ds  = xr.open_dataset(sorted(files)[0], decode_times=False)
    var = next(iter(ds.data_vars))
    arr = ds[var].values                    # (12, lat, lon)
    lat = ds["lat"].values.astype(np.float64)
    lon = ds["lon"].values.astype(np.float64)
    ds.close()
    return lat, lon, arr[MONTH_IDX].astype(np.float64)

# ---------------------------------------------------------------------------
# Norm / colorbar helpers
# ---------------------------------------------------------------------------

def make_norm(data: np.ndarray, log: bool):
    """
    Linear: simple Normalize clipped at 99th percentile.
    Log:    SymLogNorm so that zeros (now shifted to ZERO_EPS) and any
            negative CO2 values are handled without NaNs.
    """
    vmax = float(np.nanpercentile(np.abs(data), 99))

    if not log:
        has_neg = float(np.nanmin(data)) < 0
        vmin    = -vmax if has_neg else 0.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    # For log: linthresh marks the transition from linear (near zero) to log.
    # Set it to just above our epsilon so the linear band is essentially invisible.
    pos_values = data[data > ZERO_EPS]
    linthresh  = float(np.nanmin(pos_values)) * 0.5 if len(pos_values) else ZERO_EPS * 10
    has_neg    = float(np.nanmin(data)) < 0
    vmin       = -vmax if has_neg else ZERO_EPS
    return mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)


def add_colorbar(fig, ax, im, log: bool):
    cbar  = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    label = f"{UNIT}  [log₁₀]" if log else UNIT
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    if not log:
        cbar.formatter = ticker.FormatStrFormatter("%.1e")
        cbar.update_ticks()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="+",
                   default=["ssp126", "ssp245", "ssp370", "ssp585"],
                   metavar="SSP",
                   help="SSP scenarios to plot (default: all four)")
    p.add_argument("--gases", nargs="+",
                   default=["BC", "CH4", "SO2", "CO2"],
                   metavar="GAS",
                   help="Gases to plot (default: BC CH4 SO2 CO2)")
    p.add_argument("--cmap", default="viridis",
                   help="Colormap for all gases (default: viridis). "
                        "Good alternatives: plasma, inferno, magma, cividis, turbo")
    p.add_argument("--log", action="store_true",
                   help="Log scale via SymLogNorm (handles zeros and negative CO2 fluxes)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: same folder as this script)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent
    gases   = args.gases

    ncols = min(2, len(gases))
    nrows = (len(gases) + ncols - 1) // ncols

    for scenario in args.scenarios:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6.5 * ncols, 4 * nrows),
                                 squeeze=False)
        fig.suptitle(scenario.upper(), fontsize=15, fontweight="bold")

        for idx, gas in enumerate(gases):
            ax = axes[idx // ncols][idx % ncols]
            lat, lon, data = load_january(scenario, gas)

            if data is None:
                ax.set_title(f"{gas}  [no data]", fontsize=11)
                ax.axis("off")
                continue

            # Shift exact zeros to ZERO_EPS so log scale is well-defined everywhere
            data = np.where(data == 0.0, ZERO_EPS, data)

            norm = make_norm(data, args.log)
            im   = ax.pcolormesh(lon, lat, data,
                                 cmap=args.cmap, norm=norm,
                                 shading="auto", rasterized=True)

            ax.set_title(gas, fontsize=12, fontweight="bold")
            ax.set_xlabel("Longitude", fontsize=8)
            ax.set_ylabel("Latitude", fontsize=8)
            ax.tick_params(labelsize=7)
            add_colorbar(fig, ax, im, args.log)

        for idx in range(len(gases), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        suffix   = "_log" if args.log else ""
        out_path = out_dir / f"gas_flux_jan2100_{scenario}{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.close()


if __name__ == "__main__":
    main()
