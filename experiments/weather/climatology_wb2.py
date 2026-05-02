"""
WeatherBench2 climatology reader for the 1990-2017 6-hourly dataset.

Zarr layout  (all spatial arrays):  (4, 366, [13,] 512, 256)
  dim 0  – hour slot  : 0→0 UTC, 1→6 UTC, 2→12 UTC, 3→18 UTC
  dim 1  – day of year: 1-indexed (1–366), index = doy - 1
  dim 2  – pressure level (upper-level vars only): ascending hPa [50…1000]
  dim -2 – longitude  : 512 points  0 → 359.296875°
  dim -1 – latitude   : 256 points  -89.648° (S) → 89.648° (N)

Model channel ordering
  surface : [msl, u10, v10, t2m]           shape (4, npix)
  upper   : [z, q, t, u, v]  levels 1000→50  shape (5, 13, npix)
"""

from __future__ import annotations

import numpy as np
import zarr
import healpix as hp
from pathlib import Path


WB2_ZARR_PATH = Path(
    "/proj/heal_pangu/1990-2017_6h_512x256_equiangular_conservative.zarr/"
)

# zarr variable name → model channel index for surface
_SURFACE_VARS = [
    "mean_sea_level_pressure",    # 0 – msl
    "10m_u_component_of_wind",    # 1 – u10
    "10m_v_component_of_wind",    # 2 – v10
    "2m_temperature",             # 3 – t2m
]

# zarr variable name → model channel index for upper
_UPPER_VARS = [
    "geopotential",               # 0 – z
    "specific_humidity",          # 1 – q
    "temperature",                # 2 – t
    "u_component_of_wind",        # 3 – u
    "v_component_of_wind",        # 4 – v
]

# Zarr pressure levels (ascending hPa); model wants descending → reversed
_ZARR_LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]



def _precompute_bilinear(lat_grid: np.ndarray, lon_grid: np.ndarray,
                          nside: int) -> tuple:
    """
    Precompute bilinear interpolation indices and weights from a regular
    lat/lon grid to every HEALPix pixel (nested ordering).

    lat_grid : (H,) ascending latitudes   (degrees, e.g. -89.6 … 89.6)
    lon_grid : (W,) ascending longitudes  (degrees, e.g. 0 … 359.3)

    Returns six arrays, each of shape (npix,):
        i0, i1 – lower/upper lat indices   (i1 = i0+1, clamped)
        j0, j1 – lower/upper lon indices   (j1 wraps for lon periodicity)
        wy     – weight for i1 (lat fraction)
        wx     – weight for j1 (lon fraction)

    Interpolated value for a (…, W, H) array `a`:
        out = a[…, i0, j0]*(1-wx)*(1-wy) + a[…, i0, j1]*wx*(1-wy)
            + a[…, i1, j0]*(1-wx)*wy     + a[…, i1, j1]*wx*wy
    """
    npix = hp.nside2npix(nside)
    hlon, hlat = hp.pix2ang(nside, np.arange(npix), lonlat=True, nest=True)
    hlon = np.mod(hlon, 360.0)

    H = len(lat_grid)
    W = len(lon_grid)
    dlat = lat_grid[1] - lat_grid[0]
    dlon = lon_grid[1] - lon_grid[0]

    # Latitude (no wrap; clamp to valid range)
    lat_frac = (hlat - lat_grid[0]) / dlat
    i0 = np.clip(np.floor(lat_frac).astype(np.int32), 0, H - 2)
    wy = np.clip(lat_frac - i0, 0.0, 1.0).astype(np.float32)
    i1 = i0 + 1

    # Longitude (periodic: wrap at 360°)
    lon_frac = (hlon - lon_grid[0]) / dlon
    j0 = np.floor(lon_frac).astype(np.int32) % W
    wx = (lon_frac - np.floor(lon_frac)).astype(np.float32)
    j1 = (j0 + 1) % W

    return i0, i1, j0, j1, wy, wx


def _apply_bilinear(a: np.ndarray,
                    i0, i1, j0, j1, wy, wx) -> np.ndarray:
    """
    Apply precomputed bilinear weights to array `a`.

    a   : (…, W, H)  – any number of leading dims (channels, levels)
    out : (…, npix)
    """
    return (a[..., j0, i0] * (1 - wx) * (1 - wy)
          + a[..., j1, i0] *      wx  * (1 - wy)
          + a[..., j0, i1] * (1 - wx) *      wy
          + a[..., j1, i1] *      wx  *      wy).astype(np.float32)


class WeatherBenchClimatology:
    """
    Reads the WeatherBench2 1990-2017 6-hourly climatology zarr and returns
    surface + upper climate means on the HEALPix grid, normalized with the
    same statistics as DataHP.

    Usage
    -----
    clim = WeatherBenchClimatology(nside=64)
    surface, upper = clim.get(day_of_year=32, hour_utc=6)
    # surface: (4, 49152)   upper: (5, 13, 49152)
    """

    def __init__(self,
                 nside: int = 64,
                 normalize: bool = True,
                 zarr_path: str | Path | None = None):
        self.nside = nside
        self.normalize = normalize

        zarr_path = Path(zarr_path) if zarr_path is not None else WB2_ZARR_PATH
        self._z = zarr.open(str(zarr_path), mode="r")

        lat = self._z["latitude"][:]   # (256,) south→north
        lon = self._z["longitude"][:]  # (512,) 0→359.3

        # Precompute bilinear interpolation weights once
        self._i0, self._i1, self._j0, self._j1, self._wy, self._wx = (
            _precompute_bilinear(lat, lon, nside)
        )

        if normalize:
            from experiments.weather.data import deserialize_dataset_statistics
            self._stats = deserialize_dataset_statistics(nside).item()



    def get(self, day_of_year: int, hour_utc: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return climatological surface and upper arrays for a given time-of-year.

        Parameters
        ----------
        day_of_year : int
            Day-of-year, 1-indexed (1–366). Feb-29 is valid; if the zarr
            contains only 365 days the last day is used as a fallback.
        hour_utc : int
            Hour in UTC; must be one of {0, 6, 12, 18}.

        Returns
        -------
        surface : np.ndarray, shape (4, npix), dtype float32
            Channels: [msl, u10, v10, t2m].
        upper : np.ndarray, shape (5, 13, npix), dtype float32
            Channels: [z, q, t, u, v].
            Pressure levels (dim 1): [1000, 925, …, 50] hPa.
        """
        if hour_utc not in (0, 6, 12, 18):
            raise ValueError(f"hour_utc must be 0, 6, 12 or 18; got {hour_utc}")

        hour_slot = hour_utc // 6
        n_days = self._z["dayofyear"].shape[0]
        day_idx = min(day_of_year - 1, n_days - 1)  # 0-indexed; clamp for 365-day years

        surface = self._read_surface(hour_slot, day_idx)
        upper   = self._read_upper(hour_slot, day_idx)

        if self.normalize:
            from experiments.weather.data import normalize_sample
            surface, upper = normalize_sample(self._stats, surface, upper)
            surface = surface.astype(np.float32)
            upper   = upper.astype(np.float32)

        return surface, upper

    def get_from_datetime(self, dt) -> tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper accepting a datetime object."""
        from datetime import datetime
        doy = dt.timetuple().tm_yday
        # Round hour to nearest 6-h slot
        hour = (dt.hour // 6) * 6
        return self.get(day_of_year=doy, hour_utc=hour)


    def _read_surface(self, hour_slot: int, day_idx: int) -> np.ndarray:
        """Return (4, npix) surface array on HEALPix grid (not yet normalized)."""
        channels = []
        for var in _SURFACE_VARS:
            raw = np.array(self._z[var][hour_slot, day_idx], dtype=np.float32)
            # raw shape: (512, 256) = (lon, lat)
            channels.append(_apply_bilinear(raw, self._i0, self._i1,
                                            self._j0, self._j1, self._wy, self._wx))
        return np.stack(channels, axis=0)  # (4, npix)

    def _read_upper(self, hour_slot: int, day_idx: int) -> np.ndarray:
        """Return (5, 13, npix) upper array on HEALPix grid (not yet normalized)."""
        channels = []
        for var in _UPPER_VARS:
            raw = np.array(self._z[var][hour_slot, day_idx], dtype=np.float32)
            # raw shape: (13, 512, 256) = (level, lon, lat)
            # Zarr levels: ascending hPa [50…1000]; reverse to get [1000…50]
            raw = raw[::-1].copy()
            hp_map = _apply_bilinear(raw, self._i0, self._i1,
                                     self._j0, self._j1, self._wy, self._wx)
            # hp_map shape: (13, npix)
            channels.append(hp_map)
        return np.stack(channels, axis=0)  # (5, 13, npix)

if __name__ == "__main__":
    from datetime import datetime

    wb = WeatherBenchClimatology()
    date_1 = datetime(2019, 1, 1, 2)
    date_2 = datetime(2019, 1, 1, 0)
    date_3 = datetime(2020, 1, 1, 4)
    surface_1, upper_1 = wb.get_from_datetime(date_1)
    surface_2, upper_2 = wb.get_from_datetime(date_2)
    surface_3, upper_3 = wb.get_from_datetime(date_3)

    # all the surface and upper arrays should be identical since they all round to the same 6-h slot and day of year
    assert np.allclose(surface_1, surface_2) and np.allclose(surface_1, surface_3)
    assert np.allclose(upper_1, upper_2) and np.allclose(upper_1, upper_3)