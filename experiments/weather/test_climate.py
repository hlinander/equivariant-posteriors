import xarray as xr
import numpy as np

zarr_path = "/proj/heal_pangu/1990-2017_6h_512x256_equiangular_conservative.zarr/"
ds = xr.open_zarr(zarr_path, consolidated=True)

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(ds)

# ---------------------------------------------------------------
# 1. TIME INTERVALS
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("1. TIME INTERVALS")
print("=" * 70)

# This is a climatology averaged over 1990–2017, indexed by (dayofyear, hour)
print(f"Climatology period: 1990–2017 (from filename)")
print(f"\ndayofyear: {ds.sizes['dayofyear']} values "
      f"(min={int(ds.dayofyear.min())}, max={int(ds.dayofyear.max())})")
print(f"  → covers all calendar days including Feb 29 (leap day)")

hours = ds.hour.values
dt = np.diff(hours)[0] if len(hours) > 1 else None
print(f"\nhour: {ds.sizes['hour']} values = {hours.tolist()} UTC")
print(f"  → temporal resolution: {dt}-hourly snapshots per day")

total_snapshots = ds.sizes["dayofyear"] * ds.sizes["hour"]
print(f"\nTotal climatological snapshots: "
      f"{ds.sizes['dayofyear']} days × {ds.sizes['hour']} hours = {total_snapshots}")

# ---------------------------------------------------------------
# 2. GRID + RESOLUTION
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("2. GRID + RESOLUTION")
print("=" * 70)

lat = ds.latitude.values
lon = ds.longitude.values

print(f"Latitude:  {ds.sizes['latitude']} points, "
      f"range [{lat.min():.3f}, {lat.max():.3f}]°, "
      f"resolution ≈ {np.abs(np.diff(lat)).mean():.3f}°")
print(f"Longitude: {ds.sizes['longitude']} points, "
      f"range [{lon.min():.3f}, {lon.max():.3f}]°, "
      f"resolution ≈ {np.abs(np.diff(lon)).mean():.3f}°")

print(f"\nGrid type: equiangular (uniform lat/lon spacing), "
      f"conservatively regridded")
print(f"Nominal resolution: 512 × 256 → ~5.625° global grid")

if "level" in ds.dims:
    levels = ds.level.values
    print(f"\nVertical levels: {ds.sizes['level']} pressure levels (hPa)")
    print(f"  values: {levels.tolist()}")

# ---------------------------------------------------------------
# 3. SIZE
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3. SIZE")
print("=" * 70)

print(f"Number of variables: {len(ds.data_vars)}")
print(f"\nPer-variable breakdown:")
print(f"{'variable':<35} {'dims':<45} {'size (MB)':>10}")
print("-" * 92)

total_bytes = 0
for name, var in ds.data_vars.items():
    nbytes = var.size * var.dtype.itemsize
    total_bytes += nbytes
    print(f"{name:<35} {str(var.dims):<45} {nbytes / 1e6:>10.2f}")

print("-" * 92)
print(f"{'TOTAL (uncompressed, in memory)':<80} {total_bytes / 1e6:>11.2f} MB")
print(f"{'TOTAL (uncompressed, in memory)':<80} {total_bytes / 1e9:>11.3f} GB")

# On-disk size (compressed) — actual zarr store footprint
import os
def dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

disk_bytes = dir_size(zarr_path)
print(f"\nOn-disk size (compressed zarr store): "
      f"{disk_bytes / 1e6:.2f} MB ({disk_bytes / 1e9:.3f} GB)")
print(f"Compression ratio: {total_bytes / disk_bytes:.2f}x")