from pathlib import Path
import numpy as np
import xarray as xr
from experiments.weather.data import interpolate_dh_to_hp
from experiments.weather.masks.masks import mask_hp_path


mask_path = Path(__file__).parent
# Load data
land_mask = np.load(mask_path / "land_mask.npy")
soil_type = np.load(mask_path / "soil_type.npy")
topography = np.load(mask_path / "topography.npy")

# Define lat/lon coordinates
lat = np.linspace(90, -90, 721)  # Adjust if different
lon = np.linspace(0, 360, 1440, endpoint=False)

# Create xarray Dataset
ds = xr.Dataset(
    {
        "land_mask": (["latitude", "longitude"], land_mask),
        "soil_type": (["latitude", "longitude"], soil_type),
        "topography": (["latitude", "longitude"], topography),
    },
    coords={"latitude": lat, "longitude": lon},
)


def normalize(da):
    return (da - da.mean()) / da.std()


# Apply normalization
ds_normalized = ds.map(normalize)

hp_np = interpolate_dh_to_hp(256, ds_normalized)
np.save(mask_hp_path(256), hp_np)

hp_np = interpolate_dh_to_hp(64, ds_normalized)
np.save(mask_hp_path(64), hp_np)
