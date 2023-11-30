from pathlib import Path
import xarray as xr
import numpy as np
import cdsapi
from dataclasses import dataclass

# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context
ERA5_GRIB_DATA_PATH = Path("./era5_grib")


@dataclass
class ERA5SampleConfig:
    year: str
    month: str
    day: str
    time: str

    def ident(self):
        return f"{self.year}_{self.month}_{self.day}_{self.time}"

    def surface_ident(self):
        return f"surface_{self.ident()}"

    def upper_ident(self):
        return f"upper_{self.ident()}"

    def surface_file(self):
        return Path(f"{self.surface_ident()}.nc")

    def upper_file(self):
        return Path(f"{self.upper_ident()}.nc")

    def surface_grib(self):
        return Path(f"{self.surface_ident()}.grib")

    def upper_grib(self):
        return Path(f"{self.upper_ident()}.grib")

    def surface_path(self):
        return ERA5_GRIB_DATA_PATH / self.surface_file()

    def upper_path(self):
        return ERA5_GRIB_DATA_PATH / self.upper_file()

    def surface_grib_path(self):
        return ERA5_GRIB_DATA_PATH / self.surface_grib()

    def upper_grib_path(self):
        return ERA5_GRIB_DATA_PATH / self.upper_grib()


@dataclass
class ERA5Sample:
    surface: xr.Dataset  # np.ndarray
    upper: xr.Dataset  # np.ndarray
    # surface_ds: object

    DIMENSIONS = dict(pressure=0, u_wind=1, v_wind=2, temp=3)


def get_era5_sample(sample_config: ERA5SampleConfig):
    ERA5_GRIB_DATA_PATH.mkdir(exist_ok=True, parents=True)

    if not sample_config.surface_path().is_file():
        get_surface_variables(
            sample_config.surface_grib_path(), **sample_config.__dict__
        )
        xr_grib = xr.load_dataset(sample_config.surface_grib_path())
        xr_grib.to_netcdf(sample_config.surface_path())
        # np.save(sample_config.surface_path(), xr_grib.to_array().to_numpy())

    if not sample_config.upper_path().is_file():
        get_upper_variables(sample_config.upper_grib_path(), **sample_config.__dict__)
        xr_grib = xr.load_dataset(sample_config.upper_grib_path())
        xr_grib.to_netcdf(sample_config.upper_path())
        # np.save(sample_config.upper_path(), xr_grib.to_array().to_numpy())

    surface_ds = xr.open_dataset(sample_config.surface_path())
    upper_ds = xr.open_dataset(sample_config.upper_path())
    # surface = np.load(sample_config.surface_path()).astype(np.float32)
    # upper = np.load(sample_config.upper_path()).astype(np.float32)

    # xr_grib = xr.load_dataset(sample_config.surface_grib_path())
    # metadata_ds = xr.Dataset(attrs=xr_grib.attrs)

    # for var in xr_grib.data_vars:
    # Add the variable structure with dummy data (e.g., a scalar 0)
    # metadata_ds[var] = (xr_grib[var].dims, 0, xr_grib[var].attrs)

    # Copy coordinates (without data) and their attributes
    # for coord in xr_grib.coords:
    # metadata_ds.coords[coord] = (xr_grib[coord].dims, 0, xr_grib[coord].attrs)

    return ERA5Sample(surface=surface_ds, upper=upper_ds)


def get_surface_variables(target, year="2018", month="09", day="27", time="12:00"):
    c = cdsapi.Client(verify=False)
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "variable": [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ],
            "product_type": "reanalysis",
            "year": year,
            "month": month,
            "day": day,
            "time": time,
            "format": "grib",
        },
        target,
    )


def get_upper_variables(target, year="2018", month="09", day="27", time="12:00"):
    c = cdsapi.Client(verify=False)
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "variable": [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": [
                1000,
                925,
                850,
                700,
                600,
                500,
                400,
                300,
                250,
                200,
                150,
                100,
                50,
            ],
            "product_type": "reanalysis",
            "year": year,
            "month": month,
            "day": day,
            "time": time,
            "format": "grib",
        },
        target,
    )


if __name__ == "__main__":
    c = cdsapi.Client()
    target_surface = Path("./download_surface.grib")
    if not target_surface.is_file():
        get_surface_variables(target_surface)

    target_upper = Path("./download_upper.grib")
    if not target_upper.is_file():
        get_upper_variables(target_upper)

    xr_grib = xr.load_dataset(target_surface)
    np.save("surface.npy", xr_grib.to_array().to_numpy())

    xr_grib = xr.load_dataset(target_upper)
    np.save("upper.npy", xr_grib.to_array().to_numpy())
    # breakpoint()
