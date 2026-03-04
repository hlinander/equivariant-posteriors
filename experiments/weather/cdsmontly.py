import tempfile
from pathlib import Path
import xarray as xr
import numpy as np
import cdsapi
from dataclasses import dataclass
import shutil
import urllib3
import os
from datetime import datetime, timedelta
from lib.compute_env import env
from lib.log import log

# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context
WEATHER_BASE = Path(os.getenv("WEATHER_DATASET", env().paths.datasets / "era5_lite_hourly"))
ERA5_GRIB_DATA_PATH = WEATHER_BASE / "era5_grib_monthly"


def get_netcat_tmp_dir():
    if "SNIC_TMP" in os.environ:
        tmpdir = Path(os.environ["SNIC_TMP"])
        if tmpdir.is_dir():
            return tmpdir
    return Path("/tmp")


def get_tmp_netcat_file():
    tmp_dir = get_netcat_tmp_dir()
    fd, path_str = tempfile.mkstemp(dir=tmp_dir, suffix=".nc")
    os.close(fd)
    return path_str


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


def add_timedelta(instance: ERA5SampleConfig, days=0, hours=0) -> ERA5SampleConfig:
    date_time_str = f"{instance.year}-{instance.month}-{instance.day} {instance.time}"
    date_time_obj = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
    new_date_time_obj = date_time_obj + timedelta(days=days, hours=hours)
    return ERA5SampleConfig(
        year=new_date_time_obj.strftime("%Y"),
        month=new_date_time_obj.strftime("%m"),
        day=new_date_time_obj.strftime("%d"),
        time=new_date_time_obj.strftime("%H:%M:%S"),
    )


def get_era5_sample(sample_config: ERA5SampleConfig):
    ERA5_GRIB_DATA_PATH.mkdir(exist_ok=True, parents=True)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    if not sample_config.surface_grib_path().is_file():
        log("CDS", f"Downloading {sample_config}")
        log("CDS", f"Target {ERA5_GRIB_DATA_PATH}")
        get_surface_variables(
            sample_config.surface_grib_path(), **sample_config.__dict__
        )
    log("CDS", "Loading surface grib...")
    xr_grib = xr.load_dataset(sample_config.surface_grib_path(), engine="cfgrib")
    surface_nc_file = get_tmp_netcat_file()
    log("CDS", "To netcdf...")
    xr_grib.to_netcdf(surface_nc_file)

    if not sample_config.upper_grib_path().is_file():
        get_upper_variables(sample_config.upper_grib_path(), **sample_config.__dict__)
    log("CDS", "Loading upper grib...")
    xr_grib = xr.load_dataset(sample_config.upper_grib_path(), engine="cfgrib")
    upper_nc_file = get_tmp_netcat_file()
    log("CDS", "To netcdf...")
    xr_grib.to_netcdf(upper_nc_file)

    log("CDS", "Preloading NC...")
    surface_ds = xr.open_dataset(surface_nc_file, backend_kwargs=dict(lock=False))
    upper_ds = xr.open_dataset(upper_nc_file, backend_kwargs=dict(lock=False))
    surface_ds = surface_ds.sel(
        time=f"{sample_config.year}-{sample_config.month}-{sample_config.day} {sample_config.time}"
    ).load()
    upper_ds = upper_ds.sel(
        time=f"{sample_config.year}-{sample_config.month}-{sample_config.day} {sample_config.time}"
    ).load()

    log("CDS", "Closing and removing...")
    surface_ds.close()
    upper_ds.close()
    os.remove(surface_nc_file)
    os.remove(upper_nc_file)

    return ERA5Sample(surface=surface_ds, upper=upper_ds)


def get_surface_variables(target, year="2018", month="09", time="00:00", day=None):
    # urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    c = cdsapi.Client(verify=False, quiet=False)
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
            "day": list(range(1, 32)),
            "time": time,
            "format": "grib",
        },
        target,
    )


def get_upper_variables(target, year="2018", month="09", time="12:00", day=None):
    # urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    c = cdsapi.Client(verify=False, quiet=False)
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
            "day": list(range(1, 32)),
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
