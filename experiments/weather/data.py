import os
import time
import copy
import json
import torch
import numpy as np
import math
from pathlib import Path
from datetime import datetime, timedelta
import shutil
from multiprocessing import Pool
from datetime import datetime, timedelta
import tqdm


# from lib.models.healpix.swin_hp_transformer import SwinHPTransformerConfig

# from lib.models.mlp import MLPConfig
from lib.serialize_human import serialize_human
from lib.compute_env import env

from dataclasses import dataclass
import healpix
import chealpix
import xarray as xr
import experiments.weather.cdsmontly as cdstest
import experiments.weather.masks.masks as masks


def numpy_to_xds(np_array, xds_template):
    transformed_ds = xr.Dataset()
    for i, var_name in enumerate(xds_template.data_vars):
        transformed_ds[var_name] = xr.DataArray(
            np_array[i], dims=xds_template.dims, coords=xds_template.coords
        )
    return transformed_ds


ERA5_START_YEAR_TRAINING = 2007
ERA5_END_YEAR_TRAINING = 2017
ERA5_START_YEAR_TEST = 2019
ERA5_END_YEAR_TEST = 2019
EQUIVAIRIANT_CACHE_YEARS = [2012, 2019]


@dataclass
class DataHPConfig:
    nside: int = 64
    version: int = 10
    driscoll_healy: bool = False
    cache: bool = True
    normalized: bool = True
    start_year: int = 2007
    end_year: int = 2017
    lead_time_days: int = 1

    delta_t: int = 24 # hours, possilbe choices are 2, 4, 6, 12, 24

    def short_name(self):
        return f"era5_{self.start_year}_{self.end_year}"

    def serialize_human(self):
        return serialize_human(self.__dict__)

    def custom_dict(self):
        serialize_dict = copy.deepcopy(self.__dict__)
        if self.start_year == 2007 and self.end_year == 2017:
            del serialize_dict["start_year"]
            del serialize_dict["end_year"]
        del serialize_dict["lead_time_days"]
        return serialize_dict

    def statistics_name(self):
        keys = sorted(self.__dict__.keys())
        keys.remove("cache")
        keys.remove("normalized")
        keys.remove("lead_time_days")
        keys.remove("end_year")
        keys.remove("delta_t")
        return "_".join([f"{key}_{self.__dict__[key]}" for key in keys])

    def cache_name(self):
        keys = sorted(self.__dict__.keys())
        keys.remove("cache")
        keys.remove("lead_time_days")
        keys.remove("end_year")
        keys.remove("delta_t")
        return "_".join([f"{key}_{self.__dict__[key]}" for key in keys])
    
    def cache_name_equivariant(self):
        if not self.delta_t in [2, 4, 6, 8, 12, 24]:
            raise ValueError("delta_t must be one of [2, 4, 6, 8, 12, 24]")
        
        if not self.start_year in EQUIVAIRIANT_CACHE_YEARS or not self.end_year in EQUIVAIRIANT_CACHE_YEARS:
            raise ValueError(f"Equivariant cache is only implemented for the years {EQUIVAIRIANT_CACHE_YEARS}")
        
        if self.start_year != self.end_year:
            raise ValueError("Equivariant cache is only implemented for single year datasets")

        return "era5_lite_equivariant"

    def validation(self):
        ret = copy.deepcopy(self)
        ret.start_year = ERA5_START_YEAR_TEST
        ret.end_year = ERA5_END_YEAR_TEST
        return ret

    def as_hp(self):
        ret = copy.deepcopy(self)
        ret.driscoll_healy = False
        return ret

    def with_lead_time_days(self, days: int):
        ret = copy.deepcopy(self)
        ret.lead_time_days = days
        return ret


@dataclass
class DataSpecHP:
    nside: int
    n_surface: int    # output (target) surface channels — always 4
    n_upper: int
    n_surface_in: int = 0  # input surface channels; may exceed n_surface when extra forcing features (e.g. cos SZA) are added

    def __post_init__(self):
        if self.n_surface_in == 0:
            self.n_surface_in = self.n_surface


def days_between_years(start_year: int, end_year: int) -> int:
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    return (end_date - start_date).days


def days_from_start_year(start_year: int, year: int, month: int, day: int) -> int:
    start_date = datetime(start_year, 1, 1)  # Start of the start year
    specific_date = datetime(year, month, day)  # Specific date
    return (specific_date - start_date).days 


def day_index_to_datetime(day_index: int, start_year: int, end_year: int):
    start_date = datetime(start_year, 1, 1)
    target_date = start_date + timedelta(days=day_index)
    return target_date


def day_index_to_era5_config(
    day_index: int, start_year: int, end_year: int, time_str: str = "00:00:00"
) -> cdstest.ERA5SampleConfig:
    target_date = day_index_to_datetime(day_index, start_year, end_year)
    return cdstest.ERA5SampleConfig(
        year=target_date.strftime("%Y"),
        month=target_date.strftime("%m"),
        day=target_date.strftime("%d"),
        time=time_str,
    )


def day_index_to_climatology_indices(day_index, start_year, end_year):
    day_config = day_index_to_era5_config(day_index, start_year, end_year)
    indices = []
    for year in range(start_year, end_year + 1):
        indices.append(
            days_from_start_year(
                start_year, year, int(day_config.month), int(day_config.day)
            )
        )
    return indices


def index_to_datetime(index: int, config: "DataHPConfig") -> datetime:
    """Convert a DataHP dataset index to its corresponding input datetime."""
    if config.delta_t == 24:
        return day_index_to_datetime(index, config.start_year, config.end_year)
    raw_2h = index * config.delta_t // 2
    year_offset = raw_2h // (12 * 365)
    slot_in_year = raw_2h % (12 * 365)
    return datetime(config.start_year + year_offset, 1, 1) + timedelta(hours=slot_in_year * 2)


def datetime_to_dataset_index(dt: datetime, config: "DataHPConfig") -> int:
    """Convert a datetime to a DataHP dataset index for the given config."""
    if config.delta_t == 24:
        return days_from_start_year(config.start_year, dt.year, dt.month, dt.day)
    year_offset = dt.year - config.start_year
    hour_of_year = int((dt - datetime(dt.year, 1, 1)).total_seconds() // 3600)
    slot_in_year = hour_of_year // 2  # data stored every 2 hours
    total_2h = year_offset * (12 * 365) + slot_in_year
    return total_2h * 2 // config.delta_t


def index_to_climatology_indices(index: int, data_config: "DataHPConfig", climate_config: "DataHPConfig") -> list:
    """Return climate dataset indices matching the same time-of-year as `index` in data_config.

    For sub-daily delta_t the hour is also matched; for daily (delta_t=24) only month/day.
    Feb 29 entries in non-leap training years are silently skipped.
    """
    dt = index_to_datetime(index, data_config)
    match_hour = climate_config.delta_t != 24

    indices = []
    for year in range(climate_config.start_year, climate_config.end_year + 1):
        try:
            if match_hour:
                target_dt = datetime(year, dt.month, dt.day, dt.hour, 0, 0)
            else:
                target_dt = datetime(year, dt.month, dt.day, 0, 0, 0)
        except ValueError:
            continue
        indices.append(datetime_to_dataset_index(target_dt, climate_config))
    return indices




class Climatology(torch.utils.data.Dataset):
    def __init__(self, data_config: DataHPConfig, use_wb2_clim: bool = False):
        self.ds = DataHP(data_config)
        self.config = data_config
        self.use_wb2_clim = use_wb2_clim

        if use_wb2_clim:
            from experiments.weather.climatology_wb2 import WeatherBenchClimatology
            self._wb2_clim = WeatherBenchClimatology(
                nside=data_config.nside, normalize=True
            )
        else:
            climate_config = copy.deepcopy(data_config)
            # Force daily resolution: _get_24h treats indices as day-counts, not 2h-slots.
            # Multi-year datasets can't use the equivariant cache and fall through to _get_24h,
            # so a sub-daily delta_t here would misinterpret 2h-slot indices as day offsets,
            # producing dates far in the future and causing CDS API failures.
            self.ds_climate = DataHP(climate_config)
            self.climate_config = climate_config

    def get_meta(self):
        return self.ds.get_meta()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item_dict = dict(sample_id=index)
        item_dict.update(self.ds[index])

        if self.use_wb2_clim:
            dt = index_to_datetime(index, self.config)
            doy = dt.timetuple().tm_yday
            hour = (dt.hour // 6) * 6

            cache_root = self.ds.get_cache_dir() / "climate_wb2"
            cache_dir = cache_root / f"doy{doy:03d}_h{hour:02d}"
            cache_tmp = cache_root / f"doy{doy:03d}_h{hour:02d}_constructing"
            surf_path = cache_dir / "climate_target_surface.npy"
            upper_path = cache_dir / "climate_target_upper.npy"

            if cache_dir.is_dir() and self.config.cache:
                clim_surface = np.load(surf_path)
                clim_upper = np.load(upper_path)
            else:
                clim_surface, clim_upper = self._wb2_clim.get(doy, hour)
                if self.config.cache:
                    cache_tmp.mkdir(parents=True, exist_ok=True)
                    np.save(cache_tmp / "climate_target_surface.npy", clim_surface)
                    np.save(cache_tmp / "climate_target_upper.npy", clim_upper)
                    if not cache_dir.is_dir():
                        shutil.move(cache_tmp, cache_dir)

            item_dict["climate_target_surface"] = clim_surface.astype(np.float32)
            item_dict["climate_target_upper"] = clim_upper.astype(np.float32)
            return item_dict

        cache_path = self.ds_climate.get_cache_dir() / "climate"
        climate_indices = index_to_climatology_indices(
            index, self.config, self.climate_config
        )
        indices_str = "_".join(map(str, climate_indices))
        fs_cache_path = cache_path / f"{indices_str}"
        fs_cache_path_tmp = cache_path / f"{indices_str}_constructing"
        names = dict(
            climate_target_surface="climate_target_surface.npy",
            climate_target_upper="climate_target_upper.npy",
        )
        if fs_cache_path.is_dir() and self.config.cache:
            print("Climate cache")
            for key, filename in names.items():
                item_dict[key] = np.load(fs_cache_path / filename).astype(np.float32)
        else:
            print("Climate hydrating cache")
            accum_dict = dict()
            for cidx in climate_indices:
                data_dict = self.ds_climate[cidx]
                keys = ["target_surface", "target_upper"]
                for key in keys:
                    if key not in accum_dict:
                        accum_dict[key] = data_dict[key].astype(np.float64)
                    else:
                        accum_dict[key] += data_dict[key].astype(np.float64)
            for key in accum_dict.keys():
                accum_dict[key] /= len(climate_indices)

            item_dict["climate_target_surface"] = accum_dict["target_surface"]
            item_dict["climate_target_upper"] = accum_dict["target_upper"]
            if self.config.cache:
                fs_cache_path_tmp.mkdir(parents=True, exist_ok=True)
                for key, filename in names.items():
                    np.save(fs_cache_path_tmp / filename, item_dict[key])

                if not fs_cache_path.is_dir():
                    shutil.move(fs_cache_path_tmp, fs_cache_path)

        return item_dict


def interpolate_dh_to_hp(nside, variable: xr.DataArray):
    npix = healpix.nside2npix(nside)
    hlong, hlat = healpix.pix2ang(nside, np.arange(0, npix, 1), lonlat=True, nest=True)
    hlong = np.mod(hlong, 360)
    xlong = xr.DataArray(hlong, dims="z")
    xlat = xr.DataArray(hlat, dims="z")

    xhp = variable.interp(latitude=xlat, longitude=xlong, kwargs={"fill_value": None})
    hp_image = np.array(xhp.to_array().to_numpy(), dtype=np.float32)
    return hp_image


def e5_to_numpy_hp(e5xr, nside: int, normalized: bool):

    hp_surface = interpolate_dh_to_hp(nside, e5xr.surface)
    hp_upper = interpolate_dh_to_hp(nside, e5xr.upper)

    if normalized:
        stats = deserialize_dataset_statistics(nside)
        hp_surface, hp_upper = normalize_sample(stats.item(), hp_surface, hp_upper)

    return hp_surface, hp_upper


def batch_to_weatherbench2(
    input_batch, output_batch, nside: int, normalized: bool, lead_days: int = 1
):
    xds = numpy_hp_to_e5(
        output_batch["logits_surface"],
        output_batch["logits_upper"],
        times=input_batch["time"],
        nside=nside,
        normalized=normalized,
        lead_days=lead_days,
    )
    return xds


def numpy_hp_to_e5(
    hp_surface, hp_upper, times, nside: int, normalized: bool, lead_days: int = 1
):
    if torch.isnan(hp_surface).any():
        print("NaNs!")
    if torch.isnan(hp_upper).any():
        print("NaNs!")

    if normalized:
        stats = deserialize_dataset_statistics(nside)
        hp_surface, hp_upper = denormalize_sample(stats.item(), hp_surface, hp_upper)

    def regrid_to_original(hp_data, dims):
        xhp = xr.DataArray(hp_data, dims=dims)
        lat = np.linspace(-90.0, 90.0, 721, endpoint=True)
        lon = np.linspace(0.0, 360.0, 1440, endpoint=False)
        lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
        idxs = healpix.ang2pix(nside, lon2, lat2, nest=False, lonlat=True)
        idxs = chealpix.ring2nest(nside, idxs)
        idxhp = xr.DataArray(
            idxs,
            dims=["latitude", "longitude"],
            coords={"latitude": lat, "longitude": lon},
        )
        return xhp.interp(z=idxhp, kwargs={"fill_value": "extrapolate"})

    surface = regrid_to_original(
        np.expand_dims(hp_surface, 1),
        dims=("time", "prediction_timedelta", "variable", "z"),
    )
    upper = regrid_to_original(
        np.expand_dims(hp_upper, 1),
        dims=("time", "prediction_timedelta", "variable", "level", "z"),
    )

    if surface.isnull().any():
        print("surface NaN after regrid")
    if upper.isnull().any():
        print("upper NaN after regrid")

    surface = surface.assign_coords(
        {
            "time": [np.datetime64(t) for t in times],
            "prediction_timedelta": [np.timedelta64(timedelta(days=lead_days))],
        }
    )
    upper = upper.assign_coords(
        {
            "time": [np.datetime64(t) for t in times],
            "prediction_timedelta": [np.timedelta64(timedelta(days=lead_days))],
            "level": np.array(
                [
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
                dtype=np.int32,
            ),
        }
    )

    surface_ds = surface.to_dataset("variable").rename_vars(
        {
            0: "mean_sea_level_pressure",
            1: "10m_u_component_of_wind",
            2: "10m_v_component_of_wind",
            3: "2m_temperature",
        }
    )
    upper_ds = upper.to_dataset("variable").rename_vars(
        {
            0: "geopotential",
            1: "specific_humidity",
            2: "temperature",
            3: "u_component_of_wind",
            4: "v_component_of_wind",
        }
    )
    # e5xr = xr.Dataset({"surface": surface, "upper": upper})
    # Identify misaligned coordinates

    final = xr.merge(
        [surface_ds.drop_vars("z"), upper_ds.drop_vars("z")],
    )  # surface, upper

    return final


class DataHP(torch.utils.data.Dataset):
    def __init__(self, data_config: DataHPConfig):
        self.config = data_config
        self.use_equivariant_cache = self.config.start_year in EQUIVAIRIANT_CACHE_YEARS and \
                                     self.config.end_year in EQUIVAIRIANT_CACHE_YEARS and \
                                     self.config.start_year == self.config.end_year and \
                                     self.config.delta_t in [2, 4, 6, 8, 12, 24] 
        # self.masks = masks.load_mask_hp(data_config.nside)

    @staticmethod
    def data_spec(config: DataHPConfig):
        return DataSpecHP(nside=config.nside, n_surface=4, n_upper=5)

    @staticmethod
    def collate_fn(batch):
        """Collate that keeps 'time' as a list of datetimes (not a tensor)."""
        from torch.utils.data.dataloader import default_collate
        non_datetime = [{k: v for k, v in item.items() if k != "time"} for item in batch]
        collated = default_collate(non_datetime)
        collated["time"] = [item["time"] for item in batch]
        return collated

    def e5_to_numpy(self, e5xr):
        if self.config.driscoll_healy:
            return self.e5_to_numpy_dh(e5xr)
        else:
            return e5_to_numpy_hp(e5xr, self.config.nside, self.config.normalized)

    def dh_resolution(self):
        n_pixels = healpix.nside2npix(self.config.nside)
        # np_dh = w * h / 2^(2n)
        # 2^(2n) = w * h / np_dh
        # 2n = log2(w*h/np_dh)
        # n = log2(w*h/np_dh) / 2
        scale_factor = math.log2(1440 * 721 / n_pixels) / 2
        lon = math.ceil(1440 / 2**scale_factor)
        lat = math.ceil(721 / 2**scale_factor)
        return dict(lat=lat, lon=lon)

    def e5_to_numpy_dh(self, e5xr):
        n_pixels = healpix.nside2npix(self.config.nside)
        # np_dh = w * h / 2^(2n)
        # 2^(2n) = w * h / np_dh
        # 2n = log2(w*h/np_dh)
        # n = log2(w*h/np_dh) / 2
        scale_factor = math.log2(1440 * 721 / n_pixels) / 2
        lon = math.ceil(1440 / 2**scale_factor)
        lat = math.ceil(721 / 2**scale_factor)

        new_lon = np.linspace(
            e5xr.surface.longitude[0], e5xr.surface.longitude[-1], lon
        )
        new_lat = np.linspace(e5xr.surface.latitude[0], e5xr.surface.latitude[-1], lat)

        def interpolate(variable: xr.DataArray):
            xhp = variable.interp(
                latitude=new_lat, longitude=new_lon, kwargs={"fill_value": None}
            )
            np_image = np.array(xhp.to_array().to_numpy(), dtype=np.float32)
            return np_image

        dh_surface = interpolate(e5xr.surface)
        dh_upper = interpolate(e5xr.upper)
        if self.config.normalized:
            stats = deserialize_dataset_statistics(self.config.nside)
            dh_surface, dh_upper = normalize_sample(stats.item(), dh_surface, dh_upper)

        return dh_surface, dh_upper

    def get_driscoll_healy(self, idx):
        e5sc = cdstest.ERA5SampleConfig(
            year="1999", month="01", day="01", time="00:00:00"
        )
        e5s = cdstest.get_era5_sample(e5sc)
        e5_target_config = cdstest.ERA5SampleConfig(
            year="1999", month="01", day="01", time="03:00:00"
        )
        e5target = cdstest.get_era5_sample(e5_target_config)
        return e5s, e5target

    def get_template_e5s(self):
        e5sc = cdstest.ERA5SampleConfig(
            year="2007", month="01", day="01", time="00:00:00"
        )
        e5s = cdstest.get_era5_sample(e5sc)
        return e5s

    def get_meta(self):
        temp = self.get_template_e5s()

        def get_metas(variable):
            names = [str(var) for var in variable.data_vars.variables]
            long_names = [
                variable.data_vars.variables[var].attrs["long_name"] for var in names
            ]
            units = [variable.data_vars.variables[var].attrs["units"] for var in names]
            return dict(
                names=names,
                long_names=long_names,
                units=units,
            )

        surface_metas = get_metas(temp.surface)
        upper_metas = get_metas(temp.upper)

        return dict(surface=surface_metas, upper=upper_metas)

    def get_cache_dir(self):
        if not self.use_equivariant_cache:
            return env().paths.datasets / self.config.cache_name()
        else:
            return env().paths.datasets / self.config.cache_name_equivariant()

    def get_statistics_path(self):
        return env().paths.datasets / f"{self.config.statistics_name()}.npy"

    def get_old_cache_dir(self):
        if self.config.driscoll_healy:
            return (
                env().paths.datasets
                / f"era5_lite_np_cache_normalized_{self.config.normalized}_nside_{self.config.nside}_dh"
            )
        else:
            return (
                env().paths.datasets
                / f"era5_lite_np_cache_normalized_{self.config.normalized}_nside_{self.config.nside}"
            )

    def __getitem__(self, idx):
        
        if self.use_equivariant_cache:
            return self._get_th(idx)

        else:
            if self.config.lead_time_days == 1:
                return self._get_24h(idx)
            else:
                sample = self._get_24h(idx)
                target = self._get_24h(idx + self.config.lead_time_days - 1)
                sample["target_surface"] = target["target_surface"]
                sample["target_upper"] = target["target_upper"]
                sample["prediction_timedelta_hours"] = 24 * self.config.lead_time_days
                return sample

    def ds_index_to_era5_config(self, ds_idx):
        start = datetime(self.config.start_year, 1, 1, 0, 0, 0)
        dt = start + timedelta(hours=ds_idx * 2)
        year = str(dt.year)
        month = str(dt.month).zfill(2)
        day = str(dt.day).zfill(2)
        time = dt.strftime("%H:%M:%S")

        return cdstest.ERA5SampleConfig(year=year, month=month, day=day, time=time)

    def load_sample(self, year:int , ds_idx: int, names: dict):
        fs_cache_path = self.get_cache_dir() / f"year_{year}/{ds_idx}" 
        fs_cache_path_tmp = self.get_cache_dir() / f"year_{year}/{ds_idx}_constructing"

        if fs_cache_path.is_dir() and self.config.cache:
            data_dict = dict()
            for key, filename in names.items():
                data_dict[key] = np.load(fs_cache_path / filename).astype(np.float32)
            data_dict["time"] = json.loads(open(fs_cache_path / "era5_config.json").read())["time"]
            data_dict["masks"] = masks.load_mask_hp(self.config.nside)
            return data_dict


        e5s_config = self.ds_index_to_era5_config(ds_idx)
        # print("Get ERA5 sample")
        
        downloaded = False # Flag variable to eventually redownload corrupted samples

        while not downloaded:
            try:
                e5s = cdstest.get_era5_sample(e5s_config)
                downloaded = True
            except Exception as e:
                print(f"Error loading ERA5 sample for config: {e5s_config}")
                print(e)
                
                # Deleting the .grib files associated with the sample to force redownloading, since the error is likely due to corrupted files
                os.remove(e5s_config.surface_grib_path())
                os.remove(e5s_config.upper_grib_path())

                print("Retrying download...")


        # print("Get ERA5 sample done")
        hp_surface, hp_upper = self.e5_to_numpy(e5s)
        # print("To numpy done")
        data_dict = dict(
            surface=hp_surface.astype(np.float32),
            upper=hp_upper.astype(np.float32),
        )
        if self.config.cache:
            fs_cache_path_tmp.mkdir(parents=True, exist_ok=True)
            for key, filename in names.items():
                np.save(fs_cache_path_tmp / filename, data_dict[key])

            open(fs_cache_path_tmp / "era5_config.json", "w").write(
                json.dumps(e5s_config.__dict__, indent=2)
            )

            if not fs_cache_path.is_dir():
                shutil.move(fs_cache_path_tmp, fs_cache_path)

        data_dict["time"] = e5s_config.time
        data_dict["masks"] = masks.load_mask_hp(self.config.nside)
    
        return data_dict

    def _get_th(self, idx):
        idx = idx * self.config.delta_t // 2 # since the data is downloaded every 2 hours, we need to divide by 2 to get the correct index

        year = self.config.start_year + idx // (12 * 365) # since there are (12 * 365) samples per year
        year_target = year
        ds_idx = idx % (12 * 365) # index within the year
        ds_idx_target = ds_idx + self.config.delta_t // 2

        if ds_idx == (12 * 365) - (self.config.delta_t // 2):
            year_target += 1

        names = dict(
            surface="surface.npy",
            upper="upper.npy",
        )
        item_dict = dict(
            sample_id=ds_idx,
            prediction_timedelta_hours=self.config.delta_t,
        )

        sample_dict = self.load_sample(year, ds_idx, names)
        target_dict = self.load_sample(year_target, ds_idx_target, names)

        if self.config.normalized:
            stats = deserialize_dataset_statistics(self.config.nside).item()
            for d in [sample_dict, target_dict]:
                # Normalize on the fly if the cached data is in physical units.
                # Cached entries built with normalized=False have surface values in
                # the tens-of-thousands range (e.g. MSL pressure ~101325 Pa), whereas
                # normalized values are always within a few sigma of zero.
                if np.abs(d["surface"]).max() > 50:
                    surf, upper = normalize_sample(stats, d["surface"], d["upper"])
                    d["surface"] = surf.astype(np.float32)
                    d["upper"] = upper.astype(np.float32)

        input_dt = datetime(year, 1, 1, 0, 0, 0) + timedelta(hours=ds_idx * 2)

        item_dict["input_surface"] = sample_dict["surface"]
        item_dict["input_upper"] = sample_dict["upper"]
        item_dict["target_surface"] = target_dict["surface"]
        item_dict["target_upper"] = target_dict["upper"]
        item_dict["time"] = input_dt
        item_dict["masks"] = sample_dict["masks"]

        return item_dict
        
        
        



    def _get_24h(self, idx):
        if idx >= len(self):
            raise StopIteration()
        fs_cache_path = self.get_cache_dir() / f"{idx}"
        fs_cache_path_tmp = self.get_cache_dir() / f"{idx}_constructing"
        names = dict(
            input_surface="surface.npy",
            input_upper="upper.npy",
            target_surface="target_surface.npy",
            target_upper="target_upper.npy",
        )
        item_dict = dict(
            sample_id=idx,
            time=day_index_to_datetime(idx, self.config.start_year, self.config.end_year),
            prediction_timedelta_hours=24,
        )
        done = False
        while not done:
            try:
                fs_cache_path.is_dir()
                done = True
            except OSError:
                time.sleep(0.5)

        if fs_cache_path.is_dir() and self.config.cache:
            # print("Loading from cache")
            for key, filename in names.items():
                item_dict[key] = np.load(fs_cache_path / filename).astype(np.float32)
        else:
            e5s_input_config = day_index_to_era5_config(
                idx, self.config.start_year, self.config.end_year
            )
            e5s_target_config = cdstest.add_timedelta(e5s_input_config, days=1)
            # print("Get ERA5 sample")
            e5s = cdstest.get_era5_sample(e5s_input_config)
            e5target = cdstest.get_era5_sample(e5s_target_config)
            # print("Get ERA5 sample done")
            hp_surface, hp_upper = self.e5_to_numpy(e5s)
            hp_target_surface, hp_target_upper = self.e5_to_numpy(e5target)
            # print("To numpy done")
            data_dict = dict(
                input_surface=hp_surface.astype(np.float32),
                input_upper=hp_upper.astype(np.float32),
                target_surface=hp_target_surface.astype(np.float32),
                target_upper=hp_target_upper.astype(np.float32),
            )
            if self.config.cache:
                fs_cache_path_tmp.mkdir(parents=True, exist_ok=True)
                for key, filename in names.items():
                    np.save(fs_cache_path_tmp / filename, data_dict[key])

                open(fs_cache_path_tmp / "era5config_input.json", "w").write(
                    json.dumps(e5s_input_config.__dict__, indent=2)
                )
                open(fs_cache_path_tmp / "era5config_target.json", "w").write(
                    json.dumps(e5s_target_config.__dict__, indent=2)
                )

                if not fs_cache_path.is_dir():
                    shutil.move(fs_cache_path_tmp, fs_cache_path)
            # print("Write done")

            item_dict.update(data_dict)

        item_dict["masks"] = masks.load_mask_hp(self.config.nside)

        return item_dict

    def __len__(self):
        return (days_between_years(self.config.start_year, self.config.end_year) - 1) * 24 // self.config.delta_t 
    
@dataclass
class DataHPSubsetConfig:
    data_config: DataHPConfig
    reduction_factor: float = 0.1
    consecutive_samples: int = 1

class DataHPSubset(torch.utils.data.Dataset):
    """
    Since DataHP does not allow to use a subset of a year, this class allows to reduce the size of the dataset.
    """

    def __init__(self, config: DataHPSubsetConfig):
        """
        Args:
            data_config: Configuration for the data
            reduction_factor: Factor by which to reduce the dataset size (0,1] - e.g. 0.1 will use 10% of the data
            consecutive_samples: Number of consecutive samples to include in the subset.

        Example:
            reduction_factor = 0.1, consecutive_samples = 5
            -> keep 5 consecutive samples, then skip about 45, repeating through each year.
        """
        self.original_dataset = DataHP(config.data_config)

        if config.reduction_factor <= 0 or config.reduction_factor > 1:
            raise ValueError("reduction_factor must be in the range (0, 1]")
        self.reduction_factor = config.reduction_factor

        if config.consecutive_samples < 1:
            raise ValueError("consecutive_samples must be at least 1")
        self.consecutive_samples = config.consecutive_samples

        self._subset_indices = self._build_subset_indices()

        self.config = config.data_config

    def _build_subset_indices(self):
        total_len = len(self.original_dataset)
        n_years = self.original_dataset.config.end_year - self.original_dataset.config.start_year + 1

        # Split the total dataset length across years.
        # This is robust even if total_len is not perfectly divisible by n_years.
        base_year_len = total_len // n_years
        remainder = total_len % n_years

        year_lengths = [
            base_year_len + (1 if i < remainder else 0)
            for i in range(n_years)
        ]

        subset_indices = []
        offset = 0

        for year_len in year_lengths:
            target_kept = int(year_len * self.reduction_factor)

            if target_kept == 0 and year_len > 0:
                target_kept = 1

            # If reduction_factor = kept / cycle_len  => cycle_len ~= kept_block / reduction_factor
            cycle_len = max(
                self.consecutive_samples,
                round(self.consecutive_samples / self.reduction_factor)
            )

            kept = 0
            cursor = 0

            while kept < target_kept and cursor < year_len:
                take = min(
                    self.consecutive_samples,
                    target_kept - kept,
                    year_len - cursor
                )

                subset_indices.extend(range(offset + cursor, offset + cursor + take))

                kept += take
                cursor += cycle_len

            offset += year_len

        return subset_indices

    def __getitem__(self, idx):
        real_idx = self._subset_indices[idx]
        return self.original_dataset[real_idx]

    def __len__(self):
        return len(self._subset_indices)

    @staticmethod
    def collate_fn(batch):
        return DataHP.collate_fn(batch)

class DataHPConvConfig(DataHPConfig):
    input_time_dim: int = 1 # Number of input time steps
    output_time_dim: int = 1 # Number of output time steps

    def short_name(self):
        return super().short_name() + "_conv"

class DataHPConv(DataHP):
    """
    This dataset is designed to provide multiple time steps of input data for convolutional models, which can leverage temporal information. 
    It extends the DataHP class to include a sequence of input time steps (input_time_dim) and a single output time step (output_time_dim). 
    The __getitem__ method is overridden to return a sequence of input data and the corresponding target data for the specified time steps.
    """

    def __init__(self, data_config: DataHPConvConfig):
        super().__init__(data_config)

    
    def __getitem__(self, idx):
        input_time_dim = self.config.input_time_dim
        output_time_dim = self.config.output_time_dim

        if idx >= len(self):
            raise StopIteration()

        input_sequence = []
        for i in range(input_time_dim):
            input_sequence.append(super().__getitem__(idx + i))
        
        target_sequence = []
        for i in range(output_time_dim):
            target_sequence.append(super().__getitem__(idx + input_time_dim + i))

        # Stack the input sequence along a new time dimension
        input_surface_sequence = np.stack([item['input_surface'] for item in input_sequence], axis=0) # Shape: (input_time_dim, n_channels, n_pix)
        input_upper_sequence = np.stack([item['input_upper'] for item in input_sequence], axis=0) # Shape: (input_time_dim, n_channels, n_pix)
        target_surface_sequence = np.stack([item['target_surface'] for item in target_sequence], axis=0) # Shape: (output_time_dim, n_channels, n_pix)
        target_upper_sequence = np.stack([item['target_upper'] for item in target_sequence], axis=0) # Shape: (output_time_dim, n_channels, n_pix)

        

        return {
            'sample_id': idx,
            'input_surface': input_surface_sequence,
            'input_upper': input_upper_sequence,
            'target_surface': target_surface_sequence,
            'target_upper': target_upper_sequence,
            'prediction_timedelta_hours': target_sequence[0]['prediction_timedelta_hours'],
        }

    def __len__(self):
        # The length is reduced by the number of input and output time steps to ensure we don't go out of bounds
        return super().__len__() - self.config.input_time_dim - self.config.output_time_dim + 1


def deserialize_dataset_statistics(nside):
    ds = DataHP(DataHPConfig(nside=nside))
    return np.load(ds.get_statistics_path(), allow_pickle=True)


def _fix_dataset_statistics(nside):
    stats = deserialize_dataset_statistics(nside).item()
    std_surface = np.sqrt(
        stats["mean_x2_surface"].astype(np.float64)
        - (stats["mean_surface"].astype(np.float64)) ** 2
    ).astype(np.float32)
    std_upper = np.sqrt(
        stats["mean_x2_upper"].astype(np.float64)
        - (stats["mean_upper"].astype(np.float64)) ** 2
    ).astype(np.float32)
    statistics_dict = dict(
        mean_surface=stats["mean_surface"],
        mean_upper=stats["mean_upper"],
        mean_x2_surface=stats["mean_x2_surface"],
        mean_x2_upper=stats["mean_x2_upper"],
        std_surface=std_surface,
        std_upper=std_upper,
        n_samples=stats["n_samples"],
    )
    print(statistics_dict)
    # np.save(ds.get_cache_dir() / "statistics.npy", statistics_dict)


def normalize_sample(stats, surface, upper):
    if len(stats["mean_surface"].shape) == len(surface.shape):
        norm_surface = (surface - stats["mean_surface"]) / stats["std_surface"]
        norm_upper = (upper - stats["mean_upper"]) / stats["std_upper"]
    else:
        norm_surface = (surface - stats["mean_surface"][..., None]) / stats[
            "std_surface"
        ][..., None]
        norm_upper = (upper - stats["mean_upper"][..., None]) / stats["std_upper"][
            ..., None
        ]
    return norm_surface, norm_upper


def denormalize_sample(stats, surface, upper):
    if len(stats["mean_surface"].shape) == len(surface.shape[1:]):
        denorm_surface = surface * stats["std_surface"] + stats["mean_surface"]
        denorm_upper = upper * stats["std_upper"] + stats["mean_upper"]
    else:
        denorm_surface = (
            surface * stats["std_surface"][..., None] + stats["mean_surface"][..., None]
        )
        denorm_upper = (
            upper * stats["std_upper"][..., None] + stats["mean_upper"][..., None]
        )
    return denorm_surface, denorm_upper


def _get_stats_at_idx(idx_and_nside_tuple):
    idx, nside = idx_and_nside_tuple
    ds = DataHP(DataHPConfig(nside=nside, cache=False, normalized=False))
    sample = ds[idx]
    mean_surface = (
        sample["input_surface"].astype(np.float64).mean(axis=1, keepdims=True)
    )
    mean_upper = sample["input_upper"].astype(np.float64).mean(axis=2, keepdims=True)
    mean_x2_surface = (sample["input_surface"].astype(np.float64) ** 2).mean(
        axis=1, keepdims=True
    )
    mean_x2_upper = (sample["input_upper"].astype(np.float64) ** 2).mean(
        axis=2, keepdims=True
    )
    return mean_surface, mean_upper, mean_x2_surface, mean_x2_upper


def serialize_dataset_statistics(nside, test_with_one_sample=False):
    ds = DataHP(DataHPConfig(nside=nside, cache=False, normalized=False))

    if test_with_one_sample:
        n_samples = 1
        n_samples_left = 1
    else:
        n_samples = len(ds)
        n_samples_left = len(ds)

    sample = ds[0]
    mean_surface = np.zeros_like(
        sample["input_surface"].mean(axis=1, keepdims=True), dtype=np.float64
    )
    mean_upper = np.zeros_like(
        sample["input_upper"].mean(axis=2, keepdims=True), dtype=np.float64
    )
    mean_x2_surface = np.zeros_like(mean_surface)
    mean_x2_upper = np.zeros_like(mean_upper)

    import os
    import time

    batch_size = int(os.getenv("SLURM_CPUS_ON_NODE", f"{os.cpu_count()}"))
    print(f"Starting with batch size {batch_size}")
    idx = 0

    # n_samples_left = 34

    with Pool(batch_size) as p:
        while n_samples_left > 0:
            n_requests = min(batch_size, n_samples_left)
            chunks = zip(range(idx, idx + n_requests), [nside] * n_requests)
            start_time = time.time()
            res = p.map(_get_stats_at_idx, chunks)
            idx += n_requests
            n_samples_left -= n_requests
            for (
                sample_mean_surface,
                sample_mean_upper,
                sample_mean_x2_surface,
                sample_mean_x2_upper,
            ) in res:
                mean_surface += sample_mean_surface
                mean_upper += sample_mean_upper
                mean_x2_surface += sample_mean_x2_surface
                mean_x2_upper += sample_mean_x2_upper

            print(
                f"{n_requests} samples in {(time.time() - start_time):.02f}s, {n_samples_left} left"
            )

    mean_surface = mean_surface / n_samples
    mean_upper = mean_upper / n_samples
    mean_x2_surface = mean_x2_surface / n_samples
    mean_x2_upper = mean_x2_upper / n_samples

    std_surface = np.sqrt(mean_x2_surface - (mean_surface) ** 2)
    std_upper = np.sqrt(mean_x2_upper - (mean_upper) ** 2)
    statistics_dict = dict(
        mean_surface=mean_surface,
        mean_upper=mean_upper,
        mean_x2_surface=mean_x2_surface,
        mean_x2_upper=mean_x2_upper,
        std_surface=std_surface,
        std_upper=std_upper,
        n_samples=n_samples,
    )
    ds.get_cache_dir().mkdir(parents=True, exist_ok=True)
    np.save(ds.get_statistics_path(), statistics_dict)
    print(f"Saved npy {ds.get_statistics_path()}")


if __name__ == "__main__":

    # for year in [2019]:
    #     print(f"Checking year {year} for NaN values")

    #     cnf2 = DataHPConfig(nside=64, start_year=year, end_year=year, normalized=False, delta_t=2)
    #     ds = DataHP(cnf2)


    #     for i in [48, 85, 564, 1128, 1476, 2184]:
    #         corrupted_sample = ds[i]

    #         # Localize where is the nan value in the corrupted sample
    #         input_surface = corrupted_sample["input_surface"]
    #         input_upper = corrupted_sample["input_upper"]

    #         if np.isnan(input_surface).any():
    #             print("NaN values found in input_surface")
    #             nan_indices = np.argwhere(np.isnan(input_surface))
    #             print(f"NaN indices in input_surface: {nan_indices}")
    #         if np.isnan(input_upper).any():
    #             print("NaN values found in input_upper")
    #             nan_indices = np.argwhere(np.isnan(input_upper))
    #             print(f"NaN indices in input_upper: {nan_indices}")
    #             print(f"Number of NaN values in input_upper: {input_upper[nan_indices[:, 0], nan_indices[:, 1], nan_indices[:, 2]].size}")

    print("Testing subset")
    cnf = DataHPConfig(nside=64, start_year=2012, end_year=2012, delta_t=2)
    climate_ds = Climatology(cnf, use_wb2_clim=True)

    for i in range(len(climate_ds)):
        sample = climate_ds[i]
        print(sample["climate_target_surface"].shape, sample["climate_target_upper"].shape)

    