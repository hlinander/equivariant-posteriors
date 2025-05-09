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
        return "_".join([f"{key}_{self.__dict__[key]}" for key in keys])

    def cache_name(self):
        keys = sorted(self.__dict__.keys())
        keys.remove("cache")
        keys.remove("lead_time_days")
        keys.remove("end_year")
        return "_".join([f"{key}_{self.__dict__[key]}" for key in keys])

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
    n_surface: int
    n_upper: int


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
    day_index: int, start_year: int, end_year: int
) -> cdstest.ERA5SampleConfig:
    target_date = day_index_to_datetime(day_index, start_year, end_year)
    return cdstest.ERA5SampleConfig(
        year=target_date.strftime("%Y"),
        month=target_date.strftime("%m"),
        day=target_date.strftime("%d"),
        time="00:00:00",
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


class Climatology(torch.utils.data.Dataset):
    def __init__(self, data_config: DataHPConfig):
        self.ds = DataHP(data_config)
        climate_config = copy.deepcopy(data_config)
        climate_config.start_year = ERA5_START_YEAR_TRAINING
        climate_config.end_year = ERA5_END_YEAR_TRAINING
        self.ds_climate = DataHP(climate_config)
        self.climate_config = climate_config
        self.config = data_config

    def get_meta(self):
        return self.ds.get_meta()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        cache_path = self.ds_climate.get_cache_dir() / "climate"
        climate_indices = day_index_to_climatology_indices(
            index, self.climate_config.start_year, self.climate_config.end_year
        )
        indices_str = "_".join(map(str, climate_indices))
        fs_cache_path = cache_path / f"{indices_str}"
        fs_cache_path_tmp = cache_path / f"{indices_str}_constructing"
        names = dict(
            climate_target_surface="climate_target_surface.npy",
            climate_target_upper="climate_target_upper.npy",
        )
        item_dict = dict(
            sample_id=index,
        )
        if fs_cache_path.is_dir() and self.config.cache:
            print("Climate cache")
            item_dict.update(self.ds[index])
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

            item_dict.update(self.ds[index])
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
        # self.masks = masks.load_mask_hp(data_config.nside)

    @staticmethod
    def data_spec(config: DataHPConfig):
        return DataSpecHP(nside=config.nside, n_surface=4, n_upper=5)

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
        return env().paths.datasets / self.config.cache_name()

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
        if self.config.lead_time_days == 1:
            return self._get_24h(idx)
        else:
            sample = self._get_24h(idx)
            target = self._get_24h(idx + self.config.lead_time_days - 1)
            sample["target_surface"] = target["target_surface"]
            sample["target_upper"] = target["target_upper"]
            sample["prediction_timedelta_hours"] = 24 * self.config.lead_time_days
            return sample

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
            time=np.datetime_as_string(
                np.datetime64(
                    day_index_to_datetime(
                        idx, self.config.start_year, self.config.end_year
                    )
                )
            ),
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
        return days_between_years(self.config.start_year, self.config.end_year)


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
