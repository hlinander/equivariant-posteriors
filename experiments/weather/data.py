import json
import torch
import numpy as np
from datetime import datetime, timedelta
import shutil


# from lib.models.healpix.swin_hp_transformer import SwinHPTransformerConfig

# from lib.models.mlp import MLPConfig
from lib.serialize_human import serialize_human
from lib.compute_env import env

from dataclasses import dataclass
import healpix
import xarray as xr
import experiments.weather.cdsmontly as cdstest


def numpy_to_xds(np_array, xds_template):
    transformed_ds = xr.Dataset()
    for i, var_name in enumerate(xds_template.data_vars):
        transformed_ds[var_name] = xr.DataArray(
            np_array[i], dims=xds_template.dims, coords=xds_template.coords
        )
    return transformed_ds


@dataclass
class DataHPConfig:
    nside: int = 64
    version: int = 10

    def serialize_human(self):
        return serialize_human(self.__dict__)


@dataclass
class DataSpecHP:
    nside: int
    n_surface: int
    n_upper: int


def days_between_years(start_year: int, end_year: int) -> int:
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    return (end_date - start_date).days


def day_index_to_era5_config(
    day_index: int, start_year: int, end_year: int
) -> cdstest.ERA5SampleConfig:
    start_date = datetime(start_year, 1, 1)
    target_date = start_date + timedelta(days=day_index)
    return cdstest.ERA5SampleConfig(
        year=target_date.strftime("%Y"),
        month=target_date.strftime("%m"),
        day=target_date.strftime("%d"),
        time="00:00:00",
    )


class DataHP(torch.utils.data.Dataset):
    def __init__(self, data_config: DataHPConfig):
        self.config = data_config

    @staticmethod
    def data_spec(config: DataHPConfig):
        return DataSpecHP(
            nside=config.nside,
            n_surface=4,
            n_upper=5
            # input_shape=torch.Size([4, healpix.nside2npix(config.nside)]),
            # output_shape=torch.Size([4, healpix.nside2npix(config.nside)]),
            # target_shape=torch.Size([4, healpix.nside2npix(config.nside)]),
        )

    def e5_to_numpy(self, e5xr):
        stats = deserialize_dataset_statistics(self.config.nside)
        npix = healpix.nside2npix(self.config.nside)
        hlong, hlat = healpix.pix2ang(
            self.config.nside, np.arange(0, npix, 1), lonlat=True, nest=True
        )
        hlong = np.mod(hlong, 360)
        xlong = xr.DataArray(hlong, dims="z")
        xlat = xr.DataArray(hlat, dims="z")

        def interpolate(variable: xr.DataArray):
            xhp = variable.interp(
                latitude=xlat, longitude=xlong, kwargs={"fill_value": None}
            )
            hp_image = np.array(xhp.to_array().to_numpy(), dtype=np.float32)
            # hp_image = (hp_image - hp_image.mean(axis=1, keepdims=True)) / hp_image.std(
            # axis=1, keepdims=True
            # )
            return hp_image

        hp_surface = interpolate(e5xr.surface)
        hp_upper = interpolate(e5xr.upper)
        hp_surface = (hp_surface - stats.item()["mean_surface"]) / stats.item()[
            "std_surface"
        ]
        hp_upper = (hp_upper - stats.item()["mean_upper"]) / stats.item()["std_upper"]
        # max_vals = np.amax(hp_image, axis=1, keepdims=True)
        # hp_image = hp_image / max_vals
        # breakpoint()
        return hp_surface, hp_upper

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
            year="1999", month="01", day="01", time="00:00:00"
        )
        e5s = cdstest.get_era5_sample(e5sc)
        return e5s

    def get_cache_dir(self):
        return (
            env().paths.datasets
            / f"era5_lite_np_cache_normalized_nside_{self.config.nside}"
        )

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()
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
        )
        if fs_cache_path.is_dir():
            for key, filename in names.items():
                item_dict[key] = np.load(fs_cache_path / filename)
        else:
            e5s_input_config = day_index_to_era5_config(idx, 2007, 2017)
            e5s_target_config = cdstest.add_timedelta(e5s_input_config, days=1)
            # print("Get ERA5 sample")
            e5s = cdstest.get_era5_sample(e5s_input_config)
            e5target = cdstest.get_era5_sample(e5s_target_config)
            # print("Get ERA5 sample done")
            hp_surface, hp_upper = self.e5_to_numpy(e5s)
            hp_target_surface, hp_target_upper = self.e5_to_numpy(e5target)
            # print("To numpy done")
            data_dict = dict(
                input_surface=hp_surface,
                input_upper=hp_upper,
                target_surface=hp_target_surface,
                target_upper=hp_target_upper,
            )
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

        return item_dict

    def __len__(self):
        return days_between_years(2007, 2017)


def deserialize_dataset_statistics(nside):
    ds = DataHP(DataHPConfig(nside=nside))
    return np.load(ds.get_cache_dir() / "statistics.npy", allow_pickle=True)


def serialize_dataset_statistics(nside):
    ds = DataHP(DataHPConfig(nside=nside))

    mean_surface = None
    mean_x2_surface = None
    mean_upper = None
    mean_x2_upper = None

    n_samples = 0
    for idx, sample in enumerate(ds):
        # print("Start mean")
        if mean_surface is None:
            mean_surface = sample["input_surface"].mean(axis=1, keepdims=True)
            mean_upper = sample["input_upper"].mean(axis=1, keepdims=True)
            mean_x2_surface = (sample["input_surface"] ** 2).mean(axis=1, keepdims=True)
            mean_x2_upper = (sample["input_upper"] ** 2).mean(axis=1, keepdims=True)
        else:
            mean_surface += sample["input_surface"].mean(axis=1, keepdims=True)
            mean_upper += sample["input_upper"].mean(axis=1, keepdims=True)
            mean_x2_surface += (sample["input_surface"] ** 2).mean(
                axis=1, keepdims=True
            )
            mean_x2_upper += (sample["input_upper"] ** 2).mean(axis=1, keepdims=True)
        n_samples += 1

        # E((X - E(x))^2) = E(X^2 - 2XE(x)+ E(x)^2) = E(X^2 - 2 E(X)E(X) + E(x)^2) = E(X^2) - E(X)^2
        # std_surface = np.sqrt(mean_x2_surface / n_samples - (mean_surface / n_samples) ** 2)
        print(f"{idx}")
        # print("End mean")
        # print(f"mean surface: {mean_surface / n_samples}, std surface: {std_surface}")
        # hp_image = (hp_image - hp_image.mean(axis=1, keepdims=True)) / hp_image.std(
        # axis=1, keepdims=True
        # )
        # break

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
    np.save(ds.get_cache_dir() / "statistics.npy", statistics_dict)
    print(f"Saved npy {ds.get_cache_dir() / 'statistics.npy'}")
    # stats = np.load(ds.get_cache_dir() / "statistics.npy", allow_pickle=True)
