from typing import Dict
import torch
import numpy as np


# from lib.models.healpix.swin_hp_transformer import SwinHPTransformerConfig

# from lib.models.mlp import MLPConfig
from lib.serialization import serialize_human
from lib.data_utils import create_metric_sample_legacy
from lib.train_dataclasses import TrainEpochState
from lib.metric import MetricSample

from dataclasses import dataclass
import healpix
import xarray as xr
import experiments.weather.cdstest as cdstest


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
        npix = healpix.nside2npix(self.config.nside)
        hlong, hlat = healpix.pix2ang(
            self.config.nside, np.arange(0, npix, 1), lonlat=True, nest=True
        )
        hlong = np.mod(hlong, 360)
        xlong = xr.DataArray(hlong, dims="z")
        xlat = xr.DataArray(hlat, dims="z")

        def interpolate(variable):
            xhp = variable.interp(
                latitude=xlat, longitude=xlong, kwargs={"fill_value": None}
            )
            hp_image = np.array(xhp.to_array().to_numpy(), dtype=np.float32)
            hp_image = (hp_image - hp_image.mean(axis=1, keepdims=True)) / hp_image.std(
                axis=1, keepdims=True
            )
            return hp_image

        hp_surface = interpolate(e5xr.surface)
        hp_upper = interpolate(e5xr.upper)
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

    def __getitem__(self, idx):
        e5sc = cdstest.ERA5SampleConfig(
            year="1999", month="01", day="01", time="00:00:00"
        )
        e5s = cdstest.get_era5_sample(e5sc)
        e5_target_config = cdstest.ERA5SampleConfig(
            year="1999", month="01", day="01", time="03:00:00"
        )
        e5target = cdstest.get_era5_sample(e5_target_config)
        hp_surface, hp_upper = self.e5_to_numpy(e5s)
        hp_target_surface, hp_target_upper = self.e5_to_numpy(e5target)

        return dict(
            input_surface=hp_surface,
            input_upper=hp_upper,
            target_surface=hp_target_surface,
            target_upper=hp_target_upper,
            sample_id=idx,
        )
        # return create_sample_legacy(hp_surface[0:4], hp_target_surface[0:4], 0)

    # def create_metric_sample(
    #     self,
    #     output: Dict[str, torch.Tensor],
    #     batch: Dict[str, torch.Tensor],
    #     train_epoch_state: TrainEpochState,
    # ):
    #     return create_metric_sample_legacy(output, batch, train_epoch_state)

    def create_metric_sample(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        train_epoch_state: TrainEpochState,
    ):
        return MetricSample(
            output=output["logits_surface"].detach(),
            prediction=None,
            target=batch["target_surface"].detach(),
            sample_id=batch["sample_id"].detach(),
            epoch=train_epoch_state.epoch,
            batch=train_epoch_state.batch,
        )

    def __len__(self):
        return 50
