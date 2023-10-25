#!/usr/bin/env python
import torch
import numpy as np
from pathlib import Path

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_regression_metrics

from lib.models.healpix.swin_hp_transformer import SwinHPTransformerConfig
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.files import prepare_results
from lib.serialization import serialize_human

from lib.data_factory import register_dataset, get_factory
from dataclasses import dataclass
from lib.dataspec import DataSpec
import healpix
import xarray as xr
import experiments.weather.cdstest as cdstest


@dataclass
class DataHPConfig:
    nside: int = 64

    def serialize_human(self):
        return serialize_human(self.__dict__)  # dict(validation=self.validation)


class DataHP(torch.utils.data.Dataset):
    def __init__(self, data_config: DataHPConfig):
        self.config = data_config

    @staticmethod
    def data_spec(config: DataHPConfig):
        return DataSpec(
            input_shape=torch.Size([4, healpix.nside2npix(config.nside)]),
            output_shape=torch.Size([4, healpix.nside2npix(config.nside)]),
            target_shape=torch.Size([4, healpix.nside2npix(config.nside)]),
        )

    def e5_to_numpy(self, e5xr):
        npix = healpix.nside2npix(self.config.nside)
        hlong, hlat = healpix.pix2ang(
            128, np.arange(0, npix, 1), lonlat=True, nest=True
        )
        hlong = np.mod(hlong, 360)
        xlong = xr.DataArray(hlong, dims="z")
        xlat = xr.DataArray(hlat, dims="z")
        xhp = e5xr.surface.interp(latitude=xlat, longitude=xlong)
        hp_image = np.array(xhp.to_array().to_numpy(), dtype=np.float32)
        hp_image = hp_image / hp_image.max()
        return hp_image

    def __getitem__(self, idx):
        e5sc = cdstest.ERA5SampleConfig(
            year="1999", month="01", day="01", time="00:00:00"
        )
        e5s = cdstest.get_era5_sample(e5sc)
        e5_target_config = cdstest.ERA5SampleConfig(
            year="1999", month="01", day="01", time="03:00:00"
        )
        e5target = cdstest.get_era5_sample(e5_target_config)

        hp_image = self.e5_to_numpy(e5s)
        hp_target = self.e5_to_numpy(e5target)

        # breakpoint()
        # hp_image = np.zeros(self.data_spec(self.config).input_shape, dtype=np.float32)
        # hp_target = np.zeros(self.data_spec(self.config).input_shape, dtype=np.float32)
        return hp_image, hp_target, 0

    def __len__(self):
        return 50


def create_config(ensemble_id):
    loss = torch.nn.HuberLoss()

    def ce_loss(outputs, targets):
        # breakpoint()
        return loss(outputs["logits"], targets)

    train_config = TrainConfig(
        # model_config=MLPClassConfig(widths=[50, 50]),
        model_config=SwinHPTransformerConfig(base_pix=12, nside=64),
        train_data_config=DataHPConfig(nside=64),
        val_data_config=DataHPConfig(nside=64),
        loss=ce_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam,
            # kwargs=dict(weight_decay=1e-4, lr=0.001, momentum=0.9),
            kwargs=dict(weight_decay=1e-4, lr=0.001),
            # kwargs=dict(weight_decay=0.0, lr=0.001),
        ),
        batch_size=2,
        ensemble_id=ensemble_id,
    )
    train_eval = create_regression_metrics(None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=3,
        save_nth_epoch=1,
        validate_nth_epoch=5,
    )
    return train_run


if __name__ == "__main__":
    device_id = ddp_setup()

    get_factory()
    register_dataset(DataHPConfig, DataHP)
    ensemble_config = create_ensemble_config(create_config, 1)
    ensemble = create_ensemble(ensemble_config, device_id)

    result_path = prepare_results(
        Path(__file__).parent, Path(__file__).stem, ensemble_config
    )
