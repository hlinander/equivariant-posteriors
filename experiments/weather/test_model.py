#!/usr/bin/env python
import torch
import numpy as np
from pathlib import Path
import tqdm

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.regression_metrics import create_regression_metrics

from lib.models.healpix.swin_hp_transformer import SwinHPTransformerConfig
from lib.models.mlp import MLPConfig
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import symlink_checkpoint_files
from lib.files import prepare_results
from lib.serialization import serialize_human

from lib.data_factory import register_dataset, get_factory
from dataclasses import dataclass
from lib.dataspec import DataSpec
import healpix
import xarray as xr
import experiments.weather.cdstest as cdstest

NSIDE = 64


@dataclass
class DataHPConfig:
    nside: int = NSIDE
    version: int = 10

    def serialize_human(self):
        return serialize_human(self.__dict__)


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
            self.config.nside, np.arange(0, npix, 1), lonlat=True, nest=True
        )
        hlong = np.mod(hlong, 360)
        xlong = xr.DataArray(hlong, dims="z")
        xlat = xr.DataArray(hlat, dims="z")
        xhp = e5xr.surface.interp(latitude=xlat, longitude=xlong)
        hp_image = np.array(xhp.to_array().to_numpy(), dtype=np.float32)
        hp_image = (hp_image - hp_image.mean(axis=1, keepdims=True)) / hp_image.std(
            axis=1, keepdims=True
        )
        # max_vals = np.amax(hp_image, axis=1, keepdims=True)
        # hp_image = hp_image / max_vals
        # breakpoint()
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

        return hp_image[0:4], hp_target[0:4], 0

    def __len__(self):
        return 50


def create_config(ensemble_id):
    loss = torch.nn.L1Loss()

    def reg_loss(outputs, targets):
        return loss(outputs["logits"], targets)

    train_config = TrainConfig(
        model_config=SwinHPTransformerConfig(
            base_pix=12,
            nside=NSIDE,
            dev_mode=False,
            depths=[4, 8, 12],
            num_heads=[3, 6, 12],
            embed_dim=48,
            window_size=64,
            use_cos_attn=True,
            use_v2_norm_placement=True,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            rel_pos_bias="flat",
            shift_size=32,
            shift_strategy="nest_roll",
            ape=True,
        ),
        # model_config=MLPConfig(widths=[256, 256]),
        train_data_config=DataHPConfig(nside=NSIDE),
        val_data_config=DataHPConfig(nside=NSIDE),
        loss=reg_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=1e-5, lr=0.01),
        ),
        batch_size=2,
        ensemble_id=ensemble_id,
        _version=2,
    )
    train_eval = create_regression_metrics(torch.nn.functional.l1_loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=30,
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

    ds = get_factory().create(DataHPConfig(nside=NSIDE))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    result_path = prepare_results(
        Path(__file__).parent,
        f"{Path(__file__).stem}_{ensemble_config.members[0].train_config.model_config.__class__.__name__}",
        ensemble_config,
    )
    symlink_checkpoint_files(ensemble, result_path)

    for xs, ys, ids in tqdm.tqdm(dl):
        xs = xs.to(device_id)

        output = ensemble.members[0](xs)
        np.save(result_path / "of_surface.npy", output["logits"].detach().cpu().numpy())
        np.save(result_path / "if_surface.npy", xs.detach().cpu().numpy())
        np.save(result_path / "tf_surface.npy", ys.detach().cpu().numpy())
        break
