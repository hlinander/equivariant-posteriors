#!/usr/bin/env python
import torch
import numpy as np
from pathlib import Path
import tqdm
import onnxruntime as ort

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.regression_metrics import create_regression_metrics

from lib.models.healpix.swin_hp_transformer import SwinHPTransformerConfig
from lib.models.healpix.swin_hp_pangu import SwinHPPanguConfig
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
EPOCHS = 20


def numpy_to_xds(np_array, xds_template):
    transformed_ds = xr.Dataset()
    for i, var_name in enumerate(xds_template.data_vars):
        transformed_ds[var_name] = xr.DataArray(
            np_array[i], dims=xds_template.dims, coords=xds_template.coords
        )
    return transformed_ds


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

        return hp_surface[0:4], hp_target_surface[0:4], 0

    def __len__(self):
        return 50


def create_config(ensemble_id):
    loss = torch.nn.L1Loss()

    def reg_loss(outputs, targets):
        return loss(outputs["logits"], targets)

    train_config = TrainConfig(
        model_config=SwinHPPanguConfig(
            base_pix=12,
            nside=NSIDE,
            dev_mode=False,
            depths=[2, 6, 6, 2],
            # num_heads=[6, 12, 12, 6],
            num_heads=[8, 16, 16, 8],
            # embed_dims=[192 // 16, 384 // 16, 384 // 16, 192 // 16],
            # embed_dims=[16, 384 // 16, 384 // 16, 192 // 16],
            embed_dims=[x * 2 for x in [16, 32, 32, 16]],
            window_size=16,  # int(32 * (NSIDE / 256)),
            use_cos_attn=False,
            use_v2_norm_placement=True,
            drop_rate=0,  # ,0.1,
            attn_drop_rate=0,  # ,0.1,
            drop_path_rate=0,
            rel_pos_bias="earth",
            # shift_size=8,  # int(16 * (NSIDE / 256)),
            shift_size=8,  # int(16 * (NSIDE / 256)),
            shift_strategy="nest_roll",
            ape=False,
            patch_size=int(16 * (NSIDE / 256)),
        ),
        # model_config=SwinHPTransformerConfig(
        #     base_pix=12,
        #     nside=NSIDE,
        #     dev_mode=False,
        #     depths=[4, 8, 12],
        #     num_heads=[3, 6, 12],
        #     embed_dim=48,
        #     window_size=64,
        #     use_cos_attn=True,
        #     use_v2_norm_placement=True,
        #     drop_rate=0.1,
        #     attn_drop_rate=0.1,
        #     rel_pos_bias="flat",
        #     shift_size=32,
        #     shift_strategy="nest_roll",
        #     ape=True,
        #     patch_size=16,
        # ),
        # model_config=MLPConfig(widths=[256, 256]),
        train_data_config=DataHPConfig(nside=NSIDE),
        val_data_config=DataHPConfig(nside=NSIDE),
        loss=reg_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
        ),
        batch_size=1,
        ensemble_id=ensemble_id,
        _version=25,
    )
    train_eval = create_regression_metrics(torch.nn.functional.l1_loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=EPOCHS,
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
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    result_path = prepare_results(
        Path(__file__).parent,
        f"{Path(__file__).stem}_{ensemble_config.members[0].train_config.model_config.__class__.__name__}_nside_{NSIDE}_epochs_{EPOCHS}",
        ensemble_config,
    )
    symlink_checkpoint_files(ensemble, result_path)

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 16

    cuda_provider_options = {
        "arena_extend_strategy": "kSameAsRequested",
    }

    ort_session_3 = ort.InferenceSession(
        "experiments/weather/pangu_models/pangu_weather_3.onnx",
        sess_options=options,
        providers=[("CUDAExecutionProvider", cuda_provider_options)],
    )

    for xs, ys, ids in tqdm.tqdm(dl):
        xs = xs.to(device_id)

        output = ensemble.members[0](xs)
        np.save(result_path / "of_surface.npy", output["logits"].detach().cpu().numpy())
        np.save(result_path / "if_surface.npy", xs.detach().cpu().numpy())
        np.save(result_path / "tf_surface.npy", ys.detach().cpu().numpy())

        # dh, dh_target = ds.get_driscoll_healy(ids[0])
        # te5s = ds.get_template_e5s()
        # pangu_output_upper, pangu_output_surface = ort_session_3.run(
        #     None,
        #     dict(
        #         input=dh.upper.to_array().to_numpy(),
        #         input_surface=dh.surface.to_array().to_numpy(),
        #     ),
        # )
        # pangu_surface_xds = numpy_to_xds(pangu_output_surface, te5s.surface)
        # pangu_upper_xds = numpy_to_xds(pangu_output_upper, te5s.upper)
        # pangu_np_surface, pangu_np_upper = ds.e5_to_numpy(
        #     cdstest.ERA5Sample(surface=pangu_surface_xds, upper=pangu_upper_xds)
        # )
        # np.save(result_path / "pangu_pred_surface.npy", pangu_np_surface)
        # np.save(result_path / "pangu_pred_upper.npy", pangu_np_upper)
        np.save(result_path / "pangu_pred_surface.npy", xs.detach().cpu().numpy()[0])
        np.save(result_path / "pangu_pred_upper.npy", xs.detach().cpu().numpy()[0])
        break
