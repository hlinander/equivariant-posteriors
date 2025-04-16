import xarray as xr
from dataclasses import dataclass, field
from typing import List
import numpy as np
import tqdm

import torch
from experiments.weather.data import (
    denormalize_sample,
    deserialize_dataset_statistics,
    e5_to_numpy_hp,
    DataHP,
    DataHPConfig,
)
from experiments.weather.cdsmontly import ERA5Sample
from lib.render_psql import add_artifact


def dh_numpy_to_xr_surface_hp(data_surface, data_upper, meta) -> ERA5Sample:
    V, Nlat, Nlon = data_surface.shape
    # lat = np.linspace(-np.pi / 2, np.pi / 2, Nlat)
    # lon = np.linspace(0, 2 * np.pi, Nlon)
    lat = np.linspace(90, -90, Nlat)
    lon = np.linspace(0, 360, Nlon)
    pressure_levels = [
        1000.0,
        925.0,
        850.0,
        700.0,
        600.0,
        500.0,
        400.0,
        300.0,
        250.0,
        200.0,
        150.0,
        100.0,
        50.0,
    ]
    # variables = ["v1", "v2", "v3", "v4"]
    data_vars = {
        name: (["latitude", "longitude"], data_surface[idx])
        for idx, name in enumerate(meta["surface"]["names"])
    }
    surface = xr.Dataset(
        data_vars=data_vars,
        coords={"latitude": lat, "longitude": lon},
    )

    data_vars = {
        name: (["pl", "latitude", "longitude"], data_upper[idx])
        for idx, name in enumerate(meta["upper"]["names"])
    }
    upper = xr.Dataset(
        data_vars=data_vars,
        coords={"pl": pressure_levels, "latitude": lat, "longitude": lon},
    )
    return ERA5Sample(surface=surface, upper=upper)


@dataclass
class Category:
    names: List[str]
    long_names: List[str]
    units: List[str]
    levels: List[float]
    level_units: str
    level_name: str


@dataclass
class MeteorologicalData:
    surface: Category = field(
        default_factory=lambda: Category(
            names=["msl", "u10", "v10", "t2m"],
            long_names=[
                "Mean sea level pressure",
                "10 metre U wind component",
                "10 metre V wind component",
                "2 metre temperature",
            ],
            units=["Pa", "m s**-1", "m s**-1", "K"],
            levels=[],
            level_units="hPa",
            level_name="Geopotential height",
        )
    )
    upper: Category = field(
        default_factory=lambda: Category(
            names=["z", "q", "t", "u", "v"],
            long_names=[
                "Geopotential",
                "Specific humidity",
                "Temperature",
                "U component of wind",
                "V component of wind",
            ],
            units=["m**2 s**-2", "kg kg**-1", "K", "m s**-1", "m s**-1"],
            levels=[
                1000.0,
                925.0,
                850.0,
                700.0,
                600.0,
                500.0,
                400.0,
                300.0,
                250.0,
                200.0,
                150.0,
                100.0,
                50.0,
            ],
            level_units="hPa",
            level_name="Geopotential height",
        )
    )


# Example of creating an instance of MeteorologicalData
# era5_meta = MeteorologicalData()


@dataclass
class ACC:
    acc_surface: np.ndarray
    acc_upper: np.ndarray
    acc_unnorm_surface: np.ndarray
    acc_unnorm_upper: np.ndarray


@dataclass
class RMSE:
    surface: np.ndarray
    upper: np.ndarray
    mean_surface: np.ndarray
    mean_upper: np.ndarray


def anomaly_correlation_coefficient_hp(model, dataloader, device_id):
    # surface: B, variable, x
    # upper: B, variable, height, x

    initialized = False
    logit_surface_squared = None
    target_surface_squared = None
    logit_upper_squared = None
    target_upper_squared = None
    nominator_surface = None
    nominator_upper = None

    dims = [0, -1]
    stats = deserialize_dataset_statistics(dataloader.dataset.config.nside).item()
    stats = {key: torch.tensor(value).to(device_id) for key, value in stats.items()}
    # traced_model = None
    model = model.eval()

    # def model_forward(surface, upper):
    # return model._forward((surface, upper))
    # template = dataloader.dataset.ds.get_template_e5s()
    # meta = dataloader.dataset.ds.get_meta()
    # breakpoint()

    for idx, batch in enumerate(dataloader):
        batch = {k: v.to(device_id) for k, v in batch.items()}
        # if traced_model is None:
        # traced_model = torch.jit.trace(
        # model_forward, (batch["input_surface"], batch["input_upper"])
        # )
        # traced_model = torch.jit.freeze(traced_model)
        # print("Model forward...")
        with torch.no_grad():
            output = model(batch)
        output = {k: v.detach() for k, v in output.items()}
        # print("Denorming...")
        out_surface, out_upper = denormalize_sample(
            stats, output["logits_surface"].double(), output["logits_upper"].double()
        )
        target_surface, target_upper = denormalize_sample(
            stats, batch["target_surface"].double(), batch["target_upper"].double()
        )
        climate_surface, climate_upper = denormalize_sample(
            stats,
            batch["climate_target_surface"].double(),
            batch["climate_target_upper"].double(),
        )
        # print("ACC..")
        out_surface = out_surface - climate_surface
        out_upper = out_upper - climate_upper
        target_surface = target_surface - climate_surface
        target_upper = target_upper - climate_upper
        if not initialized:
            initialized = True
            logit_surface_squared = out_surface**2  # .sum(dim=dims)
            target_surface_squared = target_surface**2  # .sum(dim=dims)
            logit_upper_squared = out_upper**2  # .sum(dim=dims)
            target_upper_squared = target_upper**2  # .sum(dim=dims)

            # breakpoint()
            nominator_surface = out_surface * target_surface  # .sum(dim=dims)
            nominator_upper = out_upper * target_upper  # .sum(dim=dims)
        else:
            # print("surface")
            logit_surface_squared += out_surface**2  # .sum(dim=dims)
            target_surface_squared += target_surface**2  # .sum(dim=dims)
            # print("upper")
            logit_upper_squared += out_upper**2  # .sum(dim=dims)
            target_upper_squared += target_upper**2  # .sum(dim=dims)

            # print("nominator")
            # breakpoint()
            nominator_surface += out_surface * target_surface  # .sum(dim=dims)
            nominator_upper += out_upper * target_upper  # .sum(dim=dims)
            # print("nominator done")
        # if idx > 2:
        # break

    denominator_surface = torch.sqrt(
        logit_surface_squared.sum(dim=dims) * target_surface_squared.sum(dim=dims)
    )
    denominator_upper = torch.sqrt(
        logit_upper_squared.sum(dim=dims) * target_upper_squared.sum(dim=dims)
    )

    acc_surface = nominator_surface.sum(dim=dims) / denominator_surface
    acc_upper = nominator_upper.sum(dim=dims) / denominator_upper

    return ACC(
        acc_surface=acc_surface,
        acc_upper=acc_upper,
        acc_unnorm_surface=nominator_surface,
        acc_unnorm_upper=nominator_upper,
    )


def rmse_hp(model, dataloader, device_id):
    # surface: B, variable, x
    # upper: B, variable, height, x

    print("[eval] RMSE")
    initialized = False
    rmse_surface = None
    rmse_upper = None

    # dims = [0, -1]
    model.eval()
    n_batches = 0
    stats = deserialize_dataset_statistics(dataloader.dataset.config.nside).item()
    stats = {key: torch.tensor(value).to(device_id) for key, value in stats.items()}
    for idx, batch in tqdm.tqdm(enumerate(dataloader)):
        # batch = {k: v.to(device_id) for k, v in batch.items()}
        batch = {
            k: v.to(device_id) if hasattr(v, "to") else v for k, v in batch.items()
        }
        with torch.no_grad():
            output = model(batch)
        output = {k: v.detach() for k, v in output.items()}
        out_surface, out_upper = denormalize_sample(
            stats, output["logits_surface"].double(), output["logits_upper"].double()
        )
        target_surface, target_upper = denormalize_sample(
            stats, batch["target_surface"].double(), batch["target_upper"].double()
        )
        n_pixels = batch["target_surface"].shape[-1]
        n_samples = batch["target_surface"].shape[0]
        if not initialized:
            initialized = True
            rmse_surface_batches = torch.sqrt(
                ((out_surface - target_surface) ** 2)
            )  # .sum(dim=-1)) / n_pixels
            rmse_upper_batches = torch.sqrt(
                ((out_upper - target_upper) ** 2)
            )  # .sum(dim=-1)) / n_pixels
            rmse_surface = rmse_surface_batches.sum(dim=0) / n_samples
            rmse_upper = rmse_upper_batches.sum(dim=0) / n_samples
        else:
            rmse_surface_batches = torch.sqrt(
                ((out_surface - target_surface) ** 2)
            )  # .sum(dim=-1)) / n_pixels
            rmse_upper_batches = torch.sqrt(
                ((out_upper - target_upper) ** 2)
            )  # .sum(dim=-1)) / n_pixels
            rmse_surface += rmse_surface_batches.sum(dim=0) / n_samples
            rmse_upper += rmse_upper_batches.sum(dim=0) / n_samples
        n_batches += 1
        # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if idx > 1:
        # break

    rmse_surface /= n_batches
    rmse_upper /= n_batches
    mean_rmse_surface = rmse_surface.sum(dim=-1) / n_pixels
    mean_rmse_upper = rmse_upper.sum(dim=-1) / n_pixels
    return RMSE(
        surface=rmse_surface,
        upper=rmse_upper,
        mean_surface=mean_rmse_surface,
        mean_upper=mean_rmse_upper,
    )
    # return dict(surface=rmse_surface, upper=rmse_upper)


def rmse_dh(model, dataloader_dh, device_id):
    # surface: B, variable, x
    # upper: B, variable, height, x

    print("[eval] RMSE Driscoll-Healy on HP grid")
    initialized = False
    rmse_surface = None
    rmse_upper = None

    # dims = [0, -1]
    model.eval()
    n_batches = 0
    ds_config = dataloader_dh.dataset.config.validation()
    ds_config.driscoll_healy = False
    ds_hp = DataHP(ds_config)
    dl_hp = torch.utils.data.DataLoader(
        ds_hp,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    stats = deserialize_dataset_statistics(dl_hp.dataset.config.nside).item()
    stats = {k: torch.tensor(v).to(device_id) for k, v in stats.items()}
    # stats = {key: torch.tensor(value).to(device_id) for key, value in stats.items()}
    for idx, (batch_dh, batch_hp) in tqdm.tqdm(enumerate(zip(dataloader_dh, dl_hp))):
        # batch = {k: v.to(device_id) for k, v in batch_dh.items()}
        batch = {
            k: v.to(device_id) if hasattr(v, "to") else v for k, v in batch_dh.items()
        }
        batch_hp = {
            k: v.to(device_id) if hasattr(v, "to") else v for k, v in batch_hp.items()
        }
        with torch.no_grad():
            output = model(batch)
        output = {k: v.detach() for k, v in output.items() if hasattr(v, "detach")}
        e5s = dh_numpy_to_xr_surface_hp(
            output["logits_surface"][0].detach().cpu().numpy(),
            output["logits_upper"][0].detach().cpu().numpy(),
            dl_hp.dataset.get_meta(),
        )
        # np.save(
        #     "/tmp/surface_test.npy",
        #     e5s.surface.to_array().to_numpy().astype(np.float32),
        # )
        # add_artifact(train_run, "surface_test_rmse.npydh", "/tmp/surface_test.npy")
        # breakpoint()
        surface, upper = e5_to_numpy_hp(e5s, dl_hp.dataset.config.nside, False)
        # np.save(
        #     "/tmp/surface_test_hp.npy",
        #     surface.astype(np.float32),
        # )
        # add_artifact(train_run, "surface_test_rmse_hp.npy", "/tmp/surface_test_hp.npy")
        surface = torch.from_numpy(surface).to(device_id)
        upper = torch.from_numpy(upper).to(device_id)

        out_surface, out_upper = denormalize_sample(
            stats,
            surface.double()[None, ...],
            upper.double()[None, ...],
        )
        target_surface, target_upper = denormalize_sample(
            stats,
            batch_hp["target_surface"].double(),  # astype(np.double),
            batch_hp["target_upper"].double(),  # astype(np.double),
        )
        n_pixels = batch_hp["target_surface"].shape[-1]
        n_samples = batch_hp["target_surface"].shape[0]
        if not initialized:
            initialized = True
            rmse_surface_batches = torch.sqrt(
                ((out_surface - target_surface) ** 2)
            )  # .sum(dim=-1)) / n_pixels
            rmse_upper_batches = torch.sqrt(
                ((out_upper - target_upper) ** 2)
            )  # .sum(dim=-1)) / n_pixels
            rmse_surface = rmse_surface_batches.sum(dim=0) / n_samples
            rmse_upper = rmse_upper_batches.sum(dim=0) / n_samples
        else:
            rmse_surface_batches = torch.sqrt(
                ((out_surface - target_surface) ** 2)
            )  # .sum(dim=-1)) / n_pixels
            rmse_upper_batches = torch.sqrt(
                ((out_upper - target_upper) ** 2)
            )  # .sum(dim=-1)) / n_pixels
            rmse_surface += rmse_surface_batches.sum(dim=0) / n_samples
            rmse_upper += rmse_upper_batches.sum(dim=0) / n_samples
        n_batches += 1
        # breakpoint()
        # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if idx > 1:
        # break

    rmse_surface /= n_batches
    rmse_upper /= n_batches
    mean_rmse_surface = rmse_surface.sum(dim=-1) / n_pixels
    mean_rmse_upper = rmse_upper.sum(dim=-1) / n_pixels
    return RMSE(
        surface=rmse_surface,
        upper=rmse_upper,
        mean_surface=mean_rmse_surface,
        mean_upper=mean_rmse_upper,
    )
    # return dict(surface=rmse_surface, upper=rmse_upper)
