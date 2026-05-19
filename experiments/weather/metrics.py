import xarray as xr
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import tqdm
from healpy.rotator import Rotator
import healpy as hp
import matplotlib.pyplot as plt
from experiments.weather.models.hp_pear_conv import HEALPixPearConv, HEALPixPearConvConfig

import torch
from experiments.weather.data import (
    denormalize_sample,
    deserialize_dataset_statistics,
    e5_to_numpy_hp,
    DataHP,
    Climatology,
    DataHPConfig,
)
from experiments.weather.cdsmontly import ERA5Sample


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

@dataclass
class EquivarianceError:
    surface: Dict[float, List[float]]
    upper: Dict[float, List[float]]
    surface_std: Dict[float, List[float]]
    upper_std: Dict[float, List[float]]
    n_measurements: int


import copy
import numpy as np
import healpy as hp
import torch
import tqdm


def rotate_longitude_map_np(m, angle_deg, nested=True):
    """
    Rotate a scalar HEALPix map around the polar axis by angle_deg.
    Input:
        m: numpy array of shape [npix]
    Output:
        rotated numpy array of shape [npix]
    """
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)

    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix, nest=nested)

    angle_rad = np.deg2rad(angle_deg)

    # backward sampling: output(theta, phi) = input(theta, phi - angle)
    phi_src = (phi - angle_rad) % (2 * np.pi)

    return hp.get_interp_val(m, theta, phi_src, nest=nested)


def rotate_tensor_last_dim_healpix(x, angle_deg, nested=True):
    """
    Rotate a tensor along its last dimension, assuming the last dimension is a HEALPix map.
    Returns a tensor on the same device/dtype as x.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    original_device = x.device
    original_dtype = x.dtype
    shape = x.shape
    npix = shape[-1]

    x_flat = x.reshape(-1, npix)
    x_np = x_flat.detach().cpu().numpy()

    rotated_np = np.empty_like(x_np)
    for i in range(x_np.shape[0]):
        rotated_np[i] = rotate_longitude_map_np(x_np[i], angle_deg, nested=nested)

    rotated = torch.from_numpy(rotated_np).to(device=original_device, dtype=original_dtype)
    return rotated.reshape(shape)


@dataclass
class PrecomputedRotation:
    """Interpolation neighbours and weights for a fixed HEALPix longitude rotation."""
    pixels: torch.Tensor   # [4, npix] int64
    weights: torch.Tensor  # [4, npix] float32


def precompute_rotation(nside, angle_deg, nested=True, device="cpu"):
    """
    Precompute bilinear interpolation indices and weights for rotating a HEALPix
    map by angle_deg around the polar axis.  Call once per (nside, angle) and
    reuse across all batches/channels.
    """
    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix, nest=nested)
    phi_src = (phi - np.deg2rad(angle_deg)) % (2 * np.pi)
    pixels, weights = hp.get_interp_weights(nside, theta, phi_src, nest=nested)
    # pixels: [4, npix] int64, weights: [4, npix] float64
    pixels_t = torch.from_numpy(pixels.astype(np.int64)).to(device)
    weights_t = torch.from_numpy(weights.astype(np.float32)).to(device)
    return PrecomputedRotation(pixels=pixels_t, weights=weights_t)


def apply_rotation_precomputed(x, rot: "PrecomputedRotation"):
    """
    Apply a precomputed HEALPix rotation to tensor x (last dim = npix).
    Fully vectorised — no Python loop over channels.
    x: [*, npix] tensor on any device/dtype.
    rot.pixels / rot.weights must be on the same device as x.
    """
    shape = x.shape
    npix = shape[-1]
    x_flat = x.reshape(-1, npix)                          # [N, npix]
    # x_flat[:, rot.pixels]: [N, 4, npix], weights: [4, npix]
    rotated = (x_flat[:, rot.pixels] * rot.weights).sum(dim=1)  # [N, npix]
    return rotated.to(dtype=x.dtype).reshape(shape)


def shift_sample_precomputed(sample, rot: "PrecomputedRotation"):
    """Like shift_sample but uses a PrecomputedRotation — no per-call healpy overhead."""
    if "input_surface" in sample and "input_upper" in sample:
        surface_key, upper_key = "input_surface", "input_upper"
    elif "logits_surface" in sample and "logits_upper" in sample:
        surface_key, upper_key = "logits_surface", "logits_upper"
    else:
        raise ValueError(
            "Sample must contain either "
            "'input_surface'/'input_upper' or "
            "'logits_surface'/'logits_upper'"
        )
    rotated_sample = dict(sample)
    rotated_sample[surface_key] = apply_rotation_precomputed(sample[surface_key], rot)
    rotated_sample[upper_key] = apply_rotation_precomputed(sample[upper_key], rot)
    return rotated_sample


def shift_sample(sample, angle_deg, nested=True):
    """
    Rotate either an input batch or an output batch by angle_deg.

    Supported dictionaries:
      input:
        'input_surface', 'input_upper'
      output:
        'logits_surface', 'logits_upper'

    Expected shapes:
      input_surface / logits_surface: [B, C, Npix] or [C, Npix]
      input_upper   / logits_upper:   [B, C, L, Npix] or [C, L, Npix]

    Returns:
      shallow-copied dict with rotated relevant fields.
    """
    if "input_surface" in sample and "input_upper" in sample:
        surface_key = "input_surface"
        upper_key = "input_upper"
    elif "logits_surface" in sample and "logits_upper" in sample:
        surface_key = "logits_surface"
        upper_key = "logits_upper"
    else:
        raise ValueError(
            "Sample must contain either "
            "'input_surface'/'input_upper' or "
            "'logits_surface'/'logits_upper'"
        )

    rotated_sample = dict(sample)

    rotated_sample[surface_key] = rotate_tensor_last_dim_healpix(
        sample[surface_key], angle_deg, nested=nested
    )
    rotated_sample[upper_key] = rotate_tensor_last_dim_healpix(
        sample[upper_key], angle_deg, nested=nested
    )

    return rotated_sample


def equivariance_error(model, dataloader, device, sensitivity=4, nested=True, max_batches=None, relative=False, optimised=False):
    """
    Measure approximate rotational equivariance of a model under longitude rotations.

    For each batch:
      1. compute y = f(x)
      2. rotate input: x_r = R_a x
      3. compute y_r = f(x_r)
      4. unrotate prediction: R_-a y_r
      5. compare y with R_-a y_r

    If relative=True, reports ||f(Rα x) − Rα f(x)||_rms / ||f(x)||_rms per channel.
    This is scale-invariant and the correct metric for untrained / randomly-initialised
    models where the absolute output magnitude is meaningless.

    Interpolation indices are precomputed once per angle before the batch loop,
    eliminating the per-batch/per-channel healpy overhead.

    Returned structure:
      - surface[angle] -> tensor of shape [C]
      - upper[angle]   -> tensor of shape [C, L]
    """
    _EPS = 1e-8
    model = model.to(device)
    model.eval()

    angles = [i * (360.0 / sensitivity) for i in range(1, sensitivity)]

    surface_batches = {angle: [] for angle in angles}
    upper_batches = {angle: [] for angle in angles}

    # Precomputed rotations; populated lazily on first batch so we know nside.
    rot_fwd: dict = {}   # angle -> PrecomputedRotation
    rot_bwd: dict = {}   # angle -> PrecomputedRotation (inverse)

    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

        # Build rotation lookup once from the first batch's pixel count.
        if not rot_fwd:
            npix = batch["input_surface"].shape[-1]
            nside = hp.npix2nside(npix)
            for angle in angles:
                rot_fwd[angle] = precompute_rotation(nside, angle, nested=nested, device=device)
                rot_bwd[angle] = precompute_rotation(nside, -angle, nested=nested, device=device)

        with torch.no_grad():
            output = model(batch)

        ref_surface = output["logits_surface"]   # [B, C, Npix]
        ref_upper = output["logits_upper"]       # [B, C, L, Npix]

        for angle_deg in angles:
            rotated_input = shift_sample_precomputed(batch, rot_fwd[angle_deg])

            with torch.no_grad():
                rotated_output = model(rotated_input)


            if optimised:
                unrotated = shift_sample_precomputed(rotated_output, rot_bwd[angle_deg])
                ref_s  = ref_surface
                ref_u  = ref_upper
                pred_s = unrotated["logits_surface"]
                pred_u = unrotated["logits_upper"]
            else:
                rotated_ref = shift_sample_precomputed(
                    {"logits_surface": ref_surface, "logits_upper": ref_upper},
                    rot_fwd[angle_deg],
                )
                ref_s  = rotated_ref["logits_surface"]
                ref_u  = rotated_ref["logits_upper"]
                pred_s = rotated_output["logits_surface"]
                pred_u = rotated_output["logits_upper"]

            if relative:
                # Relative equivariance error, bounded in [0, 1]:
                #   REE = ||f(Rx) - Rf(x)||_rms / sqrt(||f(Rx)||_rms² + ||f(x)||_rms²)
                # Denominator is the RMS of the two outputs combined, so REE ∈ [0, 1].
                # 0 = perfectly equivariant; ~1 = completely uncorrelated outputs.
                # Normalising by both norms (not just ||f(x)||) avoids values > 1 that
                # arise when the two outputs are uncorrelated random vectors.
                diff_s_rms = (ref_s - pred_s).pow(2).mean(dim=-1).sqrt()   # [B, C]
                denom_s = (ref_s.pow(2).mean(dim=-1) + pred_s.pow(2).mean(dim=-1)).sqrt()  # [B, C]
                err_surface = diff_s_rms / (denom_s + _EPS)                            # [B, C]

                diff_u_rms = (ref_u - pred_u).pow(2).mean(dim=-1).sqrt()       # [B, C, L]
                denom_u = (ref_u.pow(2).mean(dim=-1) + pred_u.pow(2).mean(dim=-1)).sqrt()      # [B, C, L]
                err_upper = diff_u_rms / (denom_u + _EPS)                              # [B, C, L]

                surface_batches[angle_deg].append(err_surface.mean(dim=0).detach().cpu())   # [C]
                upper_batches[angle_deg].append(err_upper.mean(dim=0).detach().cpu())       # [C, L]
            else:
                err_surface = torch.sqrt((ref_s - pred_s) ** 2)   # [B, C, Npix]
                err_upper = torch.sqrt((ref_u - pred_u) ** 2)     # [B, C, L, Npix]

                surface_batches[angle_deg].append(err_surface.mean(dim=(0, 2)).detach().cpu())  # [C]
                upper_batches[angle_deg].append(err_upper.mean(dim=(0, 3)).detach().cpu())      # [C, L]

    mean_surface = {angle: torch.stack(surface_batches[angle]).mean(dim=0) for angle in angles}
    std_surface  = {angle: torch.stack(surface_batches[angle]).std(dim=0)  for angle in angles}
    mean_upper   = {angle: torch.stack(upper_batches[angle]).mean(dim=0)   for angle in angles}
    std_upper    = {angle: torch.stack(upper_batches[angle]).std(dim=0)    for angle in angles}

    return EquivarianceError(
        surface=mean_surface,
        upper=mean_upper,
        surface_std=std_surface,
        upper_std=std_upper,
        n_measurements={angle: len(surface_batches[angle]) for angle in angles},
    )
            
    


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
        batch = {
            k: v.to(device_id) if hasattr(v, "to") else v for k, v in batch.items()
        }
        # if traced_model is None:
        # traced_model = torch.jit.trace(
        # model_forward, (batch["input_surface"], batch["input_upper"])
        # )
        # traced_model = torch.jit.freeze(traced_model)
        # print("Model forward...")
        for _ in range(dataloader.dataset.config.lead_time_days):
            with torch.no_grad():
                output = model(batch)
            batch["input_surface"] = output["logits_surface"]
            batch["input_upper"] = output["logits_upper"]

        output = {k: v.detach() if hasattr(v, "to") else v for k, v in output.items()}
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


def anomaly_correlation_coefficient_dh(model, dataloader_dh, device_id):
    # surface: B, variable, x
    # upper: B, variable, height, x
    ds_config = dataloader_dh.dataset.config.validation()
    ds_config.driscoll_healy = False
    ds_hp = Climatology(ds_config)
    # ds_hp = DataHP(ds_config)
    dl_hp = torch.utils.data.DataLoader(
        ds_hp,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    stats = deserialize_dataset_statistics(dl_hp.dataset.config.nside).item()
    stats = {key: torch.tensor(value).to(device_id) for key, value in stats.items()}

    initialized = False
    logit_surface_squared = None
    target_surface_squared = None
    logit_upper_squared = None
    target_upper_squared = None
    nominator_surface = None
    nominator_upper = None

    dims = [0, -1]
    # stats = deserialize_dataset_statistics(dataloader.dataset.config.nside).item()
    # traced_model = None
    model = model.eval()

    # def model_forward(surface, upper):
    # return model._forward((surface, upper))
    # template = dataloader.dataset.ds.get_template_e5s()
    # meta = dataloader.dataset.ds.get_meta()
    # breakpoint()

    for idx, (batch_dh, batch_hp) in tqdm.tqdm(enumerate(zip(dataloader_dh, dl_hp))):
        batch = {
            k: v.to(device_id) if hasattr(v, "to") else v for k, v in batch_dh.items()
        }
        batch_hp = {
            k: v.to(device_id) if hasattr(v, "to") else v for k, v in batch_hp.items()
        }
        # if traced_model is None:
        # traced_model = torch.jit.trace(
        # model_forward, (batch["input_surface"], batch["input_upper"])
        # )
        # traced_model = torch.jit.freeze(traced_model)
        # print("Model forward...")
        # with torch.no_grad():
        # output = model(batch)
        for _ in range(ds_config.lead_time_days):
            with torch.no_grad():
                output = model(batch)
            batch["input_surface"] = output["logits_surface"]
            batch["input_upper"] = output["logits_upper"]

        output = {k: v.detach() if hasattr(v, "to") else v for k, v in output.items()}
        e5s = dh_numpy_to_xr_surface_hp(
            output["logits_surface"][0].detach().cpu().numpy(),
            output["logits_upper"][0].detach().cpu().numpy(),
            dl_hp.dataset.get_meta(),
        )
        surface, upper = e5_to_numpy_hp(e5s, dl_hp.dataset.config.nside, False)
        surface = torch.from_numpy(surface).to(device_id)
        upper = torch.from_numpy(upper).to(device_id)
        # print("Denorming...")
        # out_surface, out_upper = denormalize_sample(
        #     stats, output["logits_surface"].double(), output["logits_upper"].double()
        # )
        # breakpoint()
        out_surface, out_upper = denormalize_sample(
            stats, surface.double()[None, ...], upper.double()[None, ...]
        )
        target_surface, target_upper = denormalize_sample(
            stats,
            batch_hp["target_surface"].double(),
            batch_hp["target_upper"].double(),
        )
        climate_surface, climate_upper = denormalize_sample(
            stats,
            batch_hp["climate_target_surface"].double(),
            batch_hp["climate_target_upper"].double(),
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


def calculate_latitude_weight_from_surface_tensor(out_surface):
    weights = torch.zeros(out_surface.shape[-2:], dtype=out_surface.dtype)
    cos_vals = torch.zeros((out_surface.shape[-2],), dtype=out_surface.dtype)
    for lat_idx in range(out_surface.shape[-2]):
        lat = (
            np.pi
            / 2.0
            * abs(out_surface.shape[-2] // 2 - lat_idx)
            / (out_surface.shape[-2] // 2)
        )
        cos_vals[lat_idx] = np.cos(lat)
    cos_vals = out_surface.shape[-2] * cos_vals / cos_vals.sum()

    # print(cos_vals)
    weights[:] = cos_vals[:, None]

    weights = weights.to(out_surface.device)
    return weights


def anomaly_correlation_coefficient_dh_on_dh(model, dataloader_dh, device_id):
    # surface: B, variable, x
    # upper: B, variable, height, x
    ds_config = dataloader_dh.dataset.config.validation()
    # ds_config.driscoll_healy = False
    # ds_hp = Climatology(ds_config)
    # ds_hp = DataHP(ds_config)
    # dl_hp = torch.utils.data.DataLoader(
    # ds_hp,
    # batch_size=1,
    # shuffle=False,
    # drop_last=False,
    # )
    stats = deserialize_dataset_statistics(dataloader_dh.dataset.config.nside).item()
    stats = {key: torch.tensor(value).to(device_id) for key, value in stats.items()}

    initialized = False
    logit_surface_squared = None
    target_surface_squared = None
    logit_upper_squared = None
    target_upper_squared = None
    nominator_surface = None
    nominator_upper = None

    dims = [0, -1, -2]
    # stats = deserialize_dataset_statistics(dataloader.dataset.config.nside).item()
    # traced_model = None
    model = model.eval()

    # def model_forward(surface, upper):
    # return model._forward((surface, upper))
    # template = dataloader.dataset.ds.get_template_e5s()
    # meta = dataloader.dataset.ds.get_meta()
    # breakpoint()
    weights = None

    for idx, batch_dh in tqdm.tqdm(enumerate(dataloader_dh)):
        batch = {
            k: v.to(device_id) if hasattr(v, "to") else v for k, v in batch_dh.items()
        }
        # if traced_model is None:
        # traced_model = torch.jit.trace(
        # model_forward, (batch["input_surface"], batch["input_upper"])
        # )
        # traced_model = torch.jit.freeze(traced_model)
        # print("Model forward...")
        # with torch.no_grad():
        # output = model(batch)
        for _ in range(ds_config.lead_time_days):
            with torch.no_grad():
                output = model(batch)
            batch["input_surface"] = output["logits_surface"]
            batch["input_upper"] = output["logits_upper"]

        output = {k: v.detach() if hasattr(v, "to") else v for k, v in output.items()}

        out_surface, out_upper = denormalize_sample(
            stats,
            output["logits_surface"].double(),  # [None, ...],
            output["logits_upper"].double(),  # [None, ...],
        )
        target_surface, target_upper = denormalize_sample(
            stats,
            batch["target_surface"].double(),
            batch["target_upper"].double(),
        )
        climate_surface, climate_upper = denormalize_sample(
            stats,
            batch["climate_target_surface"].double(),
            batch["climate_target_upper"].double(),
        )
        # print("ACC..")
        if weights is None:
            weights = calculate_latitude_weight_from_surface_tensor(out_surface)
            # breakpoint()
            # cos_sum += np.cos(lat)

        # out_surface.shape = torch.Size([1, 4, 157, 314])
        # out_upper.shape = torch.Size([1, 5, 13, 157, 314])
        out_surface = out_surface - climate_surface
        out_upper = out_upper - climate_upper
        target_surface = target_surface - climate_surface
        target_upper = target_upper - climate_upper
        if not initialized:
            initialized = True
            # breakpoint()
            logit_surface_squared = weights * out_surface**2  # .sum(dim=dims)
            # breakpoint()
            target_surface_squared = weights * target_surface**2  # .sum(dim=dims)
            logit_upper_squared = weights * out_upper**2  # .sum(dim=dims)
            target_upper_squared = weights * target_upper**2  # .sum(dim=dims)

            # breakpoint()
            nominator_surface = weights * out_surface * target_surface  # .sum(dim=dims)
            nominator_upper = weights * out_upper * target_upper  # .sum(dim=dims)
        else:
            # print("surface")
            logit_surface_squared += weights * out_surface**2  # .sum(dim=dims)
            target_surface_squared += weights * target_surface**2  # .sum(dim=dims)
            # print("upper")
            logit_upper_squared += weights * out_upper**2  # .sum(dim=dims)
            target_upper_squared += weights * target_upper**2  # .sum(dim=dims)

            # print("nominator")
            # breakpoint()
            nominator_surface += (
                weights * out_surface * target_surface
            )  # .sum(dim=dims)
            nominator_upper += weights * out_upper * target_upper  # .sum(dim=dims)
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

    print("Lead time days:", dataloader.dataset.config.lead_time_days)

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
        for _ in range(dataloader.dataset.config.lead_time_days):
            with torch.no_grad():
                output = model(batch)
            batch["input_surface"] = output["logits_surface"]
            batch["input_upper"] = output["logits_upper"]
        # with torch.no_grad():
        #     output = model(batch)
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
        for _ in range(ds_config.lead_time_days):
            with torch.no_grad():
                output = model(batch)
            batch["input_surface"] = output["logits_surface"]
            batch["input_upper"] = output["logits_upper"]

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
        # breakpoint()
        surface, upper = e5_to_numpy_hp(e5s, dl_hp.dataset.config.nside, False)
        # np.save(
        #     "/tmp/surface_test_hp.npy",
        #     surface.astype(np.float32),
        # )
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


def rmse_dh_on_dh(model, dataloader_dh, device_id, weighted=True):
    # surface: B, variable, x
    # upper: B, variable, height, x

    print("[eval] RMSE Driscoll-Healy on DH grid")
    initialized = False
    rmse_surface = None
    rmse_upper = None

    # dims = [0, -1]
    model.eval()
    n_batches = 0
    ds_config = dataloader_dh.dataset.config.validation()

    stats = deserialize_dataset_statistics(dataloader_dh.dataset.config.nside).item()
    stats = {k: torch.tensor(v).to(device_id) for k, v in stats.items()}

    weights = None
    # stats = {key: torch.tensor(value).to(device_id) for key, value in stats.items()}
    for idx, batch_dh in tqdm.tqdm(enumerate(dataloader_dh)):
        # batch = {k: v.to(device_id) for k, v in batch_dh.items()}
        batch = {
            k: v.to(device_id) if hasattr(v, "to") else v for k, v in batch_dh.items()
        }
        for _ in range(ds_config.lead_time_days):
            with torch.no_grad():
                output = model(batch)
            batch["input_surface"] = output["logits_surface"]
            batch["input_upper"] = output["logits_upper"]

        output = {k: v.detach() for k, v in output.items() if hasattr(v, "detach")}

        out_surface, out_upper = denormalize_sample(
            stats,
            output["logits_surface"].double(),
            output["logits_upper"].double(),
        )
        target_surface, target_upper = denormalize_sample(
            stats,
            batch["target_surface"].double(),  # astype(np.double),
            batch["target_upper"].double(),  # astype(np.double),
        )
        n_pixels = batch["target_surface"].shape[-1] * batch["target_surface"].shape[-2]
        n_samples = batch["target_surface"].shape[0]
        if weights is None:
            if weighted:
                weights = calculate_latitude_weight_from_surface_tensor(out_surface)
            else:
                weights = 1.0

        if not initialized:
            initialized = True
            rmse_surface_batches = weights * torch.sqrt(
                ((out_surface - target_surface) ** 2)
            )  # .sum(dim=-1)) / n_pixels
            rmse_upper_batches = weights * torch.sqrt(
                ((out_upper - target_upper) ** 2)
            )  # .sum(dim=-1)) / n_pixels
            rmse_surface = rmse_surface_batches.sum(dim=0) / n_samples
            rmse_upper = rmse_upper_batches.sum(dim=0) / n_samples
        else:
            rmse_surface_batches = weights * torch.sqrt(
                ((out_surface - target_surface) ** 2)
            )  # .sum(dim=-1)) / n_pixels
            rmse_upper_batches = weights * torch.sqrt(
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
    mean_rmse_surface = rmse_surface.sum(dim=[-1, -2]) / n_pixels
    mean_rmse_upper = rmse_upper.sum(dim=[-1, -2]) / n_pixels
    return RMSE(
        surface=rmse_surface,
        upper=rmse_upper,
        mean_surface=mean_rmse_surface,
        mean_upper=mean_rmse_upper,
    )
    # return dict(surface=rmse_surface, upper=rmse_upper)


if __name__ == "__main__":
    from experiments.weather.data import DataHPConfig, DataHP

    config = DataHPConfig(
        nside=64,
        start_year=2019,
        end_year=2019,
        delta_t=2
    )
    ds = DataHP(config)

    model = HEALPixPearConv(HEALPixPearConvConfig())

    print(equivariance_error(model, torch.utils.data.DataLoader(ds, batch_size=4), "cuda", sensitivity=4, max_batches=10))

    # sample = ds[23]
    # shifted_sample = shift_sample(sample, 180)

    # # Plotting the samples to verify the shift
    # channel_idx = 3  # choose the surface channel to inspect

    # original_map = sample["input_surface"][channel_idx]
    # shifted_map = shifted_sample["input_surface"][channel_idx]

    # diff = shifted_map - original_map

    # print("max abs diff:", np.max(np.abs(diff)))
    # print("mean abs diff:", np.mean(np.abs(diff)))
    # print("relative L2 diff:", np.linalg.norm(diff) / np.linalg.norm(original_map))

    # plt.figure(figsize=(10, 12))

    # hp.cartview(
    #     original_map,
    #     fig=1,
    #     sub=(2, 1, 1),
    #     flip="geo",
    #     lonra=[-180, 180],
    #     latra=[-90, 90],
    #     title=f"Original surface channel {channel_idx}",
    #     nest=True
    # )

    # hp.cartview(
    #     shifted_map,
    #     fig=1,
    #     sub=(2, 1, 2),
    #     flip="geo",
    #     lonra=[-180, 180],
    #     latra=[-90, 90],
    #     title=f"Shifted surface channel {channel_idx} (+90°)",
    #     nest=True
    # )   


    # plt.savefig("shifted_samples.png")
