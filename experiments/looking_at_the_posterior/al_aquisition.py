import torch
from pathlib import Path
import numpy as np
from scipy.stats import binned_statistic_2d

import lib.data_factory as data_factory
from lib.uncertainty import uncertainty, uq_to_dataframe

from experiments.looking_at_the_posterior.al_config import ALStep


def al_aquisition_random(
    # al_config: ALConfig,
    # ensemble: object,
    # rng: np.random.Generator,
    al_step: ALStep,
    output_path: Path,
    device_id,
):
    # ds_pool = data_factory.get_factory().create(al_step.al_config.data_pool_config)
    random_idxs = al_step.rng.permutation(len(al_step.pool_ids)).tolist()
    n_samples = int(
        (al_step.al_config.n_end - al_step.al_config.n_start)
        / al_step.al_config.n_steps
    )

    # breakpoint()
    new_ids = [al_step.pool_ids[idx] for idx in random_idxs[:n_samples]]
    return al_step.aquired_ids + new_ids


def al_aquisition_calibrated_uncertainty(
    # al_step.al_config: ALConfig,
    # ensemble: object,
    # rng: np.random.Generator,
    al_step: ALStep,
    output_path: Path,
    device_id,
):
    ds_uq_calibration = data_factory.get_factory().create(
        al_step.al_config.uq_calibration_data_config
    )
    dl_uq_calibration = torch.utils.data.DataLoader(
        ds_uq_calibration,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )
    ds_pool = data_factory.get_factory().create(al_step.al_config.data_pool_config)
    dl_pool = torch.utils.data.DataLoader(
        ds_pool,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    uq_calibration = uncertainty(dl_uq_calibration, al_step.ensemble, device_id)
    uq_to_dataframe(uq_calibration).to_csv(
        output_path
        / f"uq_calibration_step_{al_step.step:03d}_{al_step.al_config.aquisition_method}.csv"
    )
    acc = (
        torch.where(
            uq_calibration.targets[:, None].cpu()
            == uq_calibration.mean_pred[:, None].cpu(),
            1.0,
            0.0,
        )
        .numpy()
        .squeeze()
    )
    log_H = torch.log(uq_calibration.H)
    log_MI = torch.log(uq_calibration.MI)
    mean_acc_calibration, x_bins, y_bins, bin_number = binned_statistic_2d(
        log_H.cpu().numpy(),
        log_MI.cpu().numpy(),
        acc,
        statistic="mean",
        bins=50,
        expand_binnumbers=True,
    )

    uq_pool = uncertainty(dl_pool, al_step.ensemble, device_id)
    acc = (
        torch.where(
            uq_pool.targets[:, None].cpu() == uq_pool.mean_pred[:, None].cpu(),
            1.0,
            0.0,
        )
        .numpy()
        .squeeze()
    )
    log_H = torch.log(uq_pool.H)
    log_MI = torch.log(uq_pool.MI)
    mean_acc_pool, x_bins, y_bins, bin_number_pool = binned_statistic_2d(
        log_H.cpu().numpy(),
        log_MI.cpu().numpy(),
        acc,
        statistic="mean",
        bins=[x_bins, y_bins],
        expand_binnumbers=True,
    )

    bin_coords = list(zip(bin_number_pool[0] - 1, bin_number_pool[1] - 1))
    bin_coords_and_sample_ids = list(zip(bin_coords, uq_pool.sample_ids))
    bin_coords_and_sample_ids_with_values = [
        (coord, sample_id)
        for (coord, sample_id) in bin_coords_and_sample_ids
        if (
            coord[0] >= 0
            and coord[0] < len(x_bins) - 1
            and coord[1] >= 0
            and coord[1] < len(y_bins) - 1
        )
    ]

    just_coords_x = [coord[0] for coord, _ in bin_coords_and_sample_ids_with_values]
    just_coords_y = [coord[1] for coord, _ in bin_coords_and_sample_ids_with_values]
    just_sample_ids = [
        sample_id for _, sample_id in bin_coords_and_sample_ids_with_values
    ]

    inferred_accs = mean_acc_calibration[just_coords_x, just_coords_y]
    accs_and_coords_and_ids = list(
        zip(inferred_accs, just_coords_x, just_coords_y, just_sample_ids)
    )

    def nans_last_in_sort(x):
        new_acc = x[0]
        if np.isnan(new_acc):
            new_acc = 2.0
        return new_acc, x[1], x[2], x[3]

    accs_and_coords_and_ids_no_nan = list(
        map(nans_last_in_sort, accs_and_coords_and_ids)
    )

    sorted_acc = sorted(accs_and_coords_and_ids_no_nan, key=lambda x: x[0])

    sorted_ids = [int(sample_idx) for _, _, _, sample_idx in sorted_acc]

    # Restrict to available pool ids
    sorted_ids = [
        sample_idx for sample_idx in sorted_ids if sample_idx in al_step.pool_ids
    ]

    # breakpoint()

    new_ids = sorted_ids[
        : int(
            (al_step.al_config.n_end - al_step.al_config.n_start)
            / al_step.al_config.n_steps
        )
    ]
    return al_step.aquired_ids + new_ids