#!/usr/bin/env python
import ssl
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
from sqlalchemy import create_engine
from lib.render_psql import get_url

from lib.data_registry import (
    DataCIFARConfig,
    DataCIFAR10CConfig,
    DataCIFAR10C,
    DataSubsetConfig,
    DataJoinConfig,
)

import lib.data_factory as data_factory
import lib.slurm as slurm
from lib.stable_hash import json_dumps_dataclass

from lib.models.mlp import MLPClassConfig

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import train_member
from lib.train import evaluate_metrics_on_data
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metric_dict

from experiments.looking_at_the_posterior.config import (
    create_config_function,
    # create_corrupted_dataset_config,
)

# from experiments.looking_at_the_posterior.uq import uq_for_ensemble
from lib.uncertainty import uncertainty

# from lib.render_psql import dict_to_normalized_json

ssl._create_default_https_context = ssl._create_unverified_context
rng = np.random.default_rng(42)

if __name__ == "__main__":
    device_id = ddp_setup()

    ensemble_config_mlp = create_ensemble_config(
        create_config_function(
            model_config=MLPClassConfig(widths=[128] * 2), batch_size=2**13
        ),
        n_members=5,
    )
    if slurm.get_task_id() is not None:
        train_member(ensemble_config_mlp, slurm.get_task_id(), device_id)
    else:
        print("Not in an array job so creating whole ensemble...")
        ensemble_mlp = create_ensemble(ensemble_config_mlp, device_id)

    if slurm.get_task_id() is not None:
        print("Exiting early since this is a SLURM array job used for training only")
        exit(0)

    ds_cifar_val = data_factory.get_factory().create(DataCIFARConfig(validation=True))
    dl_cifar_val = torch.utils.data.DataLoader(
        ds_cifar_val,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    all_subsets = DataCIFAR10C.cifarc_subsets[:]
    frost_config = DataCIFAR10CConfig(
        subsets=all_subsets[::2], severities=[1, 2, 3, 4, 5]
    )
    ds_cifar_c_frost = data_factory.get_factory().create(frost_config)
    dl_cifar_c_frost = torch.utils.data.DataLoader(
        ds_cifar_c_frost,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )
    pixelate_config = DataCIFAR10CConfig(
        subsets=all_subsets[1::2], severities=[1, 2, 3, 4, 5]
    )
    ds_cifar_c_pixelate = data_factory.get_factory().create(pixelate_config)
    dl_cifar_c_pixelate = torch.utils.data.DataLoader(
        ds_cifar_c_pixelate,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    uq_frost = uncertainty(dl_cifar_c_frost, ensemble_mlp, device_id)
    acc = (
        torch.where(
            uq_frost.targets[:, None].cpu() == uq_frost.mean_pred[:, None].cpu(),
            1.0,
            0.0,
        )
        .numpy()
        .squeeze()
    )
    log_H = torch.log(uq_frost.H)
    log_MI = torch.log(uq_frost.MI)
    mean_acc_frost, x_bins, y_bins, bin_number = binned_statistic_2d(
        log_H.cpu().numpy(),
        log_MI.cpu().numpy(),
        acc,
        statistic="mean",
        bins=50,
        expand_binnumbers=True,
    )

    uq_pixelate = uncertainty(dl_cifar_c_pixelate, ensemble_mlp, device_id)
    acc = (
        torch.where(
            uq_pixelate.targets[:, None].cpu() == uq_pixelate.mean_pred[:, None].cpu(),
            1.0,
            0.0,
        )
        .numpy()
        .squeeze()
    )
    log_H = torch.log(uq_pixelate.H)
    log_MI = torch.log(uq_pixelate.MI)
    mean_acc_pixelate, x_bins, y_bins, bin_number_pixelate = binned_statistic_2d(
        log_H.cpu().numpy(),
        log_MI.cpu().numpy(),
        acc,
        statistic="mean",
        bins=[x_bins, y_bins],
        expand_binnumbers=True,
    )

    bin_coords = list(zip(bin_number_pixelate[0] - 1, bin_number_pixelate[1] - 1))
    bin_coords_and_sample_ids = list(zip(bin_coords, uq_pixelate.sample_ids))
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

    inferred_accs = mean_acc_frost[just_coords_x, just_coords_y]
    accs_and_coords_and_ids = list(
        zip(inferred_accs, just_coords_x, just_coords_y, just_sample_ids)
    )

    # accs_and_coords_and_ids_no_nan = filter(
    # lambda x: not np.isnan(x[0]), accs_and_coords_and_ids
    # )
    def nans_last_in_sort(x):
        new_acc = x[0]
        if np.isnan(new_acc):
            new_acc = 2.0
        return new_acc, x[1], x[2], x[3]

    accs_and_coords_and_ids_no_nan = list(
        map(nans_last_in_sort, accs_and_coords_and_ids)
    )

    sorted_acc = sorted(accs_and_coords_and_ids_no_nan, key=lambda x: x[0])

    sorted_ids = [int(sample_idx) for _, _, _, (sample_idx, _, _) in sorted_acc]
    random_ids = rng.permutation(len(ds_cifar_c_pixelate))
    n_samples = len(sorted_ids)

    # al_subset_config = DataSubsetConfig(
    #     data_config=pixelate_config, subset=sorted_ids[:200]
    # )
    # metrics = create_classification_metric_dict(10)
    # evaluate_metrics_on_data(
    #     ensemble_mlp.members[0],
    #     metrics,
    #     al_subset_config,
    #     2000,
    #     ComputeConfig(distributed=False, num_workers=8),
    #     device_id,
    # )
    # breakpoint()

    def train_on_fraction(f):
        n_al_samples = int(n_samples * f)
        # n_ten_percent = int(len(sorted_ids) * 0.1)

        al_sample_ids = sorted_ids[:n_al_samples]
        al_random_ids = random_ids[:n_al_samples]
        overlap = set(al_sample_ids).intersection(set(al_random_ids))
        print(f"Aquisition overlap: {len(overlap) / len(al_sample_ids)}")

        al_subset_config = DataSubsetConfig(
            data_config=pixelate_config, subset=al_sample_ids
        )
        random_subset_config = DataSubsetConfig(
            data_config=pixelate_config, subset=al_random_ids
        )
        # )
        # breakpoint()
        al_extended_ds_config = DataJoinConfig(
            data_configs=[DataCIFARConfig(), al_subset_config]
        )
        random_extended_ds_config = DataJoinConfig(
            data_configs=[DataCIFARConfig(), random_subset_config]
        )

        al_ensemble_config = create_ensemble_config(
            create_config_function(
                model_config=MLPClassConfig(widths=[128] * 2),
                batch_size=2**13,
                data_config=al_extended_ds_config,
                # num_workers=2,
            ),
            n_members=1,
        )
        al_ensemble = create_ensemble(al_ensemble_config, device_id)

        random_ensemble_config = create_ensemble_config(
            create_config_function(
                model_config=MLPClassConfig(widths=[128] * 2),
                batch_size=2**13,
                data_config=random_extended_ds_config,
                # num_workers=2,
            ),
            n_members=1,
        )
        with open("json_ser_check.json", "a") as sf:
            sf.write(f"frac: {frac}\nal ensemble\n")
            sf.write(json_dumps_dataclass(al_ensemble_config).decode("utf-8"))
            sf.write("\nrandom ensemble\n")
            sf.write(json_dumps_dataclass(random_ensemble_config).decode("utf-8"))
        print(f"Random ensemble frac {frac}:")
        random_ensemble = create_ensemble(random_ensemble_config, device_id)
        print("Random done")
        return al_ensemble, random_ensemble

    # small_acc = list(filter(lambda x: x[0] < 0.5, accs_and_coords))
    df_rows = []
    # Evenly spaced log in interval [10^-4, 10^0]
    for frac in np.logspace(start=-4, stop=0, num=15, base=10):
        # for frac in np.linspace(0.1, 1.0, 10):
        al_ensemble, random_ensemble = train_on_fraction(frac)

        for ensemble_name, ensemble in [("al", al_ensemble), ("rnd", random_ensemble)]:
            metrics = create_classification_metric_dict(10)
            ensemble.members[0].eval()
            evaluate_metrics_on_data(
                ensemble.members[0],
                metrics,
                pixelate_config,
                2000,
                ComputeConfig(distributed=False, num_workers=8),
                device_id,
            )
            for metric_name, metric in metrics.items():
                df_rows.append(
                    dict(
                        model=ensemble_name,
                        fraction=frac,
                        metric=metric_name,
                        value=metric.mean(),
                    )
                )

    df = pd.DataFrame(columns=["model", "fraction", "metric", "value"], data=df_rows)
    df.to_csv(Path(__file__).parent / "active_learning.csv")
    psql = create_engine(get_url())
    df.to_sql("active_learning", psql, if_exists="replace")

    # mean_acc_frost[]

    import matplotlib.pyplot as plt
    from pathlib import Path

    # fig, ax = plt.subplots(1, 1)
    # x = [cx for _, cx, cy in small_acc]
    # y = [cy for _, cx, cy in small_acc]
    # ax.scatter(x, y)
    # fig.savefig(Path(__file__).parent / "small_acc_scatter.pdf")

    fig, ax = plt.subplots(1, 1)
    ax.scatter(mean_acc_frost.flatten(), mean_acc_pixelate.flatten())
    fig.savefig(Path(__file__).parent / "acc_scatter.pdf")

    # torch.histogramdd(uq.)
    # uq_for_ensemble(dl_cifar_c, ensemble_mlp, ensemble_config_mlp, "mlp", device_id)
    # uq_for_ensemble(dl_cifar_val, ensemble_mlp, ensemble_config_mlp, "mlp", device_id)
