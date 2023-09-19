#!/usr/bin/env python
import ssl
import json
from pathlib import Path
from dataclasses import dataclass

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
from lib.stable_hash import json_dumps_dataclass_str

from lib.models.mlp import MLPClassConfig

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import train_member
from lib.ensemble import EnsembleConfig
from lib.train import evaluate_metrics_on_data
from lib.train_dataclasses import ComputeConfig
from lib.stable_hash import stable_hash_small

from lib.classification_metrics import create_classification_metric_dict

from experiments.looking_at_the_posterior.config import (
    create_config_function,
    # create_corrupted_dataset_config,
)

# from experiments.looking_at_the_posterior.uq import uq_for_ensemble
from lib.uncertainty import uncertainty, uq_to_dataframe

# from lib.render_psql import dict_to_normalized_json

ssl._create_default_https_context = ssl._create_unverified_context
rng = np.random.default_rng(42)
rng_initial_data = np.random.default_rng(42)


@dataclass
class ALConfig:
    ensemble_config: EnsembleConfig
    uq_calibration_data_config: object
    data_validation_config: object
    data_pool_config: object
    n_start: int = 100
    n_end: int = 3000
    n_steps: int = 20

    def serialize_human(self):
        return {
            "ensemble_config": self.ensemble_config.serialize_human(),
            "uq_calibration_data_config": self.ensemble_config.serialize_human(),
            "data_validation_config": self.ensemble_config.serialize_human(),
            "data_pool_config": self.ensemble_config.serialize_human(),
            "n_start": self.n_start,
            "n_end": self.n_end,
            "n_steps": self.n_steps,
        }


def al(al_config: ALConfig, device):
    ensemble = create_ensemble(al_config.ensemble_config, device)

    hash = stable_hash_small(al_config)
    output_path = Path(__file__).parent / f"al_{hash}"
    output_path.mkdir(exist_ok=True, parents=True)
    open(output_path / "al_config.json", "w").write(
        json_dumps_dataclass_str(al_config, indent=2)
    )
    ds_uq_calibration = data_factory.get_factory().create(
        al_config.uq_calibration_data_config
    )
    dl_uq_calibration = torch.utils.data.DataLoader(
        ds_uq_calibration,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )
    # pixelate_config = DataCIFAR10CConfig(
    # subsets=all_subsets[1::2], severities=[1, 2, 3, 4, 5]
    # )
    ds_pool = data_factory.get_factory().create(al_config.data_pool_config)
    dl_pool = torch.utils.data.DataLoader(
        ds_pool,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    uq_calibration = uncertainty(dl_uq_calibration, ensemble, device_id)
    uq_to_dataframe(uq_calibration).to_csv(output_path / "uq_calibration.csv")
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

    uq_pool = uncertainty(dl_pool, ensemble, device_id)
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
    random_ids = rng.permutation(len(ds_pool)).tolist()
    n_samples = len(sorted_ids)

    def train_on_fraction(f):
        n_al_samples = int(n_samples * f)

        al_sample_ids = sorted_ids[:n_al_samples]
        al_random_ids = random_ids[:n_al_samples]
        overlap = set(al_sample_ids).intersection(set(al_random_ids))
        print(f"Aquisition overlap: {len(overlap) / len(al_sample_ids)}")

        al_subset_config = DataSubsetConfig(
            data_config=al_config.data_pool_config,
            subset=al_sample_ids,
            minimum_epoch_length=10000,
        )
        random_subset_config = DataSubsetConfig(
            data_config=al_config.data_pool_config,
            subset=al_random_ids,
            minimum_epoch_length=10000,
        )
        al_extended_ds_config = DataJoinConfig(
            data_configs=[
                al_config.ensemble_config.members[0].train_config.train_data_config,
                al_subset_config,
            ]
        )
        random_extended_ds_config = DataJoinConfig(
            data_configs=[
                al_config.ensemble_config.members[0].train_config.train_data_config,
                random_subset_config,
            ]
        )
        open(
            output_path / f"frac_{f}_{len(al_sample_ids)}_al_dataset_classes.json", "w"
        ).write(json.dumps([ds_pool[idx][1] for idx in al_sample_ids]))
        open(
            output_path / f"frac_{f}_{len(al_sample_ids)}_random_dataset_classes.json",
            "w",
        ).write(json.dumps([ds_pool[idx][1] for idx in al_random_ids]))
        open(output_path / f"frac_{f}_{len(al_sample_ids)}_al_dataset.json", "w").write(
            json_dumps_dataclass_str(al_subset_config, indent=2)
        )
        open(
            output_path / f"frac_{f}_{len(al_random_ids)}_rnd_dataset.json", "w"
        ).write(json_dumps_dataclass_str(random_subset_config, indent=2))

        al_ensemble_config = create_ensemble_config(
            create_config_function(
                model_config=MLPClassConfig(widths=[128] * 2),
                batch_size=2**13,
                data_config=al_extended_ds_config,
                epochs=50,
            ),
            n_members=1,
        )
        al_ensemble = create_ensemble(al_ensemble_config, device_id)

        random_ensemble_config = create_ensemble_config(
            create_config_function(
                model_config=MLPClassConfig(widths=[128] * 2),
                batch_size=2**13,
                data_config=random_extended_ds_config,
                epochs=50,
            ),
            n_members=1,
        )
        # with open("n_ser_check.json", "a") as sf:
        #     sf.write(f"frac: {frac}\nal ensemble\n")
        #     sf.write(json_dumps_dataclass(al_ensemble_config).decode("utf-8"))
        #     sf.write("\nrandom ensemble\n")
        #     sf.write(json_dumps_dataclass(random_ensemble_config).decode("utf-8"))
        print(f"Random ensemble frac {frac}:")
        random_ensemble = create_ensemble(random_ensemble_config, device_id)
        print("Random done")
        return al_ensemble, random_ensemble

    # small_acc = list(filter(lambda x: x[0] < 0.5, accs_and_coords))
    df_rows = []
    # Evenly spaced log in interval [10^-4, 10^0]
    n_pool_samples = len(ds_pool)
    # n_pool_samples * 10^fracstart = 10^2 => frac_start = log10(10^2/n_pool_samples)
    frac_start = np.log10(al_config.n_start / n_pool_samples)
    frac_end = np.log10(al_config.n_end / n_pool_samples)
    for frac in np.logspace(
        start=frac_start, stop=frac_end, num=al_config.n_steps, base=10
    ):
        # for frac in np.linspace(0.1, 1.0, 10):
        al_ensemble, random_ensemble = train_on_fraction(frac)

        for ensemble_name, ensemble in [("al", al_ensemble), ("rnd", random_ensemble)]:
            metrics = create_classification_metric_dict(10)
            ensemble.members[0].eval()
            evaluate_metrics_on_data(
                ensemble.members[0],
                metrics,
                al_config.data_validation_config,
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
    df.to_csv(output_path / "active_learning.csv")


if __name__ == "__main__":
    device_id = ddp_setup()

    ds_cifar_train = data_factory.get_factory().create(
        DataCIFARConfig(validation=False)
    )
    class_and_id = [(x[1], x[2]) for x in ds_cifar_train]
    initial_ids = []
    for class_id in range(5):
        class_ids = [x for x in class_and_id if x[0] == class_id]
        # Pick two samples per class randomly
        initial_ids = initial_ids + [
            class_ids[idx] for idx in rng_initial_data.permutation(len(class_ids))[:2]
        ]
    # Pick out the sample_idx
    initial_ids = [int(sample[1]) for sample in initial_ids]

    initial_train_data_config = DataSubsetConfig(
        data_config=DataCIFARConfig(), subset=initial_ids, minimum_epoch_length=10000
    )
    ensemble_config_mlp = create_ensemble_config(
        create_config_function(
            model_config=MLPClassConfig(widths=[128] * 2),
            batch_size=2**13,
            data_config=initial_train_data_config,
        ),
        n_members=10,
    )
    if slurm.get_task_id() is not None:
        train_member(ensemble_config_mlp, slurm.get_task_id(), device_id)
    else:
        print("Not in an array job so creating whole ensemble...")
        ensemble_mlp = create_ensemble(ensemble_config_mlp, device_id)

    if slurm.get_task_id() is not None:
        print("Exiting early since this is a SLURM array job used for training only")
        exit(0)

    all_subsets = DataCIFAR10C.cifarc_subsets[:]
    uq_calibration_data_config = DataCIFAR10CConfig(
        subsets=all_subsets[::2], severities=[1, 2, 3, 4, 5]
    )
    pool_ids = []
    # for class_id in range(5):
    #     class_ids = [x for x in class_and_id if x[0] == class_id]
    #     # Pick two samples per class randomly
    #     pool_ids = pool_ids + [
    #         class_ids[idx] for idx in rng_initial_data.permutation(len(class_ids))[:300]
    #     ]
    for class_id in range(5, 10):
        class_ids = [x for x in class_and_id if x[0] == class_id]
        # Pick two samples per class randomly
        pool_ids = pool_ids + [
            class_ids[idx] for idx in rng_initial_data.permutation(len(class_ids))[:10]
        ]
    # class_ids = [x for x in class_and_id if x[0] == 5]
    # Pick two samples per class randomly
    # pool_ids = pool_ids + [
    # class_ids[idx] for idx in rng_initial_data.permutation(len(class_ids))[:10]
    # ]
    # Pick out the sample_idx
    pool_ids = [int(sample[1]) for sample in pool_ids]
    pool_ids = pool_ids + list(initial_ids * 200)

    data_pool_config = DataSubsetConfig(data_config=DataCIFARConfig(), subset=pool_ids)
    al(
        ALConfig(
            ensemble_config=ensemble_config_mlp,
            uq_calibration_data_config=uq_calibration_data_config,
            data_validation_config=DataCIFARConfig(validation=True),
            data_pool_config=data_pool_config,
        ),
        device_id,
    )

    # psql = create_engine(get_url())
    # df.to_sql("active_learning", psql, if_exists="replace")

    # mean_acc_frost[]

    # import matplotlib.pyplot as plt
    # from pathlib import Path

    # fig, ax = plt.subplots(1, 1)
    # x = [cx for _, cx, cy in small_acc]
    # y = [cy for _, cx, cy in small_acc]
    # ax.scatter(x, y)
    # fig.savefig(Path(__file__).parent / "small_acc_scatter.pdf")

    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(mean_acc_frost.flatten(), mean_acc_pixelate.flatten())
    # fig.savefig(Path(__file__).parent / "acc_scatter.pdf")

    # torch.histogramdd(uq.)
    # uq_for_ensemble(dl_cifar_c, ensemble_mlp, ensemble_config_mlp, "mlp", device_id)
    # uq_for_ensemble(dl_cifar_val, ensemble_mlp, ensemble_config_mlp, "mlp", device_id)
