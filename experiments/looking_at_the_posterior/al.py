#!/usr/bin/env python
import ssl
import json
from pathlib import Path

# from dataclasses import dataclass
# from typing import Literal

# import torch
import numpy as np

import pandas as pd

# from sqlalchemy import create_engine
# from lib.render_psql import get_url

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
from lib.ensemble import ensemble_mean_prediction
from lib.ensemble import is_ensemble_serialized
from lib.train import evaluate_metrics_on_data
from lib.train_dataclasses import ComputeConfig
from lib.stable_hash import stable_hash_small

from lib.classification_metrics import create_classification_metric_dict

from experiments.looking_at_the_posterior.config import (
    create_config_function,
    # create_corrupted_dataset_config,
)

from experiments.looking_at_the_posterior.al_config import ALStep, ALConfig
from experiments.looking_at_the_posterior.al_aquisition import (
    al_aquisition_calibrated_uncertainty,
    al_aquisition_random,
)

# from experiments.looking_at_the_posterior.uq import uq_for_ensemble

# from lib.render_psql import dict_to_normalized_json

ssl._create_default_https_context = ssl._create_unverified_context
# rng = np.random.default_rng(42)
rng_initial_data = np.random.default_rng(42)


AQUISITION_FUNCTIONS = dict(
    calibrated_uncertainty=al_aquisition_calibrated_uncertainty,
    random=al_aquisition_random,
)


def create_output_path_and_write_config(al_config: object) -> Path:
    hash = stable_hash_small(al_config)
    output_path = Path(__file__).parent / f"al_{hash}"
    if not output_path.is_dir():
        output_path.mkdir(exist_ok=True, parents=True)
        open(output_path / "al_config.json", "w").write(
            json_dumps_dataclass_str(al_config, indent=2)
        )
    return output_path


def al_do_step(al_step: ALStep, device, output_path):
    al_sample_ids = AQUISITION_FUNCTIONS[al_config.aquisition_method](
        al_step, output_path, device
    )
    new_pool_ids = list(set(al_step.pool_ids).difference(set(al_sample_ids)))
    al_subset_config = DataSubsetConfig(
        data_config=al_config.data_pool_config,
        subset=al_sample_ids,
        minimum_epoch_length=10000,
    )
    al_extended_ds_config = DataJoinConfig(
        data_configs=[
            al_config.ensemble_config.members[0].train_config.train_data_config,
            al_subset_config,
        ]
    )
    ds_pool = data_factory.get_factory().create(al_step.al_config.data_pool_config)
    open(
        output_path
        / f"step_{al_step.step:03d}_{len(al_sample_ids)}_{al_step.al_config.aquisition_method}_classes.json",
        "w",
    ).write(json.dumps([ds_pool[idx][1] for idx in al_sample_ids]))
    open(
        output_path
        / f"step_{al_step.step}_{len(al_sample_ids)}_{al_step.al_config.aquisition_method}.json",
        "w",
    ).write(json_dumps_dataclass_str(al_subset_config, indent=2))

    al_ensemble_config = create_ensemble_config(
        create_config_function(
            model_config=MLPClassConfig(widths=[128] * 2),
            batch_size=2**13,
            data_config=al_extended_ds_config,
            epochs=al_step.al_config.n_epochs_per_step,
        ),
        n_members=al_step.al_config.n_members,
    )
    al_ensemble = create_ensemble(al_ensemble_config, device_id)

    return ALStep(
        al_config=al_step.al_config,
        ensemble=al_ensemble,
        step=al_step.step + 1,
        pool_ids=new_pool_ids,
        rng=al_step.rng,
    )

    # # small_acc = list(filter(lambda x: x[0] < 0.5, accs_and_coords))
    # # Evenly spaced log in interval [10^-4, 10^0]
    # # n_pool_samples * 10^fracstart = 10^2 => frac_start = log10(10^2/n_pool_samples)
    # # frac_start = np.log10(al_config.n_start / n_pool_samples)
    # # frac_end = np.log10(al_config.n_end / n_pool_samples)
    # # for frac in np.logspace(
    # # start=frac_start, stop=frac_end, num=al_config.n_steps, base=10
    # # ):
    # # for frac in np.linspace(0.1, 1.0, 10):
    # frac = (al_config.n_end - al_config.n_start) / al_config.n_steps
    # al_ensemble, random_ensemble = train_on_fraction(frac)
    # return al_ensemble, random_ensemble


def evaluate(al_config: ALConfig, al_step: ALStep):
    df_rows = []
    # n_pool_samples = len(ds_pool)
    ds_pool = data_factory.get_factory().create(al_config.data_pool_config)

    n_aquired = al_config.n_start + int(
        al_step.step * (al_config.n_end - al_config.n_start) / al_config.n_steps
    )
    # for ensemble_name, ensemble in [("al", al_sal_ensemble), ("rnd", random_ensemble)]:
    metrics = create_classification_metric_dict(10)
    for member in al_step.ensemble.members:
        member.eval()

    evaluate_metrics_on_data(
        lambda x: ensemble_mean_prediction(al_step.ensemble, x),
        metrics,
        al_config.data_validation_config,
        2000,
        ComputeConfig(distributed=False, num_workers=8),
        device_id,
    )
    for metric_name, metric in metrics.items():
        df_rows.append(
            dict(
                model=al_config.aquisition_method,
                fraction=n_aquired / len(ds_pool),
                n_aquired=n_aquired,
                metric=metric_name,
                value=metric.mean(),
            )
        )
    return df_rows, ["model", "fraction", "n_aquired", "metric", "value"]


def do_al_steps(al_config: ALConfig, al_initial: ALStep, output_path: Path):
    df_rows = []
    df_columns = None
    al_step = al_initial
    for step in range(al_config.n_steps):
        next_step = al_do_step(al_step, device_id, output_path)
        df_new_rows, df_columns = evaluate(al_config, next_step)
        df_rows += df_new_rows
        al_step = next_step

    df = pd.DataFrame(columns=df_columns, data=df_rows)
    df.to_csv(output_path / f"{al_config.aquisition_method}.csv")
    print(f"Wrote {output_path / al_config.aquisition_method}.csv")


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
        n_members=5,
    )
    if slurm.get_task_id() is not None:
        train_member(ensemble_config_mlp, slurm.get_task_id(), device_id)

    if slurm.get_task_id() is not None and not is_ensemble_serialized(
        ensemble_config_mlp
    ):
        print("Exiting early since this is a SLURM array job used for training only")
        exit(0)

    ensemble_mlp = create_ensemble(ensemble_config_mlp, device_id)

    all_subsets = DataCIFAR10C.cifarc_subsets[:]
    uq_calibration_data_config = DataCIFAR10CConfig(
        subsets=all_subsets[::2], severities=[1, 2, 3, 4, 5]
    )
    _pool_ids = []
    for class_id in range(5, 10):
        class_ids = [x for x in class_and_id if x[0] == class_id]
        # Pick two samples per class randomly
        _pool_ids = _pool_ids + [
            class_ids[idx] for idx in rng_initial_data.permutation(len(class_ids))[:10]
        ]
    _pool_ids = [int(sample[1]) for sample in _pool_ids]
    _pool_ids = _pool_ids + list(initial_ids * 200)

    data_pool_config = DataSubsetConfig(data_config=DataCIFARConfig(), subset=_pool_ids)
    pool_ids = list(range(len(_pool_ids)))

    cifar_10_not_in_initial = list(
        set(range(len(ds_cifar_train))).difference(set(initial_ids))
    )
    uq_calibration_c10 = DataSubsetConfig(
        data_config=DataCIFARConfig(), subset=cifar_10_not_in_initial
    )

    al_configs = []
    for aquisition_method in AQUISITION_FUNCTIONS.keys():
        al_config = ALConfig(
            ensemble_config=ensemble_config_mlp,
            uq_calibration_data_config=uq_calibration_c10,
            data_validation_config=DataCIFARConfig(validation=True),
            data_pool_config=data_pool_config,
            aquisition_method=aquisition_method,
            n_epochs_per_step=50,
            n_members=5,
            n_start=50,
            n_end=1000,
            n_steps=10,
        )
        al_configs.append(al_config)

    output_path = create_output_path_and_write_config(al_configs)

    def do_al_for_config(al_config):
        ensemble = create_ensemble(al_config.ensemble_config, device_id)

        al_initial_step = ALStep(
            al_config=al_config,
            ensemble=ensemble,
            step=0,
            pool_ids=pool_ids,
            rng=np.random.default_rng(al_config.seed),
        )
        do_al_steps(al_config, al_initial_step, output_path)

    if slurm.get_task_id() is not None:
        # train_member(ensemble_config_mlp, slurm.get_task_id(), device_id)
        if slurm.get_task_id() > len(al_configs):
            print(f"No al config available for this id: {slurm.get_task_id()}")
            exit(0)
        do_al_for_config(al_configs[slurm.get_task_id()])
    else:
        print("Not in an array job so creating whole ensemble...")
        ensemble_mlp = create_ensemble(ensemble_config_mlp, device_id)
        for al_config in al_configs:
            do_al_for_config(al_config)

    print(f"Wrote results to {output_path}")
