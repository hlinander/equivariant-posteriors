#!/usr/bin/env python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from pathlib import Path
import pandas as pd

# import tqdm

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metrics
from lib.data_factory import DataCIFARConfig, DataCIFAR10CConfig

import lib.data_factory as data_factory
import lib.slurm as slurm

# from lib.models.mlp_proj import MLPProjClassConfig
from lib.models.mlp import MLPClassConfig
from lib.models.conv import ConvConfig

# from lib.lyapunov import lambda1
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import train_member
from lib.uncertainty import uncertainty
from lib.files import prepare_results

# import rplot


def create_config_function(model_config: object, batch_size: int):
    def create_config(ensemble_id):
        loss = torch.nn.CrossEntropyLoss()

        def ce_loss(outputs, targets):
            return loss(outputs["logits"], targets)

        train_config = TrainConfig(
            # model_config=MLPClassConfig(widths=[50, 50]),
            model_config=model_config,  # MLPClassConfig(widths=[128] * 2),
            train_data_config=DataCIFARConfig(),
            val_data_config=DataCIFARConfig(validation=True),
            loss=ce_loss,
            optimizer=OptimizerConfig(
                optimizer=torch.optim.SGD,
                kwargs=dict(weight_decay=1e-4, lr=0.05, momentum=0.9),
                # kwargs=dict(weight_decay=0.0, lr=0.001),
            ),
            batch_size=batch_size,  # 2**13,
            ensemble_id=ensemble_id,
        )
        train_eval = create_classification_metrics(None, 10)
        train_run = TrainRun(
            compute_config=ComputeConfig(distributed=False, num_workers=16),
            train_config=train_config,
            train_eval=train_eval,
            epochs=300,
            save_nth_epoch=1,
            validate_nth_epoch=5,
        )
        return train_run

    return create_config


if __name__ == "__main__":
    device_id = ddp_setup()

    ensemble_config_mlp = create_ensemble_config(
        create_config_function(
            model_config=MLPClassConfig(widths=[128] * 2), batch_size=2**13
        ),
        n_members=10,
    )
    if slurm.get_task_id() is not None:
        train_member(ensemble_config_mlp, slurm.get_task_id(), device_id)
    else:
        print("Not in an array job so creating whole ensemble...")
        ensemble_mlp = create_ensemble(ensemble_config_mlp, device_id)

    ensemble_config_conv = create_ensemble_config(
        create_config_function(model_config=ConvConfig(), batch_size=3000),
        n_members=10,
    )
    if slurm.get_task_id() is not None:
        train_member(ensemble_config_conv, slurm.get_task_id(), device_id)
    else:
        ensemble_conv = create_ensemble(ensemble_config_conv, device_id)

    if slurm.get_task_id() is not None:
        print("Exiting early since this is a SLURM array job used for training only")
        exit(0)

    ds_cifar_val = data_factory.get_factory().create(DataCIFARConfig(validation=True))
    dl_cifar_val = torch.utils.data.DataLoader(
        ds_cifar_val,
        batch_size=8,
        shuffle=False,
        drop_last=False,
    )

    ds_cifar_c = data_factory.get_factory().create(
        DataCIFAR10CConfig(subset="impulse_noise", severity=1)
    )
    dl_cifar_c = torch.utils.data.DataLoader(
        ds_cifar_c,
        batch_size=8,
        shuffle=False,
        drop_last=False,
    )

    def uq_for_ensemble(ensemble, ensemble_config, model_name: str):
        uq_cifar_val = uncertainty(dl_cifar_val, ensemble, device_id)
        uq_cifar_c = uncertainty(dl_cifar_c, ensemble, device_id)

        def save_uq(config, uq, filename):
            result_path = prepare_results(
                Path(__file__).parent, Path(__file__).stem, config
            )
            data = torch.concat(
                [
                    uq.MI[:, None].cpu(),
                    uq.H[:, None].cpu(),
                    uq.sample_ids[:, None].cpu(),
                    uq.mean_pred[:, None].cpu(),
                    uq.targets[:, None].cpu(),
                ],
                dim=-1,
            )
            df = pd.DataFrame(
                columns=["MI", "H", "id", "pred", "target"], data=data.numpy()
            )

            df.to_csv(result_path / filename)

        save_uq(ensemble_config, uq_cifar_val, f"{model_name}_uq_cifar_c.csv")
        save_uq(ensemble_config, uq_cifar_c, f"{model_name}_uq_cifar_c.csv")

    uq_for_ensemble(ensemble_mlp, ensemble_config_mlp, "mlp")
    uq_for_ensemble(ensemble_conv, ensemble_config_conv, "conv")
