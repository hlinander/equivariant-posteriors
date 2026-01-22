#!/usr/bin/env python
import torch
import numpy as np
from pathlib import Path

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.metric import create_metric
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import request_ensemble
from lib.ensemble import is_ensemble_serialized
from lib.distributed_trainer import distributed_train

from experiments.weather.models.pangu import PanguConfig
from experiments.weather.data import DataHPConfig

NSIDE = 64


def create_config(ensemble_id, epoch):
    loss = torch.nn.L1Loss()

    def reg_loss(output, batch):
        return loss(output["logits_upper"], batch["target_upper"]) + 0.25 * loss(
            output["logits_surface"], batch["target_surface"]
        )

    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=PanguConfig(nside=64),
        train_data_config=DataHPConfig(nside=64, driscoll_healy=True),
        val_data_config=None,  # DataHPConfig(nside=NSIDE),
        loss=reg_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
        ),
        batch_size=1,
        ensemble_id=ensemble_id,
        _version=5,
    )
    train_eval = TrainEval(
        train_metrics=[create_metric(reg_loss)],
        validation_metrics=[],
        log_gradient_norm=True,
    )
    train_run = TrainRun(
        project="weather",
        compute_config=ComputeConfig(),
        train_config=train_config,
        train_eval=train_eval,
        epochs=epoch,
        save_nth_epoch=1,
        validate_nth_epoch=20,
        keep_nth_epoch_checkpoints=10,
        keep_epoch_checkpoints=True,
        visualize_terminal=False,
    )
    return train_run


if __name__ == "__main__":
    device_id = ddp_setup()

    def oom_observer(device, alloc, device_alloc, device_free):
        print("saving allocated state during OOM")
        torch.cuda.memory._dump_snapshot("oom_snapshot_new.pickle")

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    ensemble_config = create_ensemble_config(lambda eid: create_config(eid, 250), 1)
    if not is_ensemble_serialized(ensemble_config):
        request_ensemble(ensemble_config)
        distributed_train(ensemble_config.members)
