#!/usr/bin/env python
import sys
import torch
import torchmetrics as tm

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.metric import Metric
from lib.models.dense import DenseConfig
from lib.data_registry import DataSineConfig

from lib.regression_metrics import create_regression_metrics

from lib.train import load_or_create_state
from lib.train import do_training
from lib.ddp import ddp_setup

from lib.files import prepare_results


def create_train_run():
    loss = torch.nn.functional.mse_loss

    def mse_loss(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=DenseConfig(d_hidden=100),
        train_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        val_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        loss=mse_loss,
        optimizer=OptimizerConfig(optimizer=torch.optim.Adam, kwargs=dict()),
        batch_size=2,
    )
    train_eval = create_regression_metrics(loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=1),
        train_config=train_config,
        train_eval=train_eval,
        epochs=20,
        save_nth_epoch=5,
        validate_nth_epoch=5,
        project="test",
    )
    return train_run


def main():
    device_id = ddp_setup("gloo")
    print(f"Using device {device_id}")
    print(__file__)
    print(" ".join(sys.argv))
    train_run = create_train_run()
    state = load_or_create_state(train_run, device_id)
    prepare_results("test", train_run.train_config)
    do_training(train_run, state, device_id)


if __name__ == "__main__":
    main()
