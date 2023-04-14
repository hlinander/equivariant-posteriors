#!/usr/bin/env python
import os
import torch
import torchmetrics as tm
from pathlib import Path

from lib.train import TrainConfig
from lib.train import TrainEval
from lib.train import TrainRun
from lib.metric import Metric
from lib.models.dense import DenseConfig
from lib.data import DataSineConfig

from lib.train import load_or_create_state
from lib.train import do_training


def main():
    if torch.cuda.is_available():
        device_id = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    else:
        device_id = "cpu"

    print(f"Using device {device_id}")

    train_config = TrainConfig(
        model_config=DenseConfig(d_hidden=100),
        data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        loss=torch.nn.MSELoss(),
        batch_size=2,
    )
    train_eval = TrainEval(
        metrics=[
            lambda: Metric(tm.functional.mean_absolute_error),
            lambda: Metric(tm.functional.mean_squared_error),
        ],
    )
    train_run = TrainRun(
        train_config=train_config,
        train_eval=train_eval,
        epochs=10,
    )
    state = load_or_create_state(train_run, device_id)
    do_training(train_run, state, device_id)


if __name__ == "__main__":
    main()
