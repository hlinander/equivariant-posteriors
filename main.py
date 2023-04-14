#!/usr/bin/env python
import os
import torch
import torchmetrics as tm
from pathlib import Path

from lib.train import TrainConfig
from lib.metric import Metric
from lib.models.dense import DenseConfig
from lib.data import DataConfig

from lib.train import load_state
from lib.train import create_initial_state
from lib.train import do_training


def main():
    if torch.cuda.is_available():
        device_id = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    else:
        device_id = "cpu"

    print(f"Using device {device_id}")

    train_config = TrainConfig(
        model_config=DenseConfig(d_hidden=100),
        data_config=DataConfig(
            torch.Size([1]), output_shape=torch.Size([1]), batch_size=2
        ),
        epochs=10,
        metrics=[
            lambda: Metric(tm.functional.mean_absolute_error),
            lambda: Metric(tm.functional.mean_squared_error),
        ],
    )
    checkpoint_path = Path("checkpoint.pt")

    if checkpoint_path.is_file():
        state = load_state(train_config, checkpoint_path, device_id)
    else:
        state = create_initial_state(train_config, device_id)

    do_training(train_config, state, device_id)


if __name__ == "__main__":
    main()
