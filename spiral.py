#!/usr/bin/env python
import os
import torch
import torchmetrics as tm
from pathlib import Path

from lib.train import TrainConfig
from lib.metric import Metric
from lib.models.transformer import TransformerConfig
from lib.data import DataSpiralsConfig


from lib.train import load_state
from lib.train import create_initial_state
from lib.train import do_training


def loss(preds, target):
    return torch.nn.functional.binary_cross_entropy(preds, target, reduction="none")


def main():
    if torch.cuda.is_available():
        device_id = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    else:
        device_id = "cpu"

    print(f"Using device {device_id}")

    train_config = TrainConfig(
        model_config=TransformerConfig(embed_d=30, mlp_dim=20, n_seq=2, batch_size=500),
        data_config=DataSpiralsConfig(),
        loss=torch.nn.BCELoss(),
        batch_size=500,
        epochs=1000,
        metrics=[
            lambda: Metric(
                tm.functional.accuracy,
                metric_kwargs=dict(task="binary", multidim_average="samplewise"),
            ),
            lambda: Metric(loss),
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
