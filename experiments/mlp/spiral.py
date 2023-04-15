#!/usr/bin/env python
import torch
import torchmetrics as tm
from pathlib import Path

from lib.train import TrainConfig
from lib.train import TrainEval
from lib.train import TrainRun
from lib.metric import Metric
from lib.data import DataSpiralsConfig
from lib.models.mlp import MLPClassConfig
from lib.ablation import ablation


def loss(preds, target):
    return torch.nn.functional.binary_cross_entropy(preds, target, reduction="none")


def bce(preds, target):
    return tm.functional.classification.binary_calibration_error(
        preds, target, n_bins=15
    )


def create_config(mlp_dim):
    train_config = TrainConfig(
        model_config=MLPClassConfig(width=mlp_dim),
        data_config=DataSpiralsConfig(),
        loss=torch.nn.BCELoss(),
        batch_size=500,
        ensemble_id=0,
    )
    train_eval = TrainEval(
        metrics=[
            lambda: Metric(
                tm.functional.accuracy,
                metric_kwargs=dict(task="binary", multidim_average="samplewise"),
            ),
            lambda: Metric(loss),
            lambda: Metric(bce),
        ],
    )
    train_run = TrainRun(
        train_config=train_config,
        train_eval=train_eval,
        epochs=300,
    )
    return train_run


if __name__ == "__main__":
    ablation(Path(__file__).parent / "results", create_config, [10, 50, 100, 200])
