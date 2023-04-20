#!/usr/bin/env python
import torch
import torchmetrics as tm
from pathlib import Path

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.metric import Metric
from lib.data import DataSpiralsConfig
from lib.datasets.spiral_visualization import visualize_spiral
from lib.models.mlp import MLPClassConfig
from lib.generic_ablation import generic_ablation


def loss(preds, target):
    return torch.nn.functional.binary_cross_entropy(preds, target, reduction="none")


def bce(preds, target):
    return tm.functional.classification.binary_calibration_error(
        preds, target, n_bins=15
    )


def create_config(mlp_dim, ensemble_id):
    train_config = TrainConfig(
        model_config=MLPClassConfig(width=mlp_dim),
        train_data_config=DataSpiralsConfig(seed=0, N=1000),
        val_data_config=DataSpiralsConfig(seed=1, N=500),
        loss=torch.nn.BCELoss(),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.01)
        ),
        batch_size=500,
        ensemble_id=ensemble_id,
    )
    train_eval = TrainEval(
        train_metrics=[
            lambda: Metric(
                tm.functional.accuracy,
                metric_kwargs=dict(task="binary", multidim_average="samplewise"),
            ),
            lambda: Metric(loss),
            lambda: Metric(bce),
        ],
        validation_metrics=[
            lambda: Metric(
                tm.functional.accuracy,
                metric_kwargs=dict(task="binary", multidim_average="samplewise"),
            ),
            lambda: Metric(bce),
            lambda: Metric(loss),
        ],
        data_visualizer=visualize_spiral,
    )
    train_run = TrainRun(
        train_config=train_config,
        train_eval=train_eval,
        epochs=500,
        save_nth_epoch=20,
        validate_nth_epoch=20,
    )
    return train_run


if __name__ == "__main__":
    generic_ablation(
        Path(__file__).parent / "results",
        create_config,
        dict(mlp_dim=[100, 10, 50, 100, 200], ensemble_id=list(range(5))),
    )
