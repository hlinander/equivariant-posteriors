#!/usr/bin/env python
import torch
import torchmetrics as tm
from pathlib import Path

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.metric import Metric
from lib.models.transformer import TransformerConfig
from lib.data import DataSpiralsConfig
from lib.generic_ablation import generic_ablation


def loss(preds, target):
    return torch.nn.functional.binary_cross_entropy(preds, target, reduction="none")


def bce(preds, target):
    return tm.functional.classification.binary_calibration_error(
        preds, target, n_bins=15
    )


def create_config(mlp_dim, ensemble_id):
    train_config = TrainConfig(
        model_config=TransformerConfig(
            embed_d=50, mlp_dim=mlp_dim, n_seq=2, batch_size=500
        ),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.01)
        ),
        data_config=DataSpiralsConfig(),
        loss=torch.nn.BCELoss(),
        batch_size=500,
        ensemble_id=ensemble_id,
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
        epochs=500,
        save_nth_epoch=20,
    )
    return train_run


if __name__ == "__main__":
    generic_ablation(
        Path(__file__).parent / "results",
        create_config,
        dict(mlp_dim=[1, 5, 10, 20, 50], ensemble_id=list(range(5))),
    )
