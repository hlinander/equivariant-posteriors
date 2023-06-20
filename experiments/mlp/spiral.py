#!/usr/bin/env python
import torch
from pathlib import Path

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metrics
from lib.data_factory import DataSpiralsConfig
from lib.datasets.spiral_visualization import visualize_spiral
from lib.models.mlp import MLPClassConfig
from lib.generic_ablation import generic_ablation


def create_config(mlp_dim, ensemble_id):
    train_config = TrainConfig(
        model_config=MLPClassConfig(width=mlp_dim),
        train_data_config=DataSpiralsConfig(seed=0, N=1000),
        val_data_config=DataSpiralsConfig(seed=1, N=500),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.0001)
        ),
        batch_size=500,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(visualize_spiral, 2)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False),
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
        dict(mlp_dim=[200, 10, 50, 100, 200], ensemble_id=list(range(5))),
    )
