#!/usr/bin/env python
import torch
import torchmetrics as tm
from pathlib import Path

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.metric import Metric
from lib.models.transformer import TransformerConfig
from lib.data_registry import DataSpiralsConfig
from lib.ablation import ablation
from lib.classification_metrics import create_classification_metrics


def create_config(embed_d):
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss(outputs, batch):
        return ce_loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=TransformerConfig(
            embed_d=embed_d,
            mlp_dim=1,
            n_seq=2,
            batch_size=500,
            num_layers=2,
            num_heads=1,
            softmax=True,
        ),
        train_data_config=DataSpiralsConfig(N=1000, seed=0),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.0001)
        ),
        loss=loss,
        batch_size=500,
        ensemble_id=0,
    )
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False),
        train_config=train_config,
        train_eval=create_classification_metrics(None, 2),
        epochs=500,
        save_nth_epoch=20,
        validate_nth_epoch=1,
    )
    return train_run


def create_values():
    return [1, 5, 20, 50, 100]


if __name__ == "__main__":
    ablation(Path(__file__).parent / "results", create_config, create_values())
