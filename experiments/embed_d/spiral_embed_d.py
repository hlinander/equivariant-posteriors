#!/usr/bin/env python
import torch
import torchmetrics as tm
from pathlib import Path

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.models.transformer import TransformerConfig
from lib.data_registry import DataSpiralsConfig
from lib.datasets.spiral_visualization import visualize_spiral
from lib.generic_ablation import generic_ablation
from lib.classification_metrics import create_classification_metrics
from lib.distributed_trainer import distributed_train
from lib.ddp import ddp_setup
import lib.data_factory as data_factory

from lib.serialization import (
    deserialize_model,
    DeserializeConfig,
)


def loss(preds, target):
    return torch.nn.functional.binary_cross_entropy(preds, target, reduction="none")


def bce(preds, target):
    return tm.functional.classification.binary_calibration_error(
        preds, target, n_bins=15
    )


def create_config(embed_d, ensemble_id):
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss(outputs, batch):
        return ce_loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=TransformerConfig(
            embed_d=embed_d,
            mlp_dim=50,
            n_seq=20,
            batch_size=500,
            num_layers=2,
            num_heads=1,
            softmax=True,
        ),
        train_data_config=DataSpiralsConfig(seed=0, N=1000),
        val_data_config=DataSpiralsConfig(seed=1, N=500),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.0001)
        ),
        loss=loss,
        batch_size=500,
        ensemble_id=ensemble_id,
        _version=1,
    )
    train_eval = create_classification_metrics(visualize_spiral, 2)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=1),
        train_config=train_config,
        train_eval=train_eval,
        epochs=500,
        save_nth_epoch=20,
        validate_nth_epoch=20,
        visualize_terminal_interval=5,
    )
    return train_run


def create_values():
    return dict(embed_d=[100], ensemble_id=list(range(1)))


if __name__ == "__main__":
    configs = generic_ablation(create_config, create_values())
    distributed_train(configs)
    config = configs[0]

    device_id = ddp_setup()

    deserialized_model = deserialize_model(DeserializeConfig(config, device_id))
    model = deserialized_model.model

    ds = data_factory.get_factory().create(config.train_config.val_data_config)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=config.train_config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    for batch in dl:
        
