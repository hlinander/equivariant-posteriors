#!/usr/bin/env python
import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.classification_metrics import create_classification_metrics
from lib.models.grok_mlp import GrokMLPConfig
from lib.datasets.finite_field_det import DataFiniteFieldDetConfig
from lib.generic_ablation import get_config_grid
from lib.distributed_trainer import distributed_train


def create_config(width, depth, n, p, train_size, weight_decay, lr, ensemble_id):
    loss = torch.nn.CrossEntropyLoss()

    def ce_loss(output, batch):
        return loss(output["logits"], batch["target"])

    train_eval = create_classification_metrics(None, p)
    train_eval.log_gradient_norm = True
    train_eval.log_parameter_norm = True
    train_eval.diagnostics_interval = 1

    train_config = TrainConfig(
        model_config=GrokMLPConfig(width=width, depth=depth, activation="gelu"),
        train_data_config=DataFiniteFieldDetConfig(
            n=n, p=p, n_samples=train_size, seed=ensemble_id
        ),
        val_data_config=DataFiniteFieldDetConfig(
            n=n, p=p, n_samples=10000, seed=1000 + ensemble_id
        ),
        loss=ce_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(lr=lr, weight_decay=weight_decay),
        ),
        batch_size=min(train_size, 10000),
        ensemble_id=ensemble_id,
    )

    return TrainRun(
        project="grokking_det",
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=100000,
        save_nth_epoch=1000,
        validate_nth_epoch=500,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=5000,
        visualize_interval_s=5,
    )


if __name__ == "__main__":
    distributed_train(
        get_config_grid(
            create_config,
            dict(
                width=[512],
                depth=[3, 4, 6],
                n=[2],
                p=[5],
                train_size=[300, 1000, 3000, 10000, 30000],
                weight_decay=[10.0, 1.0, 0.1, 0.3],
                lr=[1e-3],
                ensemble_id=[0],
            ),
        )
    )
