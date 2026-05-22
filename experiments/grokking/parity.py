#!/usr/bin/env python
import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.classification_metrics import create_classification_metrics
from lib.models.grok_mlp import GrokMLPConfig
from lib.datasets.sparse_parity import DataSparseParityConfig
from lib.generic_ablation import get_config_grid
from lib.distributed_trainer import distributed_train
from experiments.grokking.visualization import visualize_parity


def create_config(width, depth, k, train_size, weight_decay, lr, ensemble_id):
    loss = torch.nn.CrossEntropyLoss()

    def ce_loss(output, batch):
        return loss(output["logits"], batch["target"])

    train_eval = create_classification_metrics(visualize_parity, 2)
    train_eval.log_gradient_norm = True
    train_eval.log_parameter_norm = True
    train_eval.diagnostics_interval = 1

    train_config = TrainConfig(
        model_config=GrokMLPConfig(width=width, depth=depth, activation="relu"),
        train_data_config=DataSparseParityConfig(
            d=40, k=k, n_samples=train_size, seed=ensemble_id, subset_seed=0
        ),
        val_data_config=DataSparseParityConfig(
            d=40, k=k, n_samples=50000, seed=1000 + ensemble_id, subset_seed=0
        ),
        loss=ce_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(lr=lr, weight_decay=weight_decay),
        ),
        batch_size=train_size,
        ensemble_id=ensemble_id,
    )

    return TrainRun(
        project="grokking_parity",
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=100000,
        save_nth_epoch=1000,
        validate_nth_epoch=500,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=5000,
    )


if __name__ == "__main__":
    distributed_train(
        get_config_grid(
            create_config,
            dict(
                width=[256],
                depth=[2, 4],
                k=[3, 4],
                train_size=[2048, 512, 1024],
                weight_decay=[1.0, 0.1, 1e-2],
                lr=[1e-3],
                ensemble_id=[0],
            ),
        )
    )
