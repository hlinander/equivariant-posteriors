"""
MNIST sweep over learning rates and model widths.

Usage:
    python run_slurm_sweep.py experiments/mnist/sweep_dense.py
    python run_slurm_sweep.py --dry-run experiments/mnist/sweep_dense.py
    python run_slurm_sweep.py --run-local experiments/mnist/sweep_dense.py
"""

import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.classification_metrics import create_classification_metrics
from lib.data_registry import DataMNISTConfig
from lib.datasets.mnist_visualization import visualize_mnist
from lib.models.transformer import TransformerConfig
from lib.generic_ablation import get_config_grid
from lib.distributed_trainer import distributed_train


def create_config(lr, width):
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss(output, batch):
        return ce_loss(output["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=TransformerConfig(
            embed_d=width,
            mlp_dim=width * 2,
            num_layers=2,
            num_heads=1,
            softmax=True,
        ),
        train_data_config=DataMNISTConfig(),
        val_data_config=DataMNISTConfig(validation=True),
        loss=loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam,
            kwargs=dict(lr=lr, weight_decay=1e-3),
        ),
        batch_size=256,
        ensemble_id=0,
        _version=1,
    )
    train_eval = create_classification_metrics(visualize_mnist, 10)
    return TrainRun(
        project="mnist",
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=20,
        save_nth_epoch=10,
        validate_nth_epoch=5,
    )


def create_configs():
    return get_config_grid(create_config, dict(
        lr=[1e-2, 1e-3, 1e-4],
        width=[32, 64, 128],
    ))


def run(config):
    distributed_train([config])
