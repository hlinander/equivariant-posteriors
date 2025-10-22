"""
Shared test fixtures and utilities for pytest
"""
import torch
from lib.train_dataclasses import TrainConfig, TrainEval, TrainRun, OptimizerConfig, ComputeConfig
from lib.models.dense import DenseConfig
from lib.data_registry import DataSineConfig
from lib.regression_metrics import create_regression_metrics


def create_train_run():
    """Create a simple test training run configuration"""
    loss = torch.nn.functional.mse_loss

    def mse_loss(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=DenseConfig(d_hidden=100),
        train_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        val_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        loss=mse_loss,
        optimizer=OptimizerConfig(optimizer=torch.optim.Adam, kwargs=dict()),
        batch_size=2,
    )
    train_eval = create_regression_metrics(loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=1),
        train_config=train_config,
        train_eval=train_eval,
        epochs=20,
        save_nth_epoch=5,
        validate_nth_epoch=5,
        project="test",
    )
    return train_run
