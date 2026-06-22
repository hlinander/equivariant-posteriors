#!/usr/bin/env python
"""Multitask operator transformer: DET + INV "sentences" sharing the operator
vocabulary, one model. Teacher-forced (op smooth-L1 + STOP BCE); per-task
free-rollout metrics (DET log|det| MAE; INV reduction + inverse error).

The test: can one model do both -- reduce-to-triangular for det AND
reduce-to-identity for inverse -- with the same predicted-operator machinery?
"""
import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.train_dataclasses import TrainEval
from lib.metric import Metric
from lib.models.matrix_operator_multi import MatrixOperatorMultiConfig
from lib.datasets.real_det_multi import DataRealDetMultiConfig
from lib.generic_ablation import get_config_grid
from lib.distributed_trainer import distributed_train
from experiments.realdet.operator_multi_metric import compute_multi_diagnostics


def op_loss(output, batch):
    return output["op_loss"]


def stop_loss(output, batch):
    return output["stop_loss"]


def _make_train_run(n, hidden, depth, num_heads, lr, seed, n_train, epochs):
    def model_loss(output, batch):
        return output["loss"].mean()

    train_eval = TrainEval(
        train_metrics=[lambda: Metric(op_loss), lambda: Metric(stop_loss)],
        validation_metrics=[lambda: Metric(op_loss), lambda: Metric(stop_loss)],
        log_gradient_norm=True,
        log_parameter_norm=True,
        log_sample_ids=False,
        diagnostics_interval=100,
    )
    train_data = DataRealDetMultiConfig(n=n, n_train=n_train, seed=seed)
    val_data = DataRealDetMultiConfig(n=n, n_train=n_train, seed=seed, validation=True)
    train_config = TrainConfig(
        model_config=MatrixOperatorMultiConfig(hidden=hidden, depth=depth, num_heads=num_heads),
        train_data_config=train_data,
        val_data_config=val_data,
        loss=model_loss,
        optimizer=OptimizerConfig(optimizer=torch.optim.AdamW, kwargs=dict(lr=lr, weight_decay=0.0)),
        batch_size=512,
        gradient_clipping=1.0,
        ensemble_id=seed,
        _version=1,
    )
    return TrainRun(
        project="real_det_operator_multi",
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=epochs,
        save_nth_epoch=50,
        validate_nth_epoch=5,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=50,
        visualize_interval_s=5,
        post_validate_hook=compute_multi_diagnostics,
    )


def create_multi_config(n, hidden, depth, num_heads, lr, seed, n_train):
    return _make_train_run(n, hidden, depth, num_heads, lr, seed, n_train, epochs=200)


def multi_configs():
    """DET+INV multitask, N=4, 2 seeds."""
    return get_config_grid(
        create_multi_config,
        dict(n=[4], hidden=[256], depth=[4], num_heads=[8], lr=[1e-4], seed=[0, 1],
             n_train=[200000]),
    )


def smoke_configs():
    return get_config_grid(
        create_multi_config,
        dict(n=[4], hidden=[128], depth=[2], num_heads=[4], lr=[1e-4], seed=[0],
             n_train=[20000]),
    )


def create_configs():
    return multi_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    import sys
    distributed_train(smoke_configs() if "--smoke" in sys.argv else create_configs())
