#!/usr/bin/env python
"""Learning curves for the determinant of real N x N matrices.

Goal: estimate how much data is needed to learn det over R, and how that data
requirement scales with N. For each N we sweep the training-set size n_train
and measure held-out sign accuracy and R^2 on log|det| (R^2 = 1 - MSE, since
the target is standardized to unit variance). Fitting test error E(n) ~
a*n^-b + c per N and inverting gives n*(N), the data to reach a target error;
n*(N) vs N is the answer.

Baseline model is a plain MLP (no determinant-specific inductive bias), which
is expected to hit the factorial wall: det has N! Leibniz terms, so a
structure-agnostic learner needs ~N! samples. Later arms (permutation-
equivariant / multilinear) measure how far the right prior bends that curve.
"""
import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.train_dataclasses import TrainEval
from lib.metric import Metric
from lib.models.mlp import MLPConfig
from lib.datasets.real_det import DataRealDetConfig
from lib.generic_ablation import get_config_grid
from lib.distributed_trainer import distributed_train


def sign_logabs_loss(output, batch):
    """BCE on sign (column 0) + MSE on standardized log|det| (column 1)."""
    logits = output["logits"]
    target = batch["target"]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], target[:, 0]
    )
    mse = torch.nn.functional.mse_loss(logits[:, 1], target[:, 1])
    return bce + mse


def sign_acc(output, batch):
    """Per-sample sign-prediction correctness (mean -> sign accuracy)."""
    pred = (output["logits"][:, 0] > 0).float()
    return (pred == batch["target"][:, 0]).float()


def logabs_mse(output, batch):
    """Per-sample squared error on standardized log|det| (mean -> MSE = 1-R^2)."""
    return (output["logits"][:, 1] - batch["target"][:, 1]) ** 2


def _metric_list():
    return [
        lambda: Metric(sign_acc),
        lambda: Metric(logabs_mse),
    ]


def _make_train_run(n, n_train, width, depth, lr, weight_decay, seed, n_val, epochs):
    train_eval = TrainEval(
        train_metrics=_metric_list(),
        validation_metrics=_metric_list(),
        log_gradient_norm=True,
        log_parameter_norm=True,
        log_sample_ids=False,
        diagnostics_interval=100,
    )

    train_data = DataRealDetConfig(
        n=n, n_train=n_train, n_val=n_val, seed=seed
    )
    val_data = DataRealDetConfig(
        n=n, n_train=n_train, n_val=n_val, seed=seed, validation=True
    )

    train_config = TrainConfig(
        model_config=MLPConfig(widths=[width] * depth, activation="relu"),
        train_data_config=train_data,
        val_data_config=val_data,
        loss=sign_logabs_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(lr=lr, weight_decay=weight_decay),
        ),
        batch_size=min(1024, n_train),
        ensemble_id=seed,
        _version=1,
    )

    return TrainRun(
        project="real_det",
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=epochs,
        save_nth_epoch=100,
        validate_nth_epoch=5,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=100,
        visualize_interval_s=5,
    )


def create_learning_curve_config(n, n_train, width, depth, lr, weight_decay, seed):
    return _make_train_run(
        n=n,
        n_train=n_train,
        width=width,
        depth=depth,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        n_val=20000,
        epochs=300,
    )


def learning_curve_configs():
    """MLP learning curves: N in {2,3,4,5} x geometric n_train x 3 seeds."""
    return get_config_grid(
        create_learning_curve_config,
        dict(
            n=[2, 3, 4, 5],
            n_train=[125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
            width=[512],
            depth=[3],
            lr=[1e-3],
            weight_decay=[1e-2],
            seed=[0, 1, 2],
        ),
    )


def smoke_configs():
    """Tiny set to validate the pipeline end to end."""
    return get_config_grid(
        create_learning_curve_config,
        dict(
            n=[2, 3],
            n_train=[256, 1024],
            width=[256],
            depth=[3],
            lr=[1e-3],
            weight_decay=[1e-2],
            seed=[0],
        ),
    )


def create_configs():
    return learning_curve_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        distributed_train(smoke_configs())
    else:
        distributed_train(create_configs())
