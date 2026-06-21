#!/usr/bin/env python
"""Stage-1 elimination-rollout experiment: learn log|det| of real NxN matrices
by a det-preserving sequence of elementary row operations (fixed order, learned
multipliers), reading off Sum_d log|diag|.

Loss = L1 on log|det| (the readout) + lam * lower-tri mass (forces genuine
triangularization, so the model can't cheat the fixed readout). Metrics:
logdet_mae and lower_rms; a post-validate hook logs the oracle ceiling,
det-conservation sanity, and end-of-rollout drift.
"""
import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.train_dataclasses import TrainEval
from lib.metric import Metric
from lib.models.elimination_rollout import EliminationRolloutConfig
from lib.datasets.real_det_matrix import DataRealDetMatrixConfig
from lib.generic_ablation import get_config_grid
from lib.distributed_trainer import distributed_train
from experiments.realdet.elim_metric import compute_elim_diagnostics


def logdet_mae(output, batch):
    return (output["logits"].squeeze(-1) - batch["target"].squeeze(-1)).abs()


def lower_rms(output, batch):
    return output["lower_tri_sq"].clamp_min(0).sqrt()


def _metric_list():
    return [lambda: Metric(logdet_mae), lambda: Metric(lower_rms)]


def _make_train_run(
    n, n_train, hidden, depth, lr, weight_decay, seed, lam, epochs,
    input_features="raw",
):
    # Closure so the loss hashes by __name__ (module-path-independent); lam is
    # captured, so keep it fixed across the sweep to avoid hash collisions.
    def elim_loss(output, batch):
        pred = output["logits"].squeeze(-1)
        tgt = batch["target"].squeeze(-1)
        readout = (pred - tgt).abs().mean()
        triangularize = output["lower_tri_sq"].mean()
        return readout + lam * triangularize

    train_eval = TrainEval(
        train_metrics=_metric_list(),
        validation_metrics=_metric_list(),
        log_gradient_norm=True,
        log_parameter_norm=True,
        log_sample_ids=False,
        diagnostics_interval=100,
    )
    train_data = DataRealDetMatrixConfig(n=n, n_train=n_train, seed=seed)
    val_data = DataRealDetMatrixConfig(n=n, n_train=n_train, seed=seed, validation=True)

    train_config = TrainConfig(
        model_config=EliminationRolloutConfig(
            hidden=hidden, depth=depth, input_features=input_features
        ),
        train_data_config=train_data,
        val_data_config=val_data,
        loss=elim_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(lr=lr, weight_decay=weight_decay),
        ),
        batch_size=1024,
        gradient_clipping=1.0,
        ensemble_id=seed,
        _version=1,
    )

    return TrainRun(
        project="real_det_elim",
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=epochs,
        save_nth_epoch=50,
        validate_nth_epoch=5,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=50,
        visualize_interval_s=5,
        post_validate_hook=compute_elim_diagnostics,
    )


def create_elim_config(n, hidden, depth, lr, weight_decay, seed, input_features):
    return _make_train_run(
        n=n, n_train=100000, hidden=hidden, depth=depth, lr=lr,
        weight_decay=weight_decay, seed=seed, lam=1.0, epochs=200,
        input_features=input_features,
    )


def elimination_configs():
    """Stage 1 feasibility/stability across small N, comparing input
    featurizations: raw vs log|M|+sign vs both (the multiplier is a ratio, so
    log -> subtraction should ease the division)."""
    return get_config_grid(
        create_elim_config,
        dict(
            n=[2, 3, 4],
            hidden=[256],
            depth=[2],
            lr=[1e-3],
            weight_decay=[0.0],
            seed=[0, 1],
            input_features=["raw", "log", "both"],
        ),
    )


def smoke_configs():
    return get_config_grid(
        create_elim_config,
        dict(
            n=[2, 3], hidden=[128], depth=[2], lr=[1e-3], weight_decay=[0.0],
            seed=[0], input_features=["both"],
        ),
    )


def create_configs():
    return elimination_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        distributed_train(smoke_configs())
    else:
        distributed_train(create_configs())
