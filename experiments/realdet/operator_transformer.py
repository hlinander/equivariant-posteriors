#!/usr/bin/env python
"""Single-task DET via the matrix-operator-token transformer (the general
substrate, first instance). Tokens are matrices; the model autoregressively
predicts elimination operators (Delta from identity) with a discrete STOP,
trained teacher-forced (delta + stop losses) as a parallel causal sequence
model. Free-rollout logdet MAE + avg ops are logged by the hook.
"""
import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.train_dataclasses import TrainEval
from lib.metric import Metric
from lib.models.matrix_operator_transformer import MatrixOperatorTransformerConfig
from lib.datasets.real_det_matrix import DataRealDetMatrixConfig
from lib.generic_ablation import get_config_grid
from lib.distributed_trainer import distributed_train
from experiments.realdet.operator_metric import compute_operator_diagnostics


def op_loss(output, batch):
    return output["op_loss"]


def stop_loss(output, batch):
    return output["stop_loss"]


def _make_train_run(n, hidden, depth, num_heads, lr, seed, epochs, n_train=100000):
    def op_total_loss(output, batch):
        return output["op_loss"].mean() + output["stop_loss"].mean()

    train_eval = TrainEval(
        train_metrics=[lambda: Metric(op_loss), lambda: Metric(stop_loss)],
        validation_metrics=[lambda: Metric(op_loss), lambda: Metric(stop_loss)],
        log_gradient_norm=True,
        log_parameter_norm=True,
        log_sample_ids=False,
        diagnostics_interval=100,
    )
    train_data = DataRealDetMatrixConfig(n=n, n_train=n_train, seed=seed)
    val_data = DataRealDetMatrixConfig(n=n, n_train=n_train, seed=seed, validation=True)
    train_config = TrainConfig(
        model_config=MatrixOperatorTransformerConfig(
            hidden=hidden, depth=depth, num_heads=num_heads, slack=4
        ),
        train_data_config=train_data,
        val_data_config=val_data,
        loss=op_total_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW, kwargs=dict(lr=lr, weight_decay=0.0)
        ),
        batch_size=512,
        gradient_clipping=1.0,
        ensemble_id=seed,
        _version=2,  # full-operator (vs I+Delta) redesign -> fresh hashes
    )
    return TrainRun(
        project="real_det_operator",
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=epochs,
        save_nth_epoch=50,
        validate_nth_epoch=5,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=50,
        visualize_interval_s=5,
        post_validate_hook=compute_operator_diagnostics,
    )


def create_det_config(n, hidden, depth, num_heads, lr, seed, n_train=100000):
    return _make_train_run(
        n=n, hidden=hidden, depth=depth, num_heads=num_heads, lr=lr, seed=seed,
        epochs=200, n_train=n_train,
    )


def det_configs():
    """Single-task DET via full-operator prediction (pivoted teacher), N in
    {3,4,5} -- the scaling test: does the parallel causal sequence model break
    the N>=4 wall the unrolled rollout couldn't?"""
    return get_config_grid(
        create_det_config,
        dict(n=[3, 4, 5], hidden=[256], depth=[4], num_heads=[8], lr=[1e-4], seed=[0, 1]),
    )


def data_scaling_configs():
    """Data-efficiency study for the operator transformer at N=4,5: does more
    data lower op_loss (hence less free-rollout drift / logdet_mae)?"""
    return get_config_grid(
        create_det_config,
        dict(
            n=[4, 5], hidden=[256], depth=[4], num_heads=[8], lr=[1e-4], seed=[0],
            n_train=[25000, 50000, 100000, 200000],
        ),
    )


def smoke_configs():
    return get_config_grid(
        create_det_config,
        dict(n=[3], hidden=[128], depth=[2], num_heads=[4], lr=[1e-4], seed=[0]),
    )


def create_configs():
    return det_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        distributed_train(smoke_configs())
    else:
        distributed_train(create_configs())
