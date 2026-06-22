#!/usr/bin/env python
"""In-context operator learning via the operator-algebra transformer's [ICL]
sentence. Context (x_i, W x_i) -> emit rank-1 SVD-component partial operators
summing to the inferred W -> apply to query. Pure teacher forcing (op smooth-L1
+ STOP). Two-axis eval: error vs #context examples (ICL) and vs #partial
operators (test-time compute). Function-class spectra control difficulty.
"""
import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.train_dataclasses import TrainEval
from lib.metric import Metric
from lib.models.operator_algebra_transformer import OperatorAlgebraConfig
from lib.datasets.icl_operator import DataICLOperatorConfig
from lib.generic_ablation import get_config_grid
from lib.distributed_trainer import distributed_train
from experiments.realdet.icl_metric import compute_icl_diagnostics


def op_loss(output, batch):
    return output["op_loss"]


def stop_loss(output, batch):
    return output["stop_loss"]


def _make(n, k, spectrum, hidden, depth, num_heads, lr, seed, n_train, epochs):
    def model_loss(output, batch):
        return output["loss"].mean()

    train_eval = TrainEval(
        train_metrics=[lambda: Metric(op_loss), lambda: Metric(stop_loss)],
        validation_metrics=[lambda: Metric(op_loss), lambda: Metric(stop_loss)],
        log_gradient_norm=True, log_parameter_norm=True, log_sample_ids=False,
        diagnostics_interval=100,
    )
    td = DataICLOperatorConfig(n=n, k=k, spectrum=spectrum, n_train=n_train, seed=seed)
    vd = DataICLOperatorConfig(n=n, k=k, spectrum=spectrum, n_train=n_train, seed=seed, validation=True)
    tc = TrainConfig(
        model_config=OperatorAlgebraConfig(hidden=hidden, depth=depth, num_heads=num_heads),
        train_data_config=td, val_data_config=vd, loss=model_loss,
        optimizer=OptimizerConfig(optimizer=torch.optim.AdamW, kwargs=dict(lr=lr, weight_decay=0.0)),
        batch_size=512, gradient_clipping=1.0, ensemble_id=seed, _version=1,
    )
    return TrainRun(
        project="real_det_icl",
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=tc, train_eval=train_eval, epochs=epochs,
        save_nth_epoch=50, validate_nth_epoch=5, keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=50, visualize_interval_s=5,
        post_validate_hook=compute_icl_diagnostics,
    )


def create_icl_config(n, k, spectrum, hidden, depth, num_heads, lr, seed, n_train):
    return _make(n, k, spectrum, hidden, depth, num_heads, lr, seed, n_train, epochs=200)


def icl_configs():
    """d=4, k=8 (over-determined), function classes powerlaw/full/lowrank, 2 seeds."""
    return get_config_grid(
        create_icl_config,
        dict(n=[4], k=[8], spectrum=["powerlaw", "full", "lowrank"], hidden=[256],
             depth=[4], num_heads=[8], lr=[1e-4], seed=[0, 1], n_train=[200000]),
    )


def smoke_configs():
    return get_config_grid(
        create_icl_config,
        dict(n=[4], k=[8], spectrum=["powerlaw"], hidden=[128], depth=[2], num_heads=[4],
             lr=[1e-4], seed=[0], n_train=[20000]),
    )


def create_configs():
    return icl_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    import sys
    distributed_train(smoke_configs() if "--smoke" in sys.argv else create_configs())
