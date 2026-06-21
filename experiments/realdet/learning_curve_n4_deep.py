#!/usr/bin/env python
"""Sweep: deep-data extension of the N=4 real-determinant learning curve to
find the function-learning crossing n*(cos>=0.9), which the v2 grid left at
>128k (best jac_cos 0.78 at 128k). Same v2 protocol (fixed ~30k steps,
_version=2) so the 128k point is identical to the v2 run (resumes as anchor);
256k/512k/1M are fresh.

N=4 only, width fixed at 512x3 -- if jac_cos plateaus below 0.9 here it points
to a capacity wall rather than a data wall (then scale width next).
"""
from lib.distributed_trainer import distributed_train
from lib.generic_ablation import get_config_grid
from experiments.realdet.learning_curve import create_learning_curve_v2_config


def create_configs():
    return get_config_grid(
        create_learning_curve_v2_config,
        dict(
            n=[4],
            n_train=[128000, 256000, 512000, 1024000],
            width=[512],
            depth=[3],
            lr=[1e-3],
            weight_decay=[1e-2],
            seed=[0, 1],
        ),
    )


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
