#!/usr/bin/env python
"""Sweep: stable TransformerEncoder plus z-loss (logit-norm penalty) on the
p=13 n=2 determinant task. Same grid as p13_transformer_encoder_stable.py so
the two sweeps isolate the effect of capping logit growth on post-grok
stability and LayerNorm-gain norm growth.

Separate sweep file so finite_field_det.create_configs() stays untouched for
in-flight SLURM array tasks.
"""
from lib.distributed_trainer import distributed_train
from experiments.grokking.finite_field_det import (
    p13_transformer_encoder_zloss_configs,
)


def create_configs():
    return p13_transformer_encoder_zloss_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
