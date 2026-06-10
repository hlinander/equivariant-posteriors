#!/usr/bin/env python
"""Sweep: TransformerEncoder with stability fixes (pre-LN, sqrt(embed_d)
embedding scale, no weight decay on LayerNorm/bias params, gradient clipping)
on the p=13 n=2 determinant task. Same grid as p13_transformer_encoder.py so
the two sweeps compare the fixes in isolation.

Separate sweep file so finite_field_det.create_configs() stays untouched for
in-flight SLURM array tasks.
"""
from lib.distributed_trainer import distributed_train
from experiments.grokking.finite_field_det import (
    p13_transformer_encoder_stable_configs,
)


def create_configs():
    return p13_transformer_encoder_stable_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
