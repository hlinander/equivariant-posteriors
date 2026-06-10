#!/usr/bin/env python
"""Sweep: self-attention-only transformer (TransformerEncoder) on the p=13 n=2
determinant task, mirroring the cross-attending Transformer grid in
finite_field_det.p13_sweep_configs for comparison.

Separate sweep file so finite_field_det.create_configs() stays untouched for
in-flight SLURM array tasks.
"""
from lib.distributed_trainer import distributed_train
from experiments.grokking.finite_field_det import p13_transformer_encoder_configs


def create_configs():
    return p13_transformer_encoder_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
