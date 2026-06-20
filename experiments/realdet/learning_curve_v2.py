#!/usr/bin/env python
"""Sweep: fixed-step (~30k optimizer steps) real-determinant learning curves,
N in {3..7} x n_train 4k..128k x 2 seeds, with the FTLE + A^{-T} Jacobian
metrics. Matched optimization across n_train (the fixed-epoch v1 under-trained
small n_train); _version=2 so these are fresh runs.

Separate sweep file so the fixed-epoch learning_curve_extended identity stays
untouched.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.learning_curve import learning_curve_v2_configs


def create_configs():
    return learning_curve_v2_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
