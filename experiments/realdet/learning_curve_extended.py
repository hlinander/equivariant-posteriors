#!/usr/bin/env python
"""Sweep: extended real-determinant learning curves (N in {3..7}, n_train up to
128k) with the FTLE + A^{-T} Jacobian metrics. Resolves the high-N data-scaling
law that the slim sweep (n_train<=32k) left unresolved.

Separate sweep file so the slim learning_curve.create_configs() identity stays
untouched.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.learning_curve import learning_curve_extended_configs


def create_configs():
    return learning_curve_extended_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
