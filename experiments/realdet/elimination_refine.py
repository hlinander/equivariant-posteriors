#!/usr/bin/env python
"""Sweep: greedy residual refinement (stage 4 / test-time compute). Trains the
elimination rollout with one extra refinement sweep on top of the pivoted base
pass; the diagnostics hook logs free-rollout logdet_mae across an eval R-grid
(0,1,2,4,8 x base), giving the inference-compute scaling curve. Fixed winners
from earlier stages: partial pivot, log-output multiplier, anneal teacher, both
input. N in {3,4,5}, 2 seeds.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.elimination import refine_configs


def create_configs():
    return refine_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
