#!/usr/bin/env python
"""Sweep: row-token transformer step vs MLP step (stage 5). Same rollout
(partial pivot, log-output multiplier, anneal teacher, one refinement sweep);
only the step architecture differs. Tests whether attention's clean entry/row
selection breaks the MLP's per-step multiplier-precision floor (mult_loss ~0.19
at N=4) and lets refinement contract. N in {3,4,5}, 2 seeds.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.elimination import arch_configs


def create_configs():
    return arch_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
