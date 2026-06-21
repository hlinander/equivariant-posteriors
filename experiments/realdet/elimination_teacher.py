#!/usr/bin/env python
"""Sweep: elimination rollout with the division fix (log-output multiplier head)
and teacher forcing / scheduled sampling, with per-step multiplier supervision.

linear vs log output head x teacher_mode {on, anneal}, input_features=both,
N in {2,3,4}. The per-step mult_loss going to ~0 means the division is solved;
the free-rollout (eval) logdet_mae / lower_rms then isolate autoregressive drift.

Separate file so elimination.create_configs() (stage-1 free rollout) is untouched.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.elimination import teacher_configs


def create_configs():
    return teacher_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
