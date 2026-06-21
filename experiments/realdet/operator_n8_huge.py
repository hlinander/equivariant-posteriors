#!/usr/bin/env python
"""Sweep: N=8 teacher-forced matrix-operator transformer with 10x data (2M) vs
operator_n8 (200k) -- does even more data shrink the long-rollout drift? 50
epochs (each 10x the steps), single seed.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.operator_transformer import n8_huge_configs


def create_configs():
    return n8_huge_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
