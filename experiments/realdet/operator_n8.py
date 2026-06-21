#!/usr/bin/env python
"""Sweep: teacher-forced matrix-operator transformer at N=8 -- a size where
direct determinant regression collapses (MLP A^{-T} sensitivity gone by N=5).
14 elimination ops, 200k data, 2 seeds.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.operator_transformer import n8_configs


def create_configs():
    return n8_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
