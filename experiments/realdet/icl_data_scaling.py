#!/usr/bin/env python
"""Data-scale study of the operator-algebra transformer's [ICL] sentence:
n_train in {25k,50k,100k,200k,400k} across all three function classes
(powerlaw/full/lowrank), seed 0. Does more data (more distinct latent W's)
lower in-context query error and sharpen difficulty-adaptive halting?
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.operator_icl import data_scaling_configs


def create_configs():
    return data_scaling_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
