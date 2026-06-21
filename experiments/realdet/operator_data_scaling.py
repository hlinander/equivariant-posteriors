#!/usr/bin/env python
"""Sweep: data-efficiency of the matrix-operator transformer at N=4,5 -- does
more training data lower op_loss and hence reduce free-rollout drift
(logdet_mae)? n_train in {25k,50k,100k,200k}, seed 0.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.operator_transformer import data_scaling_configs


def create_configs():
    return data_scaling_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
