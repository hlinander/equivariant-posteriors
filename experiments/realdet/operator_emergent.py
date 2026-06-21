#!/usr/bin/env python
"""Sweep: emergent thinking for the matrix-operator transformer. Per-step
operator supervision (op_loss) anneals 1->0 while an end-state objective
(triangularize + match log|det|) on the model's OWN free rollout takes over.
Tests whether the model retains/discovers the row operations as imitation
vanishes. N=3, 2 seeds.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.operator_transformer import emergent_configs


def create_configs():
    return emergent_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
