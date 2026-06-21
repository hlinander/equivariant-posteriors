#!/usr/bin/env python
"""Sweep: N=4/5 transformer-step confirmation at the trainable recipe (lr=1e-4,
final LayerNorm). The MLP step floored at mult_loss ~0.19 at N=4 and refinement
couldn't contract; the probe showed the transformer reaches ~0.01 at N=3. This
checks whether it also breaks the N>=4 wall. Partial pivot + log + anneal +
both + one refine sweep; N in {4,5}, 2 seeds.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.elimination import transformer_confirm_configs


def create_configs():
    return transformer_confirm_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
