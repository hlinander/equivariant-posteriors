#!/usr/bin/env python
"""Sweep: transformer-step trainability probe (stage 5b). The transformer step
was flat-untrained with the MLP recipe (lr=1e-3, no warmup); unrolling the
encoder over the rollout makes it effectively deep. Probe lower LR x shallower
depth, with a final LayerNorm now added. N=3 only (MLP reaches mult_loss 0.048
there); 4 configs.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.elimination import transformer_probe_configs


def create_configs():
    return transformer_probe_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
