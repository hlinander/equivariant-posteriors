#!/usr/bin/env python
"""Sweep: elimination rollout with pivoting (stage 3). Fixed winners from earlier
stages -- log-output multiplier, anneal teacher forcing, both-input -- and
compares pivot none/partial/learned across N in {3,4,5}.

partial pivoting (oracle argmax|col|) bounds |c|<=1, removing the heavy-tailed
multiplier targets that grew with N and stalled the no-pivot runs at logdet_mae
~0.4 (N=3) / ~2.0 (N=4). learned pivoting adds a scorer supervised against the
partial-pivot argmax (pivot_loss). The test: does pivoting bring free-rollout
logdet_mae toward the oracle's 0 at N>=3, and does learned match partial?

Separate file so elimination.create_configs() stays untouched.
"""
from lib.distributed_trainer import distributed_train
from experiments.realdet.elimination import pivoting_configs


def create_configs():
    return pivoting_configs()


def run(config):
    distributed_train([config])


if __name__ == "__main__":
    distributed_train(create_configs())
