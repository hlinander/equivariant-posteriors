#!/usr/bin/env bash
export PYTHONPATH=$(pwd)
torchrun --standalone --nnodes=1 --nproc_per_node=1 $@
