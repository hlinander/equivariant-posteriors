#!/usr/bin/env bash
set -x
export PYTHONPATH=$(pwd)
ulimit -n 64000
torchrun --standalone --nnodes=1 --nproc_per_node=1 $@
