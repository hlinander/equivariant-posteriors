#!/usr/bin/env bash
set -x
export PYTHONPATH=$(pwd)
ulimit -n 64000
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 $@
