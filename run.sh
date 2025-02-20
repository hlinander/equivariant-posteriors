#!/bin/sh
set -x
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONBREAKPOINT=ipdb.set_trace
export EP_TORCHRUN=1
ulimit -n 64000
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 $@
