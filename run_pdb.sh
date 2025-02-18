#!/bin/sh
export PYTHONPATH=$(pwd):$PYTHONPATH
export TORCH_DEVICE="cuda:0"
export PYTHONBREAKPOINT=ipdb.set_trace
python -m ipdb $@
