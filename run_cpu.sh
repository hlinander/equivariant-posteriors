#!/bin/sh
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TORCH_DEVICE="cpu"
python $@
