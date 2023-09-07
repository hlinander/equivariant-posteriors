#!/usr/bin/env bash
export PYTHONPATH=$(pwd)
export TORCH_DEVICE="cpu"
python $@
