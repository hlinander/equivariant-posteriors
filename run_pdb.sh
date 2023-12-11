#!/bin/bash
export PYTHONPATH=$(pwd)
export TORCH_DEVICE="cpu"
python -m ipdb $@
