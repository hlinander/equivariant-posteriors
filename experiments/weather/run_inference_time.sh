#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --job-name=inference_time

export TORCH_DEVICE=cuda

for cfg_epoch in \
    "experiments/weather/persisted_configs/pear.py 200" \
    "experiments/weather/persisted_configs/pangu.py 300" \
    "experiments/weather/persisted_configs/pangu_large.py 250" \
    "experiments/weather/persisted_configs/pangu_physicsnemo.py 300" \
    "experiments/weather/persisted_configs/graphcast_physicsnemo.py 500" \
    "experiments/weather/persisted_configs/fengwu_physicsnemo.py 500"; do
    echo "=== $cfg_epoch ==="
    uv run python experiments/weather/inference_time.py $cfg_epoch
    echo
done
