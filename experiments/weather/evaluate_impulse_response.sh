uv run run_slurm.py experiments/weather/evaluate_impulse_response.py \
      experiments/weather/persisted_configs/equivariant_ds/convnext_pear_isolatitude_1block_3x3_ds.py \
      --epoch 0 \
      --impulse equator \
      --amplitude 5.0 \
      --lat-width 5.0