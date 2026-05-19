#!/usr/bin/env bash

uv run python run_slurm.py experiments/weather/plot_training_curves.py \
    experiments/weather/persisted_configs/equivariant_ds_24h_pred/pear_equiv_ds_24h_pred.py \
    experiments/weather/persisted_configs/equivariant_ds_24h_pred/pear_isolatitude_ring_shift_equiv_ds_24h_pred.py \
    experiments/weather/persisted_configs/equivariant_ds_24h_pred/conv_pear_isolatitude_1block_3x3_ds_24h_pred.py \
    experiments/weather/persisted_configs/equivariant_ds_24h_pred/conv_pear_isolatitude_1block_3x3_ds_no_conv_embedding_24h_pred.py \
    experiments/weather/persisted_configs/equivariant_ds_24h_pred/conv_pear_isolatitude_1block_5x5_ds_24h_pred.py \
    experiments/weather/persisted_configs/equivariant_ds_24h_pred/conv_pear_isolatitude_1block_5x5_ds_no_conv_embedding_24h_pred.py \
    experiments/weather/persisted_configs/equivariant_ds_24h_pred/convnext_pear_isolatitude_1block_3x3_ds_24h_pred.py \
    experiments/weather/persisted_configs/equivariant_ds_24h_pred/convnext_pear_isolatitude_1block_3x3_ds_no_conv_embedding_24h_pred.py \
    --compute-rmse --device cuda \
    --epochs 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,200 \
    --reduction-factor 0.1 --consecutive-samples 1 \
    --lead-time-days 1 --upper --tag cluster_computed \
    --out-dir experiments/weather/plots/validation/24h_pred_comparison_cvt



    # experiments/weather/persisted_configs/equivariant_ds_24h_pred/cvt_isolatitude_ds_grad_clip_24h_pred.py \
    # experiments/weather/persisted_configs/equivariant_ds_24h_pred/cvt_conv_attn_isolatitude_ds_grad_clip_v2_cosine_24h_pred.py \
    # experiments/weather/persisted_configs/equivariant_ds_24h_pred/simple_conv_pear_isolatitude_ds_24h_pred.py \