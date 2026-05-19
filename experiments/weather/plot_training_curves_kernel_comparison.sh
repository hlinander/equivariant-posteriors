#!/usr/bin/env bash
# uv run python run_slurm.py experiments/weather/plot_training_curves.py \
#     experiments/weather/persisted_configs/pear.py \
#     experiments/weather/persisted_configs/equivariant_ds/pear_equiv_ds.py \
#     experiments/weather/persisted_configs/equivariant_ds/swin_isolatitude_ds.py \
#     experiments/weather/persisted_configs/equivariant_ds/swin_isolatitude_ds_grad_clip.py \
#     experiments/weather/persisted_configs/equivariant_ds/swin_equiv_ds_ring_shift.py \
#     --compute-rmse --device cuda \
#     --epochs 20,40,60,80,100,120,140,160,180,200 \
#     --reduction-factor 0.1 --consecutive-samples 1 \
#     --lead-time-days 1 --upper --tag manual \
#     --out-dir experiments/weather/plots/validation/pear


    # experiments/weather/persisted_configs/equivariant_ds/conv_pear_isolatitude_1block_5x5_ds.py \
    # experiments/weather/persisted_configs/equivariant_ds/conv_pear_isolatitude_1block_5x5_ds_no_conv_embedding.py \
    # experiments/weather/persisted_configs/equivariant_ds/simple_conv_pear_isolatitude_large_kernel_ds.py\
    # experiments/weather/persisted_configs/equivariant_ds/simple_conv_pear_isolatitude_ds.py \

uv run python run_slurm.py experiments/weather/plot_training_curves.py \
    experiments/weather/persisted_configs/equivariant_ds/pear_equiv_ds.py\
    experiments/weather/persisted_configs/equivariant_ds/pear_isolatitude_ring_shift_equiv_ds.py \
    experiments/weather/persisted_configs/equivariant_ds/conv_pear_isolatitude_1block_3x3_ds.py \
    experiments/weather/persisted_configs/equivariant_ds/conv_pear_isolatitude_1block_3x3_ds_no_conv_embedding.py \
    experiments/weather/persisted_configs/equivariant_ds/convnext_pear_isolatitude_1block_3x3_ds.py \
    experiments/weather/persisted_configs/equivariant_ds/convnext_pear_isolatitude_1block_3x3_ds_no_conv_embedding.py \
    --compute-rmse --device cuda \
    --epochs 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200 \
    --reduction-factor 0.1 --consecutive-samples 1 \
    --lead-time-days 1 --upper --tag cluster_computed \
    --out-dir experiments/weather/plots/validation/kernel_comparison


# uv run python run_slurm.py experiments/weather/plot_training_curves.py \
#     experiments/weather/persisted_configs/equivariant_ds_24h_pred/pear_equiv_ds_24h_pred.py \
#     experiments/weather/persisted_configs/equivariant_ds_24h_pred/pear_isolatitude_ring_shift_equiv_ds_24h_pred.py \
#     experiments/weather/persisted_configs/pear.py \
#     experiments/weather/persisted_configs/pear_isolatitude.py \
#     --compute-rmse --device cuda \
#     --epochs 20,40,60,80,100,120,140,160,180,200 \
#     --reduction-factor 0.1 --consecutive-samples 1 \
#     --lead-time-days 1 --upper --tag cluster_computed \
#     --out-dir experiments/weather/plots/validation/24h_pred_comparison