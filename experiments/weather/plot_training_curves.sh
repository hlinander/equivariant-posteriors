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



uv run python run_slurm.py experiments/weather/plot_training_curves.py \
    experiments/weather/persisted_configs/equivariant_ds/pear_equiv_ds.py \
    experiments/weather/persisted_configs/equivariant_ds/pear_isolatitude_ring_shift_equiv_ds.py \
    experiments/weather/persisted_configs/equivariant_ds/cvt_isolatitude_ds_grad_clip.py \
    experiments/weather/persisted_configs/equivariant_ds/cvt_conv_attn_isolatitude_ds_grad_clip_v2.py \
    experiments/weather/persisted_configs/equivariant_ds/cvt_conv_attn_isolatitude_ds_grad_clip_v2_complex_scheduler.py\
    experiments/weather/persisted_configs/equivariant_ds/simple_conv_pear_isolatitude_ds.py \
    experiments/weather/persisted_configs/equivariant_ds/swin_equiv_ds_adjusted_grad_clip.py \
    --compute-rmse --device cuda \
    --epochs 20,40,60,80,100,120,140,160,180,200,220,240,260,280,300 \
    --reduction-factor 0.1 --consecutive-samples 1 \
    --lead-time-days 1 --upper --tag cluster_computed \
    --out-dir experiments/weather/plots/validation/isolat_comparison


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