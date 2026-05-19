# uv run python run_slurm.py experiments/weather/evaluate_equivariance.py \
#     experiments/weather/persisted_configs/equivariant_ds/pear_equiv_ds.py \
#     experiments/weather/persisted_configs/equivariant_ds/pear_isolatitude_ring_shift_equiv_ds.py \
#     experiments/weather/persisted_configs/equivariant_ds/swin_equiv_ds_adjusted_grad_clip.py \
#     experiments/weather/persisted_configs/equivariant_ds/simple_conv_pear_isolatitude_ds.py \
#     experiments/weather/persisted_configs/equivariant_ds/simple_conv_pear_isolatitude_large_kernel_ds.py \
#     experiments/weather/persisted_configs/equivariant_ds/cvt_conv_attn_isolatitude_ds_grad_clip_v2_complex_scheduler.py \
#     --run-name weather_equivariance_thesis_evaluation \
#     --run-local \
#     --device cuda \
#     --labels PEAR,PEAR_Isolat,PEAR_Equiv,PEAR_Isolat+Conv,PEAR_Isolat+Conv+Larger_Kernel,PEAR_Isolat+ConvNext\
#     --max-batches 40 \
#     --epochs 50,100,150,200

# uv run python run_slurm.py experiments/weather/evaluate_equivariance.py \
#     experiments/weather/persisted_configs/equivariant_ds/pear_isolatitude_ring_shift_equiv_ds.py \
#     experiments/weather/persisted_configs/equivariant_ds/cvt_isolatitude_ds_grad_clip.py \
#     experiments/weather/persisted_configs/equivariant_ds/cvt_conv_attn_isolatitude_ds_grad_clip_v2.py \
#     --run-name conv_embedding \
#     --epochs 20,50,80,100,120,150,180,200
# #    --epochs 25,50,75,100,125,150,175,200

# uv run python run_slurm.py experiments/weather/evaluate_equivariance.py \
#     experiments/weather/persisted_configs/pear.py \
#     experiments/weather/persisted_configs/equivariant_ds_24h_pred/pear_equiv_ds_24h_pred.py \
#     --run-name weather_equivariance_24h_pred_std_new \
#     --run-local \
#     --device cuda \
#     --labels PEAR_24h,PEAR_2h\
#     --untrained-config experiments/weather/persisted_configs/pear.py \
#     --untrained-epoch 0 \
#     --max-batches 40 \
#     --epochs 100,200 \
#     --no-cache

uv run python run_slurm.py experiments/weather/evaluate_equivariance.py \
    experiments/weather/persisted_configs/pear.py \
    experiments/weather/persisted_configs/pear_conv_embeddings.py \
    --run-name weather_equivariance_conv_embedding \
    --run-local \
    --device cuda \
    --labels PEAR,PEAR_Conv_Embedding\
    --max-batches 40 \
    --epochs 0,100 \
    --no-cache