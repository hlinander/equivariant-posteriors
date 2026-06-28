"""
Sweep file for training UNet (latfix) with multiple seeds across one or more
climate models, followed by automatic evaluation at every saved checkpoint.

Usage:
    # 3 seeds on NorESM2-LM (climate model index 12):
    CLIMATE_MODEL_IDX=12 N_SEEDS=3 python run_slurm_sweep.py \
        experiments/climate/persisted_configs/train_unet_nohp_multiseed_latfix_sweep.py

    # 3 seeds across 5 models starting at index 0:
    CLIMATE_MODEL_IDX=0 NUM_VARIANTS=5 N_SEEDS=3 python run_slurm_sweep.py \
        experiments/climate/persisted_configs/train_unet_nohp_multiseed_latfix_sweep.py
"""

import os
from lib.generic_ablation import get_config_grid

from experiments.climate.persisted_configs.final_climateset_baseline_runs.train_unet_nohp_multiseed_latfix import (
    create_config,
    ClimatesetConfig,
    ClimatesetData,
    UNetConfig,
    UNet,
)
from experiments.climate.evaluation.evaluate_climate_nohp import evaluate_climate
from lib.train_distributed import request_train_run
from lib.distributed_trainer import distributed_train
import lib.data_factory as data_factory
import lib.model_factory as model_factory

N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
CLIMATE_MODEL_START = int(os.environ.get("CLIMATE_MODEL_IDX", "0"))
NUM_VARIANTS = int(os.environ.get("NUM_VARIANTS", "15"))
MAX_EPOCH = 200
EVAL_STEP = 10  # matches keep_nth_epoch_checkpoints


def create_configs():
    return get_config_grid(
        lambda **x: dict(**x),
        dict(
            seed=list(range(N_SEEDS)),
            climate_model_idx=list(range(CLIMATE_MODEL_START, CLIMATE_MODEL_START + NUM_VARIANTS)),
        ),
    )


def run(config):
    seed = config["seed"]
    climate_model_idx = config["climate_model_idx"]
    print(f"Training climate_model_idx={climate_model_idx}, seed={seed}")

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetConfig, ClimatesetData)
    mf = model_factory.get_factory()
    mf.register(UNetConfig, UNet)

    train_run = create_config(
        ensemble_id=seed, epoch=MAX_EPOCH, climate_model_idx=climate_model_idx,
    )
    request_train_run(train_run)
    distributed_train([train_run])

    curried = lambda ensemble_id, **kw: create_config(
        ensemble_id=ensemble_id, epoch=MAX_EPOCH, climate_model_idx=climate_model_idx, **kw
    )

    model_name = train_run.train_config.train_data_config.climate_model
    epochs = list(range(0, MAX_EPOCH + 1, EVAL_STEP))
    print(f"=== Evaluating {model_name} (seed={seed}), epochs={epochs} ===")

    best_epoch, best_rmse = None, float("inf")
    for epoch in epochs:
        print(f"[{model_name} seed={seed}] epoch {epoch}")
        rmse = evaluate_climate(curried, epoch, variant_idx=seed)
        if rmse is not None and rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch

    if best_epoch is not None:
        print(f"=== BEST [{model_name} seed={seed}]: epoch {best_epoch}, RMSE {best_rmse:.6f} ===")
