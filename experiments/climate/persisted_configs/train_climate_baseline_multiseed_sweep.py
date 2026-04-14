"""
Sweep file for training SwinHP with multiple seeds on a single climate model.

Usage:
    # 3 seeds on NorESM2-LM (climate model index 12):
    CLIMATE_MODEL_IDX=12 N_SEEDS=3 python run_slurm_sweep.py \
        experiments/climate/persisted_configs/train_climate_baseline_multiseed_sweep.py

    # Dry run to inspect the batch script:
    CLIMATE_MODEL_IDX=12 N_SEEDS=3 python run_slurm_sweep.py --dry-run \
        experiments/climate/persisted_configs/train_climate_baseline_multiseed_sweep.py

    # Run locally (sequential, no SLURM):
    CLIMATE_MODEL_IDX=12 N_SEEDS=3 python run_slurm_sweep.py --run-local \
        experiments/climate/persisted_configs/train_climate_baseline_multiseed_sweep.py
"""

import os
from lib.generic_ablation import get_config_grid

from experiments.climate.persisted_configs.train_climate_baseline_multiseed import (
    create_config,
    ClimatesetHPConfig,
    ClimatesetDataHP,
    SwinHPClimatesetConfig,
    SwinHPClimateset,
)
from lib.train_distributed import request_train_run
from lib.distributed_trainer import distributed_train
import lib.data_factory as data_factory
import lib.model_factory as model_factory

N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
CLIMATE_MODEL_IDX = int(os.environ.get("CLIMATE_MODEL_IDX", "0"))


def create_configs():
    return get_config_grid(
        lambda **x: dict(**x),
        dict(seed=list(range(N_SEEDS))),
    )


def run(config):
    seed = config["seed"]
    print(f"Training climate_model_idx={CLIMATE_MODEL_IDX}, seed={seed}")

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)

    train_run = create_config(
        ensemble_id=seed, epoch=200, climate_model_idx=CLIMATE_MODEL_IDX,
    )
    request_train_run(train_run)
    distributed_train([train_run])

