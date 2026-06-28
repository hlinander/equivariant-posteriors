"""
Sweep file for training SwinHP with multiple seeds on a single climate model.

Usage:
    # 3 seeds on NorESM2-LM (climate model index 12):
    CLIMATE_MODEL_IDX=12 N_SEEDS=3 python run_slurm_sweep.py \
        experiments/climate/persisted_configs/train_climate_pear_multiseed_sweep.py
"""

import os
from lib.generic_ablation import get_config_grid

from experiments.climate.persisted_configs.final_climate_pear_runs.train_climate_pear_multiseed import (
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
N_MODELS = int(os.environ.get("N_MODELS", "15"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "250"))


def create_configs():
    return get_config_grid(
        lambda **x: dict(**x),
        dict(
            seed=list(range(N_SEEDS)),
            climate_model_idx=list(range(N_MODELS)),
        ),
    )


def run(config):
    seed = config["seed"]
    climate_model_idx = config["climate_model_idx"]
    print(f"Training climate_model_idx={climate_model_idx}, seed={seed}")

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)

    train_run = create_config(
        ensemble_id=seed, epoch=N_EPOCHS, climate_model_idx=climate_model_idx,
    )
    request_train_run(train_run)
    distributed_train([train_run])

