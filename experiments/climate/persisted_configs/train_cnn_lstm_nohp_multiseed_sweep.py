"""
Sweep file for training CNN-LSTM with multiple seeds across one or more climate models.

Usage:
    # 3 seeds on NorESM2-LM (climate model index 12):
    CLIMATE_MODEL_IDX=12 N_SEEDS=3 python run_slurm_sweep.py \
        experiments/climate/persisted_configs/train_cnn_lstm_nohp_multiseed_sweep.py

    # 3 seeds across 5 models starting at index 0:
    CLIMATE_MODEL_IDX=0 NUM_VARIANTS=5 N_SEEDS=3 python run_slurm_sweep.py \
        experiments/climate/persisted_configs/train_cnn_lstm_nohp_multiseed_sweep.py

    # Dry run to inspect the batch script:
    CLIMATE_MODEL_IDX=0 NUM_VARIANTS=5 N_SEEDS=3 python run_slurm_sweep.py --dry-run \
        experiments/climate/persisted_configs/train_cnn_lstm_nohp_multiseed_sweep.py
"""

import os
from lib.generic_ablation import get_config_grid

from experiments.climate.persisted_configs.train_cnn_lstm_nohp_multiseed import (
    create_config,
    ClimatesetConfig,
    ClimatesetData,
    CNNLSTMConfig,
    CNNLSTM_ClimateBench,
)
from lib.train_distributed import request_train_run
from lib.distributed_trainer import distributed_train
import lib.data_factory as data_factory
import lib.model_factory as model_factory

N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
CLIMATE_MODEL_START = int(os.environ.get("CLIMATE_MODEL_IDX", "0"))
NUM_VARIANTS = int(os.environ.get("NUM_VARIANTS", "15"))


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
    mf.register(CNNLSTMConfig, CNNLSTM_ClimateBench)

    train_run = create_config(
        ensemble_id=seed, epoch=400, climate_model_idx=climate_model_idx,
    )
    request_train_run(train_run)
    distributed_train([train_run])
