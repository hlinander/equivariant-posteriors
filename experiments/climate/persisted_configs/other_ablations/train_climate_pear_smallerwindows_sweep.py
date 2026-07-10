"""
Sweep file for training SwinHP (smaller windows) over all climate models and multiple seeds.

Usage:
    # All 15 models, 5 seeds (75 SLURM array jobs):
    N_SEEDS=5 python run_slurm_sweep.py \
        experiments/climate/persisted_configs/train_climate_pear_smallerwindows_sweep.py

    # Dry run to inspect the batch script:
    N_SEEDS=5 python run_slurm_sweep.py --dry-run \
        experiments/climate/persisted_configs/train_climate_pear_smallerwindows_sweep.py

    # Run locally (sequential, no SLURM):
    N_SEEDS=5 python run_slurm_sweep.py --run-local \
        experiments/climate/persisted_configs/train_climate_pear_smallerwindows_sweep.py

    # Single model (e.g. NorESM2-LM, index 12), 3 seeds:
    N_SEEDS=3 CLIMATE_MODEL_IDX=12 python run_slurm_sweep.py \
        experiments/climate/persisted_configs/train_climate_pear_smallerwindows_sweep.py
"""

import os
from lib.generic_ablation import get_config_grid

from experiments.climate.persisted_configs.other_ablations.train_climate_pear_smallerwindows import (
    create_config,
    ClimatesetHPConfig,
    ClimatesetDataHP,
    SwinHPClimatesetConfig,
    SwinHPClimateset,
    CLIMATE_MODELS,
)
from experiments.climate.evaluation.evaluate_climate_hp import evaluate_climate
from lib.train_distributed import request_train_run
from lib.distributed_trainer import distributed_train
import lib.data_factory as data_factory
import lib.model_factory as model_factory

N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "250"))
# Set CLIMATE_MODEL_IDX to restrict the sweep to a single model; omit to sweep all.
_CLIMATE_MODEL_IDX_ENV = os.environ.get("CLIMATE_MODEL_IDX")
_MODEL_INDICES = (
    [int(_CLIMATE_MODEL_IDX_ENV)]
    if _CLIMATE_MODEL_IDX_ENV is not None
    else list(range(len(CLIMATE_MODELS)))
)


def create_configs():
    return get_config_grid(
        lambda **x: dict(**x),
        dict(
            climate_model_idx=_MODEL_INDICES,
            seed=list(range(N_SEEDS)),
        ),
    )


def _make_epochs(max_epoch, step=10):
    return list(range(0, max_epoch + 1, step))


def run(config):
    climate_model_idx = config["climate_model_idx"]
    seed = config["seed"]
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

    curried = lambda ensemble_id, **kw: create_config(
        epoch=N_EPOCHS,
        ensemble_id=ensemble_id,
        climate_model_idx=climate_model_idx,
        **kw,
    )

    model_name = train_run.train_config.train_data_config.climate_model
    epochs = _make_epochs(N_EPOCHS)
    print(f"=== Evaluating {model_name} (seed={seed}), epochs={epochs} ===")

    best_epoch, best_rmse = None, float("inf")
    for epoch in epochs:
        print(f"[{model_name} seed={seed}] epoch {epoch}")
        rmse = evaluate_climate(curried, epoch, variant_idx=seed)
        if rmse is not None and rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch

    if best_epoch is not None:
        print(f"=== BEST [{model_name} seed={seed}]: epoch {best_epoch}, RMSE {best_rmse:.6f} ===")
