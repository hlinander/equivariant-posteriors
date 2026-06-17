"""
Sweep file for training SwinHPClimatesetTemporalAtnCausal with multiple seeds
across all 15 climate models, with post-training evaluation across checkpoints.

Usage:
    # All 15 models × 5 seeds (75 SLURM array tasks):
    python run_slurm_sweep.py \
        experiments/climate/persisted_configs/train_climate_pear_temporal_atn_causal_sweep_with_eval.py

    # Override N_SEEDS / N_MODELS / N_EPOCHS via env vars:
    N_SEEDS=3 N_MODELS=15 N_EPOCHS=400 python run_slurm_sweep.py \
        experiments/climate/persisted_configs/train_climate_pear_temporal_atn_causal_sweep_with_eval.py

    # Dry run to inspect the batch script:
    python run_slurm_sweep.py --dry-run \
        experiments/climate/persisted_configs/train_climate_pear_temporal_atn_causal_sweep_with_eval.py

    # Run locally (sequential, no SLURM):
    N_SEEDS=1 N_MODELS=1 python run_slurm_sweep.py --run-local \
        experiments/climate/persisted_configs/train_climate_pear_temporal_atn_causal_sweep_with_eval.py
"""

import os
from lib.generic_ablation import get_config_grid

from experiments.climate.persisted_configs.train_climate_pear_temporal_atn_causal_multiseed import (
    create_config,
    ClimatesetHPConfig,
    ClimatesetDataHP,
    SwinHPClimatesetTemporalAtnCausalConfig,
    SwinHPClimatesetTemporalAtnCausal,
)
from experiments.climate.evaluation.evaluate_climate_hp import evaluate_climate
from lib.train_distributed import request_train_run
from lib.distributed_trainer import distributed_train
import lib.data_factory as data_factory
import lib.model_factory as model_factory

N_SEEDS   = int(os.environ.get("N_SEEDS",   "5"))
N_MODELS  = int(os.environ.get("N_MODELS",  "15"))
N_EPOCHS  = int(os.environ.get("N_EPOCHS",  "250"))


def create_configs():
    return get_config_grid(
        lambda **x: dict(**x),
        dict(
            seed=list(range(N_SEEDS)),
            climate_model_idx=list(range(N_MODELS)),
        ),
    )


def _make_epochs(max_epoch, step=10):
    # Start from step, not 0: the epoch-0 checkpoint is saved before the first
    # forward pass, so with the lazy-init training model it has no layer weights.
    return list(range(step, max_epoch + 1, step))


def run(config):
    seed = config["seed"]
    climate_model_idx = config["climate_model_idx"]
    print(f"Training climate_model_idx={climate_model_idx}, seed={seed}")

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetTemporalAtnCausalConfig, SwinHPClimatesetTemporalAtnCausal)

    train_run = create_config(
        ensemble_id=seed, epoch=N_EPOCHS, climate_model_idx=climate_model_idx,
    )
    request_train_run(train_run)
    distributed_train([train_run])

    curried = lambda ensemble_id, **kw: create_config(
        ensemble_id=ensemble_id,
        epoch=N_EPOCHS,
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
        print(f"=== BEST [{model_name} seed={seed}]: "
              f"epoch {best_epoch}, RMSE {best_rmse:.6f} ===")
