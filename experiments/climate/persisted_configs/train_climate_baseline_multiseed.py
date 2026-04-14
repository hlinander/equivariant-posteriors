#!/usr/bin/env python
"""
Multi-seed training config for SwinHP on the ClimateSet HEALPix dataset.

Based on train_climate_baseline.py but separates the random seed (ensemble_id)
from the climate model index (climate_model_idx) so the same architecture
can be trained with multiple seeds for statistical comparisons.

Usage:
    # Single climate model, 3 seeds (SLURM array 0-2):
    N_SEEDS=3 CLIMATE_MODEL_IDX=12 python run_slurm_sweep.py \
        experiments/climate/persisted_configs/train_climate_baseline_multiseed_sweep.py
"""

import os
import torch

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.train_dataclasses import TrainEval
from lib.metric import create_metric
from lib.train_distributed import request_train_run
import lib.data_factory as data_factory
import lib.model_factory as model_factory
from lib.distributed_trainer import distributed_train

from experiments.climate.data.climateset_data_hp import ClimatesetHPConfig
from experiments.climate.data.climateset_data_hp import ClimatesetDataHP
from experiments.climate.data.climateset_data_hp import get_fire_type
from experiments.climate.models.swin_hp_climateset import SwinHPClimatesetConfig
from experiments.climate.models.swin_hp_climateset import SwinHPClimateset

NSIDE = 32
CLIMATE_MODELS = [
    ("AWI-CM-1-1-MR", "r1i1p1f1"),
    ("BCC-CSM2-MR",   "r1i1p1f1"),
    ("CAS-ESM2-0",    "r3i1p1f1"),
    ("CNRM-CM6-1-HR", "r1i1p1f2"),
    ("EC-Earth3",     "r1i1p1f1"),
    ("EC-Earth3-Veg-LR", "r1i1p1f1"),
    ("FGOALS-f3-L",   "r1i1p1f1"),
    ("GFDL-ESM4",     "r1i1p1f1"),
    ("INM-CM4-8",     "r1i1p1f1"),
    ("INM-CM5-0",     "r1i1p1f1"),
    ("MPI-ESM1-2-HR", "r1i1p1f1"),
    ("MRI-ESM2-0",    "r1i1p1f1"),
    ("NorESM2-LM",    "r1i1p1f1"),
    ("NorESM2-MM",    "r1i1p1f1"),
    ("TaiESM1",       "r1i1p1f1"),
]


def create_config(
    ensemble_id,
    epoch=200,
    batch_size=12,
    climate_model_idx=0,
    lr=2e-4 ,
    embed_dims=[192, 384, 384, 192],
    drop_rate=0.0,
    ):
    """Create a training config for a specific climate model and seed.

    Parameters
    ----------
    ensemble_id : int
        Random seed index (0, 1, 2, ...).  Controls torch.manual_seed
        (via TrainConfig.ensemble_id) and the train/val data split
        (via random_seed = ensemble_id + 1).
    climate_model_idx : int
        Index into CLIMATE_MODELS selecting which GCM to train on.
    """
    model_name, ensemble = CLIMATE_MODELS[climate_model_idx]
    print(f"climate_model={model_name}, ensemble={ensemble}, seed={ensemble_id}")

    loss = torch.nn.MSELoss()

    def loss_fn(output, batch):
        return loss(output["logits_output"], batch["target"])

    # Each seed gets a different train/val split and model initialisation
    random_seed  = ensemble_id + 1
    val_fraction = 0.1
    seq_len      = 1
    seq_to_seq   = True
    normalized   = True

    data_cfg_common = dict(
        nside=NSIDE,
        climate_model=model_name,
        ensemble=ensemble,
        scenarios=["ssp126", "ssp370", "ssp585"],
        val_fraction=val_fraction,
        random_seed=random_seed,
        seq_len=seq_len,
        seq_to_seq=seq_to_seq,
        normalized=normalized,
        cache=True,
        fire_type=get_fire_type(model_name),
    )

    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=SwinHPClimatesetConfig(
            base_pix=12,
            nside=NSIDE,
            dev_mode=False,
            depths=[2, 6, 6, 2],
            num_heads=[6, 12, 12, 6],
            embed_dims=embed_dims,
            window_size=[1, 64],
            use_cos_attn=False,
            use_v2_norm_placement=True,
            drop_rate=0, # potential increase
            attn_drop_rate=0.0,
            drop_path_rate=0,
            rel_pos_bias="single",
            shift_size=4,
            shift_strategy="ring_shift",
            ape=False,
            patch_size=16,
        ),
        train_data_config=ClimatesetHPConfig(
            **data_cfg_common,
            split="train",
        ),
        val_data_config=ClimatesetHPConfig(
            **data_cfg_common,
            split="val",
        ),
        loss=loss_fn,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(
                weight_decay=3e-6,
                lr=lr,
            ),
        ),
        batch_size=batch_size,
        ensemble_id=ensemble_id,
        _version=10,
    )

    train_eval = TrainEval(
        train_metrics=[create_metric(loss_fn)],
        validation_metrics=[create_metric(loss_fn)],
        log_gradient_norm=True,
    )

    train_run = TrainRun(
        project="climate",
        compute_config=ComputeConfig(),
        train_config=train_config,
        train_eval=train_eval,
        epochs=epoch,
        save_nth_epoch=1,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=10,
        validate_nth_epoch=5,
        visualize_terminal=False,
    )
    return train_run


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0").strip()
    variant_idx = int(task_id) if task_id else 0

    N_SEEDS = int(os.environ.get("N_SEEDS", "10"))
    climate_model_idx = int(
        os.environ.get("CLIMATE_MODEL_IDX", str(variant_idx // N_SEEDS))
    )
    seed_idx = variant_idx % N_SEEDS

    print(f"SLURM_ARRAY_TASK_ID={variant_idx}, "
          f"climate_model_idx={climate_model_idx}, seed_idx={seed_idx}")

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)

    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)

    print("Starting distributed training...")
    config = create_config(
        ensemble_id=seed_idx, epoch=200, climate_model_idx=climate_model_idx
    )
    request_train_run(config)
    distributed_train([config])
