#!/usr/bin/env python
"""
Training config for the UNet baseline on the ClimateSet grid dataset.

Follows the same structure as train_climate_baseline.py so that adapting
other baselines from models_climatesetrepo/baselines.py is a matter of:
  1. Creating a new adapter in models_climatesetrepo/ (see unet_adapter.py)
  2. Copying this file and swapping the Config / Model imports below.
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

# ---- Dataset (grid-based, no HEALPix) ------------------------------------
from experiments.climate.climateset_data_no_hp import ClimatesetConfig
from experiments.climate.climateset_data_no_hp import ClimatesetData

# ---- Model (UNet adapter — no emulator/Lightning dependency) --------------
from experiments.climate.models_climatesetrepo.adapted_models.cnn_lstm import (
    CNNLSTMConfig,
    CNNLSTM_ClimateBench,
)

# ---------------------------------------------------------------------------
# Climate model roster (same as train_climate_baseline.py for easy comparison)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def create_config(ensemble_id, epoch=200, batch_size=4):
    loss = torch.nn.MSELoss()

    model_name, ensemble = CLIMATE_MODELS[ensemble_id]
    print(model_name, ensemble)

    def loss_fn(output, batch):
        return loss(output["logits_output"], batch["target"])

    # Shared train / val dataset parameters
    random_seed   = 7
    val_fraction  = 0.1
    seq_len       = 12
    seq_to_seq    = True
    normalized    = True

    data_cfg_common = dict(
        climate_model=model_name,
        ensemble=ensemble,
        scenarios=["ssp126", "ssp370", "ssp585"],
        seq_len=seq_len,
        seq_to_seq=seq_to_seq,
        normalized=normalized,
        cache=True,
        val_fraction=val_fraction,
        random_seed=random_seed,
        channels_last=False,   # UNetAdapter expects channels-first
    )

    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=CNNLSTMConfig(
            num_conv_filters=20,
            lstm_hidden_size=25,
            num_lstm_layers=1,
            seq_to_seq=True,
            seq_len=12,
            dropout=0.0,
        ),
        train_data_config=ClimatesetConfig(
            **data_cfg_common,
            split="train",
        ),
        val_data_config=ClimatesetConfig(
            **data_cfg_common,
            split="val",
        ),
        loss=loss_fn,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(
                weight_decay=3e-6,
                lr=2e-4,
            ),
        ),
        batch_size=batch_size,
        ensemble_id=ensemble_id,
        _version=1,
    )

    train_eval = TrainEval(
        train_metrics=[create_metric(loss_fn)],
        validation_metrics=[create_metric(loss_fn)],
        log_gradient_norm=True,
    )

    train_run = TrainRun(
        project="climate_unet_baseline",
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
    print(f"SLURM_ARRAY_TASK_ID = {variant_idx}")

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetConfig, ClimatesetData)

    mf = model_factory.get_factory()
    mf.register(CNNLSTMConfig, CNNLSTM_ClimateBench)

    print("Starting distributed training...")
    config = create_config(ensemble_id=variant_idx, epoch=200)
    request_train_run(config)
    distributed_train([config])
