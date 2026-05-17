#!/usr/bin/env python
import torch
import numpy as np
from pathlib import Path
import tqdm
import json
from typing import List
import math
import matplotlib.pyplot as plt
import os

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.train_dataclasses import TrainEval
from lib.metric import create_metric


from lib.regression_metrics import create_regression_metrics

from lib.ddp import ddp_setup

from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble

# from lib.ensemble import request_ensemble
# from lib.ensemble import symlink_checkpoint_files
from lib.files import prepare_results

# from lib.render_psql import add_artifact, add_parameter, has_artifact
from lib.render_duck import insert_artifact, insert_model_parameter
from lib.serialization import serialize_human
#from lib.generic_ablation import generic_ablation
from lib.train_distributed import request_train_run

# from lib.data_factory import register_dataset, get_factory
import lib.data_factory as data_factory
import lib.model_factory as model_factory

# from lib.models.mlp import MLPConfig
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy
from lib.distributed_trainer import distributed_train
from lib.serialization import (
    deserialize_model,
    DeserializeConfig,
)


from experiments.climate.data.climateset_data_hp import ClimatesetHPConfig
from experiments.climate.data.climateset_data_hp import ClimatesetDataHP
from experiments.climate.data.climateset_data_hp import get_fire_type
from experiments.climate.models.swin_hp_climateset import SwinHPClimatesetConfig
from experiments.climate.models.swin_hp_climateset import SwinHPClimateset
from experiments.climate.models.GRU_wrapper import GRUTemporalWrapperConfig, GRUTemporalWrapper

NSIDE = 32
CLIMATE_MODELS = [
    ("AWI-CM-1-1-MR", "r1i1p1f1"),
    ("BCC-CSM2-MR", "r1i1p1f1"),
    ("CAS-ESM2-0",   "r3i1p1f1"),
    ("CNRM-CM6-1-HR", "r1i1p1f2"),
    ("EC-Earth3",    "r1i1p1f1"),
    ("EC-Earth3-Veg-LR", "r1i1p1f1"),
    ("FGOALS-f3-L",  "r1i1p1f1"),
    ("GFDL-ESM4",    "r1i1p1f1"),
    ("INM-CM4-8",    "r1i1p1f1"),
    ("INM-CM5-0",    "r1i1p1f1"),
    ("MPI-ESM1-2-HR", "r1i1p1f1"),
    ("MRI-ESM2-0",   "r1i1p1f1"),
    ("NorESM2-LM",   "r1i1p1f1"), # Note: NorESM2-LM has multiple ensemble members
    ("NorESM2-MM",   "r1i1p1f1"),
    ("TaiESM1",      "r1i1p1f1"),
]


def create_config(ensemble_id, epoch=400, batch_size=12, climate_model_idx=0, lr=2e-4):
    loss = torch.nn.MSELoss()
    model_name, ensemble = CLIMATE_MODELS[climate_model_idx]  # <-- use idx, not ensemble_id

    def loss_fn(output, batch):
        target = batch["target"]
        pred = output["logits_output"]
        return loss(pred, target)

    random_seed = ensemble_id + 1  # <-- each seed gets different split
    val_fraction = 0.1
    seq_len = 12
    seq_to_seq = False
    normalized = True

    backbone_config = SwinHPClimatesetConfig(
        # ... same as before
    )

    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=GRUTemporalWrapperConfig(
            backbone_config=backbone_config,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
        ),
        train_data_config=ClimatesetHPConfig(
            nside=NSIDE,
            climate_model=model_name,
            ensemble=ensemble,
            scenarios=["ssp126", "ssp370", "ssp585"],
            split="train",
            val_fraction=val_fraction,
            random_seed=random_seed,  # <-- uses ensemble_id+1
            seq_len=seq_len,
            seq_to_seq=seq_to_seq,
            normalized=normalized,
            cache=True,
            fire_type=get_fire_type(model_name),
        ),
        val_data_config=ClimatesetHPConfig(
            nside=NSIDE,
            climate_model=model_name,
            ensemble=ensemble,
            scenarios=["ssp126", "ssp370", "ssp585"],
            split="val",
            val_fraction=val_fraction,
            random_seed=random_seed,
            seq_len=seq_len,
            seq_to_seq=seq_to_seq,
            normalized=normalized,
            cache=True,
            fire_type=get_fire_type(model_name),
        ),
        loss=loss_fn,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=lr),
        ),
        batch_size=batch_size,
        ensemble_id=ensemble_id,
        gradient_clipping=1.0,
        _version=10,  # <-- bump to avoid checkpoint collisions
    )
    # ... rest unchanged
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

if __name__ == "__main__":
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0").strip()
    variant_idx = int(task_id) if task_id else 0
    print(f"SLURM_ARRAY_TASK_ID = {variant_idx}")

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)

    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)
    mf.register(GRUTemporalWrapperConfig, GRUTemporalWrapper)

    print("Starting distributed training...")
    config = create_config(ensemble_id=variant_idx, epoch=1000)
    request_train_run(config)
    distributed_train([config])
    exit(0)
    