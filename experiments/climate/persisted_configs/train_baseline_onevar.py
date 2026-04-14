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

# Not included now
# CAMS-CSM1-0/r1i1p1f1
# CMIP6/CESM2/r4i1p1f1
# CMIP6/CMCC-CM2-SR5/r1i1p1f1
#("EC-Earth3-Veg", "r1i1p1f1"),
# CMIP6/CMCC-ESM2/r1i1p1f1
# CMIP6/CNRM-CM6-1-HR/r1i1p1f2
# CMIP6/EC-Earth3/r1i1p1f1
# CMIP6/EC-Earth3-Veg/r1i1p1f1
# CMIP6/EC-Earth3-Veg-LR/r1i1p1f1
# CMIP6/FGOALS-f3-L/r1i1p1f1
# CMIP6/GFDL-ESM4/r1i1p1f1
# CMIP6/INM-CM4-8/r1i1p1f1
# CMIP6/INM-CM5-0/r1i1p1f1
# CMIP6/MPI-ESM1-2-HR/r1i1p1f1
# CMIP6/MRI-ESM2-0/r1i1p1f1
# CMIP6/NorESM2-LM/r1i1p1f1
# CMIP6/NorESM2-LM/r2i1p1f1
# CMIP6/NorESM2-LM/r3i1p1f1
# CMIP6/NorESM2-MM/r1i1p1f1
# CMIP6/TaiESM1/r1i1p1f1



def create_config(ensemble_id, epoch=200, batch_size=12,):
    loss = torch.nn.MSELoss()
    
    model, ensemble = CLIMATE_MODELS[ensemble_id]
    print(model, ensemble)

    def loss_fn(output, batch):
        return loss(output["logits_output"], batch["target"])

    # params same for train and val
    data_cfg_common = dict(
            nside=NSIDE,
            climate_model=model,
            ensemble=ensemble,
            output_vars=["tas"],
            scenarios=["ssp126", "ssp370", "ssp585"],
            seq_len=1,
            seq_to_seq=True,
            normalized=True,
            cache=True,
            val_fraction=0.1,
            random_seed=7,
            fire_type=get_fire_type(model),
        )
    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=SwinHPClimatesetConfig(
            base_pix=12,
            nside=NSIDE,
            dev_mode=False,
            depths=[2, 6, 6, 2],
            num_heads=[6, 12, 12, 6],
            embed_dims=[192 // 4, 384 // 4, 384 // 4, 192 // 4],
            window_size=[1, 64],
            use_cos_attn=False,
            use_v2_norm_placement=True,
            drop_rate=0,
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
                lr=2e-4
            ),
        ),
        batch_size=batch_size,
        ensemble_id=ensemble_id,
        _version=6,
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


if __name__ == "__main__":
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0").strip()
    variant_idx = int(task_id) if task_id else 0
    print(f"SLURM_ARRAY_TASK_ID = {variant_idx}")

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)

    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)
    
    print("Starting distributed training...")
    config = create_config(ensemble_id=variant_idx, epoch=200)
    request_train_run(config)
    distributed_train([config])
    exit(0)
    
    # ablation test
    # configs = generic_ablation(
    #     create_config,
    #     dict(
    #         ensemble_id=[variant_idx],
    #         batch_size=[24, 48, 96],
    #     ),
    # )
    # distributed_train(configs)

    # ensemble_config = create_ensemble_config(
    #    lambda eid: create_config(eid, dataset_years=dataset_years), 1
    # )

    # if not is_ensemble_serialized(ensemble_config):
    # request_ensemble(config)



