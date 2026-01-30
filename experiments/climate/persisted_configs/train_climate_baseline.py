#!/usr/bin/env python
import torch
import numpy as np
from pathlib import Path
import tqdm
import json
from typing import List
import math
import matplotlib.pyplot as plt

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.train_dataclasses import TrainEval
from lib.metric import create_metric


from lib.regression_metrics import create_regression_metrics

from lib.ddp import ddp_setup

# from lib.ensemble import create_ensemble_config
# from lib.ensemble import create_ensemble

# from lib.ensemble import request_ensemble
# from lib.ensemble import symlink_checkpoint_files
from lib.files import prepare_results

# from lib.render_psql import add_artifact, add_parameter, has_artifact
from lib.render_duck import insert_artifact, insert_model_parameter
from lib.serialization import serialize_human
from lib.generic_ablation import generic_ablation
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


from experiments.climate.climateset_data import ClimatesetHPConfig
from experiments.climate.climateset_data import ClimatesetDataHP
from experiments.climate.models.swin_hp_climateset import SwinHPClimatesetConfig
from experiments.climate.models.swin_hp_climateset import SwinHPClimateset


NSIDE = 32


# def create_config(ensemble_id, epoch=200, dataset_years=10):
def create_config(ensemble_id, epoch=200):
    loss = torch.nn.L1Loss()

    def reg_loss(output, batch): # TODO: Change to something appropriate for climate
        # return loss(output["logits_output"], batch["target"]) + 0.25 * loss(
        #     output["logits_output"], batch["target"]
        # )
        return loss(output["logits_output"], batch["target"])
    
    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=SwinHPClimatesetConfig(
            base_pix=12,
            nside=NSIDE,
            dev_mode=False,
            depths=[2, 6, 6, 2],
            num_heads=[6, 12, 12, 6],
            embed_dims=[192 // 4, 384 // 4, 384 // 4, 192 // 4],
            window_size=[1, 64],  # int(32 * (NSIDE / 256)),
            use_cos_attn=False,
            use_v2_norm_placement=True,
            drop_rate=0,  # ,0.1,
            attn_drop_rate=0.0,  # ,0.1,
            drop_path_rate=0,
            rel_pos_bias="single",
            shift_size=4,  # int(16 * (NSIDE / 256)),
            shift_strategy="ring_shift",
            ape=False,
            patch_size=16,
        ),
        train_data_config=ClimatesetHPConfig(nside=NSIDE, years="2015-2100"),
        val_data_config=None,  # DataHPConfig(nside=NSIDE),
        loss=reg_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
        ),
        batch_size=12, # kanske ska ökas?
        ensemble_id=ensemble_id,
        _version=17, # TODO: AM I even using versioning here?
    )

    train_eval = TrainEval(
        train_metrics=[create_metric(reg_loss)],
        validation_metrics=[],
        log_gradient_norm=True,
    )  # create_regression_metrics(torch.nn.functional.l1_loss, None)

    train_run = TrainRun( # TODO: Change
        project="climate",
        compute_config=ComputeConfig(),
        train_config=train_config,
        train_eval=train_eval,
        epochs=epoch,
        save_nth_epoch=1,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=10,
        validate_nth_epoch=20,
        visualize_terminal=False,
        notes=dict(shift="fixed: ring shift uses shift_size instead of window/2"),
    )
    return train_run


if __name__ == "__main__":
    #ddp_setup() should this be here, what does it do
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)

    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)
    
    print("Starting distributed training...")
    config = create_config(ensemble_id=0)
    request_train_run(config)
    distributed_train([config])
    exit(0)
    
    # ablation test for later 
    # configs = generic_ablation(
    #     create_config,
    #     dict(
    #         ensemble_id=[0],
    #         attn_drop_rate=[0.0, 0.1],
    #     ),
    # )
    # distributed_train(config)

    # ensemble_config = create_ensemble_config(
    #    lambda eid: create_config(eid, dataset_years=dataset_years), 1
    # )

    # if not is_ensemble_serialized(ensemble_config):
    # request_ensemble(config)



