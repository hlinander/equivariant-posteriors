#!/usr/bin/env python
"""
Training config for the seq-to-seq SwinHP model.

Differences from train_climate_baseline.py:
  - seq_len=12  → the dataloader returns a full year (12 months) per sample
  - Model: SwinHPClimatesetSeq  (shared spatial backbone + temporal mixing MLP)
  - batch_size=4  (each sample is 12× larger than the baseline's single timestep)
  - _version=3 to keep checkpoints independent from the baseline runs
"""

import torch
import os

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.train_dataclasses import TrainEval
from lib.metric import create_metric
from lib.regression_metrics import create_regression_metrics
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config, create_ensemble
from lib.files import prepare_results
from lib.render_duck import insert_artifact, insert_model_parameter
from lib.serialization import serialize_human
from lib.generic_ablation import generic_ablation
from lib.train_distributed import request_train_run

import lib.data_factory as data_factory
import lib.model_factory as model_factory

from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.distributed_trainer import distributed_train
from lib.serialization import deserialize_model, DeserializeConfig

from experiments.climate.climateset_data_hp import ClimatesetHPConfig, ClimatesetDataHP
from experiments.climate.models.swin_hp_climateset_seq import (
    SwinHPClimatesetSeqConfig,
    SwinHPClimatesetSeq,
)


NSIDE = 32
CLIMATE_MODELS = [
    ("AWI-CM-1-1-MR",   "r1i1p1f1"),
    ("BCC-CSM2-MR",     "r1i1p1f1"),
    ("CAS-ESM2-0",      "r3i1p1f1"),
    ("CNRM-CM6-1-HR",   "r1i1p1f2"),
    ("EC-Earth3",       "r1i1p1f1"),
    ("EC-Earth3-Veg-LR","r1i1p1f1"),
    ("FGOALS-f3-L",     "r1i1p1f1"),
    ("GFDL-ESM4",       "r1i1p1f1"),
    ("INM-CM4-8",       "r1i1p1f1"),
    ("INM-CM5-0",       "r1i1p1f1"),
    ("MPI-ESM1-2-HR",   "r1i1p1f1"),
    ("MRI-ESM2-0",      "r1i1p1f1"),
    ("NorESM2-LM",      "r1i1p1f1"),
    ("NorESM2-MM",      "r1i1p1f1"),
    ("TaiESM1",         "r1i1p1f1"),
]

SEQ_LEN = 12   # one full year of monthly data


def create_config(ensemble_id, epoch=300, batch_size=4):
    loss = torch.nn.MSELoss()

    model, ensemble = CLIMATE_MODELS[ensemble_id]
    print(model, ensemble)

    def loss_fn(output, batch):
        # output["logits_output"]: (B, T, C_out, N_pix)
        # batch["target"]:         (B, T, C_out, N_pix)
        return loss(output["logits_output"], batch["target"])

    train_config = TrainConfig(
        extra=dict(loss_variant="full", seq_len=SEQ_LEN),
        model_config=SwinHPClimatesetSeqConfig(
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
            nside=NSIDE,
            climate_model=model,
            ensemble=ensemble,
            scenarios=["ssp126", "ssp370", "ssp585"],
            split="train",
            val_fraction=0.1,
            random_seed=42,
            seq_len=SEQ_LEN,
            seq_to_seq=True,
            normalized=True,
            cache=True,
        ),
        val_data_config=ClimatesetHPConfig(
            nside=NSIDE,
            climate_model=model,
            ensemble=ensemble,
            scenarios=["ssp126", "ssp370", "ssp585"],
            split="val",
            val_fraction=0.1,
            random_seed=42,
            seq_len=SEQ_LEN,
            seq_to_seq=True,
            normalized=True,
            cache=True,
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
        _version=3,
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
    variant_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    print(f"SLURM_ARRAY_TASK_ID = {variant_idx}")

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)

    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetSeqConfig, SwinHPClimatesetSeq)

    print("Starting distributed training (seq model)...")
    config = create_config(ensemble_id=variant_idx, epoch=200)
    request_train_run(config)
    distributed_train([config])
    exit(0)
