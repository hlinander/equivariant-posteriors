#!/usr/bin/env python
"""
Training config for the ClimaX baseline on the ClimateSet grid dataset.

Follows the same structure as train_cnn_lstm_nohp_newlossfn.py so that adapting
other baselines from models_climatesetrepo/baselines.py is a matter of:
  1. Creating a new adapter in models_climatesetrepo/ (see climax/climax_module.py)
  2. Copying this file and swapping the Config / Model imports below.
"""

import os
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig, SchedulerConfig
from lib.train_dataclasses import TrainEval
from lib.metric import create_metric
from lib.train_distributed import request_train_run
import lib.data_factory as data_factory
import lib.model_factory as model_factory
from lib.distributed_trainer import distributed_train

# ---- LR Scheduler (ported from emulator.src.utils.lr_scheduler, without pytorch_lightning) --
class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup then cosine annealing, matching ClimateSet ClimaX training."""

    def __init__(self, optimizer: Optimizer, warmup_epochs: int, max_epochs: int,
                 warmup_start_lr: float = 0.0, eta_min: float = 0.0, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr)
                / max(1, self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (
                1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs)
                             / (self.max_epochs - self.warmup_epochs))
            )
            for base_lr in self.base_lrs
        ]

    def get_lr(self):
        return self._get_closed_form_lr()


# ---- Loss (copied from emulator.src.core.losses.LLweighted_RMSELoss_Climax) --
_lat_weighted_rmse_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LLweighted_RMSELoss_Climax(torch.nn.Module):
    def __init__(self, deg2rad: bool = True, mask=None):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction="none")
        self.deg2rad = deg2rad
        self.mask = mask

    def forward(self, pred, y):
        mse = self.mse(pred, y)
        lat_size = y.shape[-1] # NOTE Changed from -1
        lats = torch.linspace(-90, 90, lat_size)
        if self.deg2rad:
            weights = torch.cos((torch.pi * lats) / 180)
        else:
            weights = torch.cos(lats)
        weights = weights / weights.mean()
        weights = weights.to(pred.device)
        if self.mask is not None:
            error = (mse * weights * self.mask).sum() / self.mask.sum()
        else:
            error = (mse * weights).mean()
        return torch.sqrt(error)


# ---- Dataset (grid-based, no HEALPix) ------------------------------------
from experiments.climate.data.climateset_data_no_hp import ClimatesetConfig
from experiments.climate.data.climateset_data_no_hp import ClimatesetData
from experiments.climate.data.climateset_data_no_hp import get_fire_type

# ---- Model (ClimaX adapter — no emulator/Lightning dependency) ------------
from experiments.climate.adapted_climateset_baselines.adapted_models.climax.climax_module import (
    ClimaXConfig,
    ClimaX,
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

def create_config(ensemble_id, epoch=50, batch_size=4):
    model_name, ensemble = CLIMATE_MODELS[ensemble_id]
    print(model_name, ensemble)

    criterion = LLweighted_RMSELoss_Climax()

    def loss_fn(output, batch):
        pred = output["logits_output"]   # (B, T, C, H, W)
        target = batch["target"]         # (B, T, C, H, W)
        n_vars = pred.shape[2]
        loss = sum(
            criterion(pred[:, :, i, :, :], target[:, :, i, :, :])
            for i in range(n_vars)
        )
        return loss / n_vars

    # Shared train / val dataset parameters
    random_seed   = 1
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
        channels_last=False,   # ClimaX expects channels-first
        fire_type=get_fire_type(model_name),
    )

    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=ClimaXConfig(
            # Grid dimensions for ClimateSet 250 km grid
            lon=96,
            lat=144,
            # Variable names must match the order the data loader yields channels
            in_vars=["BC", "CH4", "SO2", "CO2"],
            out_vars=["tas", "pr"],
            # Architecture — matches climax.yaml defaults for ClimateSet
            patch_size=16,
            embed_dim=1024,
            depth=8,
            decoder_depth=2,
            num_heads=16,
            mlp_ratio=4.0,
            drop_path=0.1,
            drop_rate=0.1,
            channel_agg="attention",
            init_mode="small",
            learn_pos_emb=False,
            freeze_encoder=False,
            # Sequence settings
            seq_len=seq_len,
            seq_to_seq=seq_to_seq,
            no_time_aggregation=seq_to_seq,  # seq_to_seq → no time aggregation
            time_history=1,  # each of the 12 timesteps gets its own time positional embedding
            channels_last=False,
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
                weight_decay=1e-5,
                lr=5e-4,
            ),
        ),
        scheduler_config=SchedulerConfig(
            scheduler=LinearWarmupCosineAnnealingLR,
            kwargs=dict(warmup_epochs=5, max_epochs=epoch, warmup_start_lr=1e-8, eta_min=1e-8),
        ),
        gradient_clipping=1.0,
        batch_size=batch_size,
        ensemble_id=ensemble_id,
        _version=2,
    )

    train_eval = TrainEval(
        train_metrics=[create_metric(loss_fn)],
        validation_metrics=[create_metric(loss_fn)],
        log_gradient_norm=True,
    )

    train_run = TrainRun(
        project="climate_climax_baseline",
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
    mf.register(ClimaXConfig, ClimaX)

    print("Starting distributed training...")
    config = create_config(ensemble_id=variant_idx, epoch=50)
    request_train_run(config)
    distributed_train([config])
