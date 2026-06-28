"""
Multi-seed training config for SwinHP with alternative Conv weight initialisation.

Uses SwinHPClimatesetInitAltConfig (distinct from SwinHPClimatesetConfig) so that
runs get a different hash and do not share checkpoints with the standard config.
The only behavioural difference is that SwinHPClimatesetTruncInit applies trunc_normal_
to Conv/ConvTranspose1d layers instead of PyTorch's default Kaiming init.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Literal

import torch.nn as nn

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
from experiments.climate.models.swin_hp_climateset_trunc_init import SwinHPClimatesetTruncInit
from lib.serialize_human import serialize_human as _serialize_human

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


@dataclass
class SwinHPClimatesetInitAltConfig:
    """Identical fields to SwinHPClimatesetConfig; distinct class for unique run hashes."""
    base_pix: int = 12
    nside: int = 64
    patch_size: int = 16
    window_size: int = 36
    shift_size: int = 2
    shift_strategy: Literal["nest_roll", "nest_grid_shift", "ring_shift"] = "nest_roll"
    rel_pos_bias: Optional[Literal["flat"]] = None
    patch_embed_norm_layer: Optional[Literal[nn.LayerNorm]] = None
    depths: List[int] = field(default_factory=lambda: [2, 6, 6, 2])
    num_heads: List[int] = field(default_factory=lambda: [6, 12, 12, 6])
    embed_dims: List[int] = field(default_factory=lambda: [192, 384, 384, 192])
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    use_cos_attn: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Literal[nn.LayerNorm] = nn.LayerNorm
    use_v2_norm_placement: bool = False
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    dev_mode: bool = False
    pad_fix: bool = True

    def serialize_human(self):
        return _serialize_human(self.__dict__)


def create_config(
    ensemble_id,
    epoch=250,
    batch_size=12,
    climate_model_idx=0,
    lr=2e-4,
    embed_dims=[192 // 4, 384 // 4, 384 // 4, 192 // 4],
    drop_rate=0.0,
    depths=[4, 12, 12, 4],
    output_vars=None,
):
    model_name, ensemble = CLIMATE_MODELS[climate_model_idx]
    print(f"climate_model={model_name}, ensemble={ensemble}, seed={ensemble_id}")

    mse = torch.nn.MSELoss()

    def loss_fn(output, batch):
        return mse(output["logits_output"], batch["target"])

    random_seed  = 1
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
        **({"output_vars": output_vars} if output_vars is not None else {}),
    )

    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=SwinHPClimatesetInitAltConfig(
            base_pix=12,
            nside=NSIDE,
            dev_mode=False,
            depths=depths,
            num_heads=[6, 12, 12, 6],
            embed_dims=embed_dims,
            window_size=[1, 64],
            use_cos_attn=False,
            use_v2_norm_placement=True,
            drop_rate=drop_rate,
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
        _version=13,
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
