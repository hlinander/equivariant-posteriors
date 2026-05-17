import os
import torch
from lib.generic_ablation import get_config_grid

from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.train_dataclasses import TrainEval
from lib.metric import create_metric
from lib.train_distributed import request_train_run
from lib.distributed_trainer import distributed_train
import lib.data_factory as data_factory
import lib.model_factory as model_factory

from experiments.climate.data.climateset_data_hp import ClimatesetHPConfig, ClimatesetDataHP
from experiments.climate.data.climateset_data_hp import get_fire_type
from experiments.climate.models.swin_hp_climateset import SwinHPClimatesetConfig
from experiments.climate.models.swin_hp_climateset_fixed import SwinHPClimatesetFixed

NSIDE = 32
CLIMATE_MODELS = [
    ("AWI-CM-1-1-MR",    "r1i1p1f1"),
    ("BCC-CSM2-MR",      "r1i1p1f1"),
    ("CAS-ESM2-0",       "r3i1p1f1"),
    ("CNRM-CM6-1-HR",    "r1i1p1f2"),
    ("EC-Earth3",        "r1i1p1f1"),
    ("EC-Earth3-Veg-LR", "r1i1p1f1"),
    ("FGOALS-f3-L",      "r1i1p1f1"),
    ("GFDL-ESM4",        "r1i1p1f1"),
    ("INM-CM4-8",        "r1i1p1f1"),
    ("INM-CM5-0",        "r1i1p1f1"),
    ("MPI-ESM1-2-HR",    "r1i1p1f1"),
    ("MRI-ESM2-0",       "r1i1p1f1"),
    ("NorESM2-LM",       "r1i1p1f1"),
    ("NorESM2-MM",       "r1i1p1f1"),
    ("TaiESM1",          "r1i1p1f1"),
]

N_SEEDS  = int(os.environ.get("N_SEEDS",  "5"))
N_MODELS = int(os.environ.get("N_MODELS", "15"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "300"))


def create_config(
    ensemble_id,
    climate_model_idx=0,
    epoch=200,
    batch_size=12,
    embed_dims=[192 // 4, 384 // 4, 384 // 4, 192 // 4],
    depths=[4, 12, 12, 4],
    drop_rate=0.0,
    lr=2e-4,
):
    model_name, ensemble = CLIMATE_MODELS[climate_model_idx]
    print(f"climate_model={model_name}, ensemble={ensemble}, seed={ensemble_id}")

    loss = torch.nn.MSELoss()

    def loss_fn(output, batch):
        return loss(output["logits_output"], batch["target"])

    data_cfg_common = dict(
        nside=NSIDE,
        climate_model=model_name,
        ensemble=ensemble,
        output_vars=["tas"],
        scenarios=["ssp126", "ssp370", "ssp585"],
        seq_len=1,
        seq_to_seq=True,
        normalized=True,
        cache=True,
        val_fraction=0.1,
        random_seed=7,
        fire_type=get_fire_type(model_name),
    )

    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=SwinHPClimatesetConfig(
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
        _version=11,
    )

    train_eval = TrainEval(
        train_metrics=[create_metric(loss_fn)],
        validation_metrics=[create_metric(loss_fn)],
        log_gradient_norm=True,
    )

    return TrainRun(
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


def create_configs():
    return get_config_grid(create_config, dict(
        ensemble_id=list(range(N_SEEDS)),
        climate_model_idx=list(range(N_MODELS)),
        embed_dims=[[192//4, 384//4, 384//4, 192//4]],
        depths=[[4, 12, 12, 4]],
        batch_size=[12],
        epoch=[N_EPOCHS],
        drop_rate=[0.0],
        lr=[2e-4],
    ))


def run(config):
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimatesetFixed)

    request_train_run(config)
    distributed_train([config])


if __name__ == "__main__":
    configs = create_configs()
    print(f"{len(configs)} jobs:")
    for c in configs:
        print(" ", c())
