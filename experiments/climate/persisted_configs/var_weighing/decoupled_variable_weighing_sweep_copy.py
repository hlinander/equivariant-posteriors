u"""
Sweep over pr_variable_weighing (0.1–0.9) for the three decoupled decoder variants:
  "dual_head", "split_decoder", "split_decoder_norm"

Mirrors variable_weighing_sweep_with_eval.py but uses SwinHPClimatesetDecoupled
instead of the shared-trunk SwinHPClimateset.
"""

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

from experiments.climate.data.climateset_data_hp import (
    ClimatesetHPConfig,
    ClimatesetDataHP,
    get_fire_type,
)
from experiments.climate.models.swin_hp_climateset_decoupled import (
    SwinHPClimatesetDecoupledConfig,
    SwinHPClimatesetDecoupled,
)
from experiments.climate.evaluation.evaluate_climate_hp import evaluate_climate
from experiments.climate.persisted_configs.var_weighing.train_climate_pear_pr_weighted_multiseed import (
    CLIMATE_MODELS,
    NSIDE,
)

N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
N_MODELS = int(os.environ.get("N_MODELS", "15"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "200"))

#PR_WEIGHINGS = [0.1, 0.4, 0.5, 0.6, 0.9]  # [0.1, 0.2, ..., 0.9]
DECOUPLING_VARIANTS = ["dual_head"] #, "split_decoder", "split_decoder_norm"]


def create_config(
    ensemble_id,
    decoupling="split_decoder",
    epoch=200,
    batch_size=12,
    climate_model_idx=0,
    lr=2e-4,
    embed_dims=None,
    drop_rate=0.0,
    depths=None,
    #pr_variable_weighing=0.5,
):
    if embed_dims is None:
        embed_dims = [192, 384, 384, 192]
    if depths is None:
        depths = [2, 6, 6, 2]

    model_name, ensemble = CLIMATE_MODELS[climate_model_idx]

    mse = torch.nn.MSELoss()

    def loss_fn(output, batch):
        return mse(output["logits_output"], batch["target"])

    data_cfg_common = dict(
        nside=NSIDE,
        climate_model=model_name,
        ensemble=ensemble,
        scenarios=["ssp126", "ssp370", "ssp585"],
        val_fraction=0.1,
        random_seed=1,
        seq_len=1,
        seq_to_seq=True,
        normalized=True,
        cache=True,
        fire_type=get_fire_type(model_name),
    )

    train_config = TrainConfig(
        extra=dict(
            loss_variant="full",
            #pr_variable_weighing=pr_variable_weighing,
            decoupling=decoupling,
        ),
        model_config=SwinHPClimatesetDecoupledConfig(
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
            decoupling=decoupling,
        ),
        train_data_config=ClimatesetHPConfig(**data_cfg_common, split="train"),
        val_data_config=ClimatesetHPConfig(**data_cfg_common, split="val"),
        loss=loss_fn,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=lr),
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
    return get_config_grid(
        lambda **kw: {"_fn": create_config, **kw},
        dict(
            ensemble_id=list(range(N_SEEDS)),
            climate_model_idx=list(range(N_MODELS)),
            embed_dims=[[192 // 4, 384 // 4, 384 // 4, 192 // 4]],
            depths=[[4, 12, 12, 4]],
            batch_size=[12],
            epoch=[N_EPOCHS],
            drop_rate=[0.0],
            lr=[2e-4],
            #pr_variable_weighing=PR_WEIGHINGS,
            decoupling=DECOUPLING_VARIANTS,
        ),
    )


def _make_epochs(max_epoch, step=10):
    return list(range(0, max_epoch + 1, step))


def run(config):
    base_create_config   = config["_fn"]
    ensemble_id          = config["ensemble_id"]
    climate_model_idx    = config["climate_model_idx"]
    embed_dims           = config["embed_dims"]
    depths               = config["depths"]
    batch_size           = config["batch_size"]
    drop_rate            = config["drop_rate"]
    lr                   = config["lr"]
    pr_variable_weighing = config["pr_variable_weighing"]
    decoupling           = config["decoupling"]
    max_epoch            = config["epoch"]

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetDecoupledConfig, SwinHPClimatesetDecoupled)

    train_run = base_create_config(
        epoch=max_epoch,
        ensemble_id=ensemble_id,
        climate_model_idx=climate_model_idx,
        embed_dims=embed_dims,
        depths=depths,
        batch_size=batch_size,
        drop_rate=drop_rate,
        lr=lr,
        pr_variable_weighing=pr_variable_weighing,
        decoupling=decoupling,
    )
    request_train_run(train_run)
    distributed_train([train_run])

    curried = lambda ensemble_id, **kw: base_create_config(
        epoch=max_epoch,
        ensemble_id=ensemble_id,
        climate_model_idx=climate_model_idx,
        embed_dims=embed_dims,
        depths=depths,
        batch_size=batch_size,
        drop_rate=drop_rate,
        lr=lr,
        pr_variable_weighing=pr_variable_weighing,
        decoupling=decoupling,
        **kw,
    )

    model_name = train_run.train_config.train_data_config.climate_model
    epochs = _make_epochs(max_epoch)
    print(
        f"=== Evaluating {model_name} (seed={ensemble_id}, "
        f"decoupling={decoupling}, pr_w={pr_variable_weighing}), epochs={epochs} ==="
    )

    best_epoch, best_rmse = None, float("inf")
    for epoch in epochs:
        print(
            f"[{model_name} seed={ensemble_id} "
            f"decoupling={decoupling} pr_w={pr_variable_weighing}] epoch {epoch}"
        )
        rmse = evaluate_climate(curried, epoch, variant_idx=ensemble_id)
        if rmse is not None and rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch

    if best_epoch is not None:
        print(
            f"=== BEST [{model_name} seed={ensemble_id} "
            f"decoupling={decoupling} pr_w={pr_variable_weighing}]: "
            f"epoch {best_epoch}, RMSE {best_rmse:.6f} ==="
        )