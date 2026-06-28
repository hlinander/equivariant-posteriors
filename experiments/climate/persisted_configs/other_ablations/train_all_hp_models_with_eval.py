import os
from lib.generic_ablation import get_config_grid

from experiments.climate.persisted_configs.final_climate_pear_runs.train_climate_pear_multiseed import (
    create_config as create_pear_config,
    ClimatesetHPConfig,
    ClimatesetDataHP,
    SwinHPClimatesetConfig,
    SwinHPClimateset,
)
from experiments.climate.persisted_configs.multi_timestep_pear_runs.train_climate_pear_temporal_atn_multiseed import (
    create_config as create_temporal_atn_config,
    SwinHPClimatesetTemporalAtnConfig,
    SwinHPClimatesetTemporalAtn,
)
from experiments.climate.evaluation.evaluate_climate_hp import evaluate_climate
from lib.train_distributed import request_train_run
from lib.distributed_trainer import distributed_train
import lib.data_factory as data_factory
import lib.model_factory as model_factory

N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
N_MODELS = int(os.environ.get("N_MODELS", "15"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "1000"))


def create_configs():
    return get_config_grid(lambda **kw: {"_fn": create_pear_config, **kw}, dict(
        ensemble_id=list(range(N_SEEDS)),
        climate_model_idx=list(range(N_MODELS)),
        embed_dims=[[192 * 2, 384 * 2, 384 * 2, 192 * 2]],
        depths=[[2, 6, 6, 2]],
        batch_size=[12],
        epoch=[N_EPOCHS],
        drop_rate=[0.0],
        lr=[2e-4],
    ))


def _make_epochs(max_epoch, step=10):
    return list(range(0, max_epoch + 1, step))


def run(config):
    base_create_config = config["_fn"]
    ensemble_id        = config["ensemble_id"]
    climate_model_idx  = config["climate_model_idx"]
    embed_dims         = config["embed_dims"]
    depths             = config["depths"]
    batch_size         = config["batch_size"]
    drop_rate          = config["drop_rate"]
    lr                 = config["lr"]
    max_epoch          = config["epoch"]

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)
    mf.register(SwinHPClimatesetTemporalAtnConfig, SwinHPClimatesetTemporalAtn)

    train_run = base_create_config(
        epoch=max_epoch,
        ensemble_id=ensemble_id,
        climate_model_idx=climate_model_idx,
        embed_dims=embed_dims,
        depths=depths,
        batch_size=batch_size,
        drop_rate=drop_rate,
        lr=lr,
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
        **kw,
    )

    model_name = train_run.train_config.train_data_config.climate_model
    epochs = _make_epochs(max_epoch)
    print(f"=== Evaluating {model_name} (seed={ensemble_id}), epochs={epochs} ===")

    best_epoch, best_rmse = None, float("inf")
    for epoch in epochs:
        print(f"[{model_name} seed={ensemble_id}] epoch {epoch}")
        rmse = evaluate_climate(curried, epoch, variant_idx=ensemble_id)
        if rmse is not None and rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch

    if best_epoch is not None:
        print(f"=== BEST [{model_name} seed={ensemble_id}]: "
              f"epoch {best_epoch}, RMSE {best_rmse:.6f} ===")


if __name__ == "__main__":
    configs = create_configs()
    print(f"{len(configs)} jobs:")
    for c in configs:
        print(" ", c())
