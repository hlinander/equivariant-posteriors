import os
from lib.generic_ablation import get_config_grid
from lib.paths import get_model_epoch_checkpoint_path
from experiments.climate.evaluation.evaluate_climate_hp import evaluate_climate
from experiments.climate.persisted_configs.train_climate_pear_multiseed import (
    create_config as create_pear_config,
)

N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
N_MODELS = int(os.environ.get("N_MODELS", "15"))

# Matches the current sweep in train_all_hp_models.py:
#   embed_dims=[192*2, 384*2, 384*2, 192*2], depths=[2,6,6,2], lr=2e-4, drop_rate in [0.0, 0.1]
_TARGET_HP_COMBOS = [
    dict(embed_dims=[384, 768, 768, 384], lr=2e-4, depths=[2, 6, 6, 2], batch_size=12, drop_rate=0.0),
    dict(embed_dims=[384, 768, 768, 384], lr=2e-4, depths=[2, 6, 6, 2], batch_size=12, drop_rate=0.1),
]


def _is_done(hp, ensemble_id, climate_model_idx, epoch=180):
    train_run = create_pear_config(
        epoch=epoch,
        ensemble_id=ensemble_id,
        climate_model_idx=climate_model_idx,
        **hp,
    )
    return get_model_epoch_checkpoint_path(train_run.train_config, epoch).is_file()


def create_configs():
    configs = []
    for hp in _TARGET_HP_COMBOS:
        for ensemble_id in range(N_SEEDS):
            for climate_model_idx in range(N_MODELS):
                if _is_done(hp, ensemble_id, climate_model_idx):
                    configs += get_config_grid(
                        lambda **kw: {"_fn": create_pear_config, **kw},
                        dict(
                            ensemble_id=[ensemble_id],
                            climate_model_idx=[climate_model_idx],
                            epoch=[180],
                            **{k: [v] for k, v in hp.items()},
                        ),
                    )
    return configs


def _make_epochs(max_epoch, step=10):
    return list(range(0, max_epoch + 1, step))


def run(config):
    base_create_config = config["_fn"]
    ensemble_id = config["ensemble_id"]
    climate_model_idx = config["climate_model_idx"]
    embed_dims = config["embed_dims"]
    depths = config["depths"]
    batch_size = config["batch_size"]
    drop_rate = config["drop_rate"]
    lr = config["lr"]
    max_epoch = config["epoch"]
    epochs = _make_epochs(max_epoch)

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

    sample = curried(ensemble_id=ensemble_id)
    model_name = sample.train_config.train_data_config.climate_model
    print(f"=== Evaluating {model_name} (idx={climate_model_idx}), seed={ensemble_id}, "
          f"embed_dims={embed_dims}, depths={depths}, batch_size={batch_size}, lr={lr}, epochs={epochs} ===")

    best_epoch, best_rmse = None, float("inf")
    for epoch in epochs:
        print(f"[{model_name} seed={ensemble_id}] epoch {epoch}")
        rmse = evaluate_climate(curried, epoch, variant_idx=ensemble_id)
        if rmse is not None and rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch

    if best_epoch is not None:
        print(f"=== BEST [{model_name} seed={ensemble_id}]: epoch {best_epoch}, RMSE {best_rmse:.6f} ===")


if __name__ == "__main__":
    configs = create_configs()
    print(f"{len(configs)} jobs found, evaluating the first one...")
    run(configs[0]())
