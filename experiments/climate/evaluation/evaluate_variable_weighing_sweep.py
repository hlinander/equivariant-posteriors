import os
from lib.generic_ablation import get_config_grid
from experiments.climate.evaluation.evaluate_climate_hp import evaluate_climate
from experiments.climate.persisted_configs.train_climate_pear_multiseed import (
    create_config as create_pear_config,
)

N_SEEDS = int(os.environ.get("N_SEEDS", "1"))
N_MODELS = int(os.environ.get("N_MODELS", "15"))

PR_WEIGHINGS = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]  # 0.5 is the default, excluded


# def create_configs():
#     return get_config_grid(lambda **kw: {"_fn": create_pear_config, **kw}, dict(
#         ensemble_id=list(range(N_SEEDS)),
#         climate_model_idx=list(range(N_MODELS)),
#         embed_dims=[[192*2, 384*2, 384*2, 192*2]],
#         depths=[[2, 6, 6, 2]],
#         batch_size=[12],
#         epoch=[530],
#         drop_rate=[0.0],
#         lr=[2e-4],
#         pr_variable_weighing=PR_WEIGHINGS,
#     ))

# def create_configs():
#     return get_config_grid(lambda **kw: {"_fn": create_pear_config, **kw}, dict(
#         ensemble_id=list(range(N_SEEDS)),
#         climate_model_idx=list(range(N_MODELS)),
#         embed_dims=[[192, 384, 384, 192]],
#         depths=[[2, 6, 6, 2]],
#         batch_size=[12],
#         epoch=[240],
#         drop_rate=[0.0],
#         lr=[2.5e-4],
#         pr_variable_weighing=PR_WEIGHINGS,
#     ))

def create_configs():
    return get_config_grid(lambda **kw: {"_fn": create_pear_config, **kw}, dict(
        ensemble_id=list(range(N_SEEDS)),
        climate_model_idx=list(range(N_MODELS)),
        embed_dims=[[192//4, 384//4, 384//4, 192//4]],
        depths=[[4, 12, 12, 4]],
        batch_size=[12],
        epoch=[250],
        drop_rate=[0.0],
        lr=[2e-4],
        pr_variable_weighing=PR_WEIGHINGS,
    ))


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
    pr_variable_weighing = config["pr_variable_weighing"]
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
        pr_variable_weighing=pr_variable_weighing,
        **kw,
    )

    sample = curried(ensemble_id=ensemble_id)
    model_name = sample.train_config.train_data_config.climate_model
    print(f"=== Evaluating {model_name} (idx={climate_model_idx}), seed={ensemble_id}, "
          f"pr_weighing={pr_variable_weighing}, epochs={epochs} ===")

    best_epoch, best_rmse = None, float("inf")
    for epoch in epochs:
        print(f"[{model_name} seed={ensemble_id} pr_w={pr_variable_weighing}] epoch {epoch}")
        rmse = evaluate_climate(curried, epoch, variant_idx=ensemble_id)
        if rmse is not None and rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch

    if best_epoch is not None:
        print(f"=== BEST [{model_name} seed={ensemble_id} pr_w={pr_variable_weighing}]: "
              f"epoch {best_epoch}, RMSE {best_rmse:.6f} ===")


if __name__ == "__main__":
    configs = create_configs()
    print(f"{len(configs)} jobs:")
    for c in configs:
        print(" ", c())
