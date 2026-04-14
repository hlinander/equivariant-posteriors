"""
Evaluation sweep for train_all_hp_models configurations.

Edit the grid below to control what gets evaluated, then run:
    python run_slurm_sweep.py experiments/climate/evaluation/evaluate_all_hp_models.py

Or dry-run:
    python run_slurm_sweep.py --dry-run experiments/climate/evaluation/evaluate_all_hp_models.py
"""

from lib.generic_ablation import get_config_grid
from experiments.climate.evaluation.evaluate_climate_hp import evaluate_climate
from experiments.climate.persisted_configs.train_all_hp_models import create_pear_config

# ── Edit these to match what you want to evaluate ──────────────────────────
N_SEEDS = 5
CLIMATE_MODEL_IDX = list(range(2))   # e.g. [12] for a single model
EMBED_DIMS = [[192, 384, 384, 192], [192//2, 384//2, 384//2, 192//2]]
BATCH_SIZE = [12, 24, 48]
EPOCHS = list(range(0, 201, 10))     # 200 epochs, checkpoint every 10
# ───────────────────────────────────────────────────────────────────────────


def create_configs():
    return get_config_grid(
        lambda **x: dict(**x),
        dict(
            seed=list(range(N_SEEDS)),
            climate_model_idx=CLIMATE_MODEL_IDX,
            embed_dims=EMBED_DIMS,
            batch_size=BATCH_SIZE,
        ),
    )


def run(config):
    seed = config["seed"]
    climate_model_idx = config["climate_model_idx"]
    embed_dims = config["embed_dims"]
    batch_size = config["batch_size"]

    curried = lambda ensemble_id, **kw: create_pear_config(
        ensemble_id=ensemble_id,
        climate_model_idx=climate_model_idx,
        embed_dims=embed_dims,
        batch_size=batch_size,
        **kw,
    )

    sample = curried(ensemble_id=seed)
    model_name = sample.train_config.train_data_config.climate_model
    print(f"=== Evaluating {model_name} (idx={climate_model_idx}), seed={seed}, "
          f"embed_dims={embed_dims}, batch_size={batch_size}, epochs={EPOCHS} ===")

    best_epoch, best_rmse = None, float("inf")
    for epoch in EPOCHS:
        print(f"[{model_name} seed={seed}] epoch {epoch}")
        rmse = evaluate_climate(curried, epoch, variant_idx=seed)
        if rmse is not None and rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch

    if best_epoch is not None:
        print(f"=== BEST [{model_name} seed={seed}]: epoch {best_epoch}, RMSE {best_rmse:.6f} ===")


if __name__ == "__main__":
    configs = create_configs()
    print(f"{len(configs)} jobs:")
    for c in configs:
        print(" ", c())
