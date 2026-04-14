"""
Evaluation sweep for multi-seed climate experiments.

Creates one SLURM job per (seed × climate_model_idx) pair.  Each job loops
over all checkpoints sequentially, so epochs do NOT multiply the job count.

Usage:
    # All checkpoints for NorESM2-LM (idx 12) across 3 seeds → 3 jobs:
    CONFIG=experiments/climate/persisted_configs/train_cnn_lstm_nohp_newlossfn_multiseed.py \
    CLIMATE_MODEL_IDX=12 N_SEEDS=3 NUM_VARIANTS=1 \
    python run_slurm_sweep.py experiments/climate/evaluation/evaluate_all_checkpoints_multiseed.py

    # Evaluate only the final epoch (EPOCH=500):
    CONFIG=experiments/climate/persisted_configs/train_cnn_lstm_nohp_newlossfn_multiseed.py \
    CLIMATE_MODEL_IDX=12 N_SEEDS=3 NUM_VARIANTS=1 EPOCH=500 \
    python run_slurm_sweep.py experiments/climate/evaluation/evaluate_all_checkpoints_multiseed.py

    # HEALPix baseline (SwinHP) — evaluator is auto-detected from CONFIG:
    CONFIG=experiments/climate/persisted_configs/train_climate_baseline_multiseed.py \
    CLIMATE_MODEL_IDX=12 N_SEEDS=3 NUM_VARIANTS=1 \
    python run_slurm_sweep.py experiments/climate/evaluation/evaluate_all_checkpoints_multiseed.py
"""

import os
from lib.generic_ablation import get_config_grid
def _detect_evaluator():
    import importlib.util
    from pathlib import Path
    config_path = os.environ["CONFIG"]
    spec = importlib.util.spec_from_file_location(Path(config_path).stem, config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        train_run = mod.create_config(0, climate_model_idx=0)
    except TypeError:
        train_run = mod.create_config(0)
    is_hp = "HP" in type(train_run.train_config.train_data_config).__name__
    return "hp" if is_hp else "nohp"

_evaluator = _detect_evaluator()
if _evaluator == "nohp":
    from experiments.climate.evaluation.evaluate_climate_nohp import evaluate_climate, load_create_config
else:
    from experiments.climate.evaluation.evaluate_climate_hp import evaluate_climate, load_create_config


def _get_epochs():
    """Return the list of epochs to evaluate."""
    create_config = load_create_config(os.environ["CONFIG"])
    c = create_config(0, climate_model_idx=0)
    if "EPOCH" in os.environ:
        return [int(os.environ["EPOCH"])]
    step = int(os.environ.get("EVAL_EVERY", str(c.keep_nth_epoch_checkpoints)))
    return list(range(0, c.epochs + 1, step))


def create_configs():
    n_seeds = int(os.environ.get("N_SEEDS", "5"))
    n_variants = int(os.environ.get("NUM_VARIANTS", "1"))
    epochs = _get_epochs()
    print(
        f"Sweep: {n_variants} climate model(s) × {n_seeds} seeds = "
        f"{n_variants * n_seeds} jobs, each evaluating {len(epochs)} epochs sequentially"
    )
    # One job per (climate_model, seed) — epochs are looped inside run().
    climate_model_start = int(os.environ.get("CLIMATE_MODEL_IDX", "0"))
    return get_config_grid(
        lambda **x: dict(**x),
        dict(
            seed=list(range(n_seeds)),
            climate_model_idx=list(range(climate_model_start, climate_model_start + n_variants)),
        ),
    )


def run(config):
    base_create_config = load_create_config(os.environ["CONFIG"])
    climate_model_idx = config["climate_model_idx"]
    seed = config["seed"]
    epochs = _get_epochs()

    # Resolve the model name for clearer logging
    sample_run = base_create_config(ensemble_id=seed, climate_model_idx=climate_model_idx)
    model_name = sample_run.train_config.train_data_config.climate_model

    config_name = os.path.basename(os.environ["CONFIG"]).replace(".py", "")
    print(f"=== [{config_name}] Evaluating {model_name} (idx={climate_model_idx}), seed={seed}, epochs={epochs} ===")

    curried = lambda ensemble_id, **kw: base_create_config(
        ensemble_id=ensemble_id, climate_model_idx=climate_model_idx, **kw
    )

    best_epoch, best_rmse = None, float("inf")
    for epoch in epochs:
        print(f"[{model_name} seed={seed}] epoch {epoch}")
        rmse = evaluate_climate(curried, epoch, variant_idx=seed)
        if rmse is not None and rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch

    if best_epoch is not None:
        print(f"=== BEST [{model_name} seed={seed}]: epoch {best_epoch}, RMSE {best_rmse:.6f} ===")


if __name__ == "__main__":
    configs = create_configs()
    print(configs)
    for config in configs:
        run(config())
