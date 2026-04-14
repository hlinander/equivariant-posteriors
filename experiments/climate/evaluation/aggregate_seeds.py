"""
Aggregate evaluation metrics across random seeds and print statistics.

For each (climate_model, seed) pair the script loads the trained checkpoint,
computes latitude-weighted RMSE on the held-out ssp245 test set, and then
reports mean ± std across seeds.

Environment variables
---------------------
CONFIG          Path to the multi-seed training config (required).
N_SEEDS         Number of seeds (default: 10).
NUM_VARIANTS    Number of climate models to evaluate (default: 1).
EPOCH           Checkpoint epoch to evaluate (default: last epoch in config).

Usage:
    CONFIG=experiments/climate/persisted_configs/train_cnn_lstm_nohp_newlossfn_multiseed.py \
    N_SEEDS=10 NUM_VARIANTS=15 EPOCH=500 \
    python run.py experiments/climate/evaluation/aggregate_seeds.py
"""

import os
import sys
import copy
import importlib.util
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from lib.ddp import ddp_setup
from lib.serialization import deserialize_model, DeserializeConfig
import lib.data_factory as data_factory
import lib.model_factory as model_factory

from experiments.climate.data.climateset_data_no_hp import (
    ClimatesetConfig,
    ClimatesetData,
    load_training_stats_from_config,
)
from experiments.climate.adapted_climateset_baselines.adapted_models.unet import UNetConfig, UNet
from experiments.climate.adapted_climateset_baselines.adapted_models.cnn_lstm import (
    CNNLSTMConfig,
    CNNLSTM_ClimateBench,
)
from experiments.climate.models.GRU_wrapper import GRUTemporalWrapperConfig, GRUTemporalWrapper
from experiments.climate.evaluation.metrics import rmse_climate_nohp


def load_create_config(module_file_path):
    module_name = Path(module_file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    return config_file.create_config


def evaluate_single(create_config, epoch, seed, climate_model_idx, device_id):
    """Load a checkpoint and return RMSE metrics for one (climate_model, seed)."""
    train_run = create_config(ensemble_id=seed, climate_model_idx=climate_model_idx)
    train_run.epochs = epoch

    # Build test set (ssp245, not used in training)
    test_data_config = copy.deepcopy(train_run.train_config.train_data_config)
    test_data_config.scenarios = ["ssp245"]
    test_data_config.split = "test"
    test_ds = ClimatesetData(test_data_config)

    stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    test_ds.set_normalization_stats(**stats)

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=4, shuffle=False, drop_last=False,
    )

    deser_config = DeserializeConfig(train_run=train_run, device_id=device_id)
    deser_model = deserialize_model(deser_config)
    if deser_model is None:
        print(f"  [SKIP] Cannot deserialize climate_model={climate_model_idx}, seed={seed}, epoch={epoch}")
        return None

    model = deser_model.model
    model.eval()

    output_stats = stats["output_stats"]
    rmse_results = rmse_climate_nohp(model, test_dl, device_id, output_stats)

    output_var_names = train_run.train_config.train_data_config.output_vars
    per_var = {
        var_name: rmse_results["rmse_per_channel"][i].item()
        for i, var_name in enumerate(output_var_names)
    }
    per_var["overall"] = rmse_results["overall_rmse"].item()
    return per_var


def main():
    device_id = ddp_setup()

    # Register datasets and models
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetConfig, ClimatesetData)
    mf = model_factory.get_factory()
    mf.register(UNetConfig, UNet)
    mf.register(CNNLSTMConfig, CNNLSTM_ClimateBench)
    mf.register(GRUTemporalWrapperConfig, GRUTemporalWrapper)

    config_path = os.environ["CONFIG"]
    create_config = load_create_config(config_path)
    n_seeds = int(os.environ.get("N_SEEDS", "10"))
    n_variants = int(os.environ.get("NUM_VARIANTS", "1"))

    # Default epoch: use the epoch from create_config
    default_epoch = create_config(0, climate_model_idx=0).epochs
    epoch = int(os.environ.get("EPOCH", str(default_epoch)))

    print(f"Aggregating: {n_variants} climate model(s), {n_seeds} seeds, epoch {epoch}")
    print(f"Config: {config_path}")
    print("=" * 80)

    # Collect results: {climate_model_idx: [per_var_dict_seed0, per_var_dict_seed1, ...]}
    all_results = defaultdict(list)

    for variant in range(n_variants):
        for seed in range(n_seeds):
            print(f"\n--- climate_model={variant}, seed={seed}, epoch={epoch} ---")
            result = evaluate_single(create_config, epoch, seed, variant, device_id)
            if result is not None:
                all_results[variant].append(result)
                for k, v in result.items():
                    print(f"  {k}: {v:.6f}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY (mean ± std across seeds)")
    print("=" * 80)

    # Determine variable names from first successful result
    var_names = None
    for variant_results in all_results.values():
        if variant_results:
            var_names = list(variant_results[0].keys())
            break

    if var_names is None:
        print("No successful evaluations!")
        sys.exit(1)

    # Header
    header = f"{'Model':>6s}  {'n':>3s}"
    for var in var_names:
        header += f"  {var:>20s}"
    print(header)
    print("-" * len(header))

    csv_rows = []
    for variant in range(n_variants):
        results = all_results[variant]
        n = len(results)
        if n == 0:
            print(f"{variant:>6d}  {0:>3d}  (no results)")
            continue

        row = f"{variant:>6d}  {n:>3d}"
        csv_row = {"climate_model_idx": variant, "n_seeds": n}
        for var in var_names:
            values = np.array([r[var] for r in results])
            mean, std = values.mean(), values.std()
            row += f"  {mean:>8.6f} ± {std:.6f}"
            csv_row[f"{var}_mean"] = mean
            csv_row[f"{var}_std"] = std
        print(row)
        csv_rows.append(csv_row)

    # Save CSV alongside the config
    csv_path = Path(config_path).parent / f"{Path(config_path).stem}_seed_stats_epoch{epoch}.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
