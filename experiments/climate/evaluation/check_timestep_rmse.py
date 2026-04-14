"""
Quick local check: print per-timestep lat-weighted RMSE for a seq-to-seq NoHP model.
No database writes — results go straight to the terminal.

Usage:
  CONFIG=experiments/climate/persisted_configs/train_climax_nohp.py \
  SLURM_ARRAY_TASK_ID=0 \
  EPOCH=50 \
  python experiments/climate/evaluation/check_timestep_rmse.py
"""

import os
import sys
import importlib
import torch
import numpy as np
from pathlib import Path

from lib.ddp import ddp_setup
from lib.serialization import deserialize_model, DeserializeConfig

import lib.data_factory as data_factory
import lib.model_factory as model_factory

from experiments.climate.data.climateset_data_no_hp import (
    ClimatesetConfig,
    ClimatesetData,
    load_training_stats_from_config,
)
from experiments.climate.adapted_climateset_baselines.adapted_models.climax.climax_module import (
    ClimaXConfig,
    ClimaX,
)
from experiments.climate.evaluation.metrics import LLweighted_RMSE_Climax


def load_create_config(module_file_path):
    module_name = Path(module_file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    return config_file.create_config


def main():
    device_id = ddp_setup()

    config_path = os.environ["CONFIG"]
    variant_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0").strip() or "0")
    epoch       = int(os.environ.get("EPOCH", "50"))
    split       = os.environ.get("SPLIT", "train")  # "train" or "val" or "test"

    print(f"Config : {config_path}")
    print(f"Variant: {variant_idx}  Epoch: {epoch}  Split: {split}")

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetConfig, ClimatesetData)
    mf = model_factory.get_factory()
    mf.register(ClimaXConfig, ClimaX)

    create_config = load_create_config(config_path)
    train_run = create_config(ensemble_id=variant_idx)
    train_run.epochs = epoch

    import copy
    data_cfg = copy.deepcopy(train_run.train_config.train_data_config)
    data_cfg.split = split
    if split == "test":
        data_cfg.scenarios = ["ssp245"]

    ds = ClimatesetData(data_cfg)
    if split in ("test", "val"):
        stats = load_training_stats_from_config(train_run.train_config.train_data_config)
        ds.set_normalization_stats(**stats)

    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, drop_last=False)

    deser_model = deserialize_model(DeserializeConfig(train_run=train_run, device_id=device_id))
    if deser_model is None:
        print("ERROR: could not deserialize model — check variant/epoch.")
        sys.exit(1)

    model = deser_model.model
    model.eval()

    all_preds   = []
    all_targets = []
    with torch.no_grad():
        for batch in dl:
            batch_device = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
            out = model(batch_device)
            all_preds.append(out["logits_output"].cpu())
            all_targets.append(batch_device["target"].cpu())

    preds = torch.cat(all_preds,   dim=0)  # (N, T, C, H, W)
    tgts  = torch.cat(all_targets, dim=0)

    if preds.dim() != 5:
        print(f"Model output shape {tuple(preds.shape)} is not seq-to-seq (need 5-D). Exiting.")
        sys.exit(1)

    N, T, C, H, W = preds.shape
    print(f"\nOutput shape: N={N}  T={T}  C={C}  H={H}  W={W}")

    preds_np = preds.numpy()
    tgts_np  = tgts.numpy()

    var_names = train_run.train_config.train_data_config.output_vars

    print(f"\n{'Timestep':>10}  {'Overall':>10}  " +
          "  ".join(f"{v:>10}" for v in var_names))
    print("-" * (10 + 12 + 12 * C))

    for t in range(T):
        per_channel = [
            LLweighted_RMSE_Climax(preds_np[:, t, c, :, :], tgts_np[:, t, c, :, :])
            for c in range(C)
        ]
        overall = float(np.mean(per_channel))
        row = f"{t:>10}  {overall:>10.4f}  " + "  ".join(f"{v:>10.4f}" for v in per_channel)
        print(row)

    # Summary: how much does error grow from t=0 to t=T-1?
    rmse_t = [
        float(np.mean([
            LLweighted_RMSE_Climax(preds_np[:, t, c, :, :], tgts_np[:, t, c, :, :])
            for c in range(C)
        ]))
        for t in range(T)
    ]
    print(f"\nRMSE t=0  : {rmse_t[0]:.4f}")
    print(f"RMSE t={T-1:<2}: {rmse_t[-1]:.4f}")
    print(f"Ratio last/first: {rmse_t[-1] / rmse_t[0]:.3f}x")


if __name__ == "__main__":
    main()
