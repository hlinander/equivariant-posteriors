"""
Per-timestep RMSE diagnostic for HEALPix seq-to-seq causal models.
No database writes — results go straight to the terminal.

Directly imports create_config from the base multiseed module and accepts
all hyperparameters as env vars so the checkpoint hash matches exactly.

Usage (d=4, seq_len=4 model):
  ENSEMBLE_ID=0 CLIMATE_MODEL_IDX=0 EPOCH=500 SEQ_LEN=4 WINDOW_SIZE_D=4 \
  python experiments/climate/evaluation/diagnose_per_position_rmse.py

ENV vars:
  ENSEMBLE_ID         seed index (default 0)
  CLIMATE_MODEL_IDX   GCM index 0-14 (default 0)
  EPOCH               checkpoint epoch (default 500)
  LR                  learning rate used during training (default 2e-4)
  SEQ_LEN             sequence length used during training (default 4)
  WINDOW_SIZE_D       temporal window size (default 4)
  WINDOW_SIZE_HP      spatial window size (default 64)
  BATCH_SIZE          batch size used during training (default 12)
  WEIGHT_DECAY        weight decay used during training (default 3e-6)
  SPLIT               test | val | train (default test)
"""

import os
import sys
import copy
import torch

from lib.ddp import ddp_setup
from lib.serialization import deserialize_model, DeserializeConfig
import lib.data_factory as data_factory
import lib.model_factory as model_factory

from experiments.climate.data.climateset_data_hp import (
    ClimatesetHPConfig,
    ClimatesetDataHP,
    load_training_stats_from_config,
)
from experiments.climate.models.climate_pear_temporal_atn_causal import (
    SwinHPClimatesetTemporalAtnCausalConfig,
    SwinHPClimatesetTemporalAtnCausal,
)
from experiments.climate.models.swin_hp_climateset import SwinHPClimatesetConfig, SwinHPClimateset
from experiments.climate.models.climate_pear_temporal_atn import (
    SwinHPClimatesetTemporalAtnConfig,
    SwinHPClimatesetTemporalAtn,
)
from experiments.climate.persisted_configs.multi_timestep_pear_runs.train_climate_pear_temporal_atn_causal_multiseed import (
    create_config,
)


def main():
    device_id = ddp_setup()

    ensemble_id       = int(os.environ.get("ENSEMBLE_ID",       "0"))
    climate_model_idx = int(os.environ.get("CLIMATE_MODEL_IDX", "0"))
    epoch             = int(os.environ.get("EPOCH",             "500"))
    lr                = float(os.environ.get("LR",              "2e-4"))
    seq_len           = int(os.environ.get("SEQ_LEN",           "4"))
    window_size_d     = int(os.environ.get("WINDOW_SIZE_D",     "4"))
    window_size_hp    = int(os.environ.get("WINDOW_SIZE_HP",    "64"))
    batch_size        = int(os.environ.get("BATCH_SIZE",        "12"))
    weight_decay      = float(os.environ.get("WEIGHT_DECAY",    "3e-6"))
    split             = os.environ.get("SPLIT", "test")

    window_size = [window_size_d, window_size_hp]

    print(f"ensemble_id       : {ensemble_id}")
    print(f"climate_model_idx : {climate_model_idx}")
    print(f"epoch             : {epoch}")
    print(f"lr                : {lr}")
    print(f"seq_len           : {seq_len}")
    print(f"window_size       : {window_size}")
    print(f"batch_size        : {batch_size}")
    print(f"weight_decay      : {weight_decay}")
    print(f"split             : {split}")

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetTemporalAtnCausalConfig, SwinHPClimatesetTemporalAtnCausal)
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)
    mf.register(SwinHPClimatesetTemporalAtnConfig, SwinHPClimatesetTemporalAtn)

    train_run = create_config(
        ensemble_id=ensemble_id,
        climate_model_idx=climate_model_idx,
        epoch=epoch,
        lr=lr,
        seq_len=seq_len,
        window_size=window_size,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )
    train_run.epochs = epoch

    data_cfg = copy.deepcopy(train_run.train_config.train_data_config)
    data_cfg.split = split
    if split == "test":
        data_cfg.scenarios = ["ssp245"]

    ds = ClimatesetDataHP(data_cfg)
    if split in ("test", "val"):
        stats = load_training_stats_from_config(train_run.train_config.train_data_config)
        ds.set_normalization_stats(**stats)

    dl = torch.utils.data.DataLoader(ds, batch_size=12, shuffle=False, drop_last=False)

    deser_model = deserialize_model(DeserializeConfig(train_run=train_run, device_id=device_id))
    if deser_model is None:
        print("ERROR: could not deserialize model — check ENSEMBLE_ID/CLIMATE_MODEL_IDX/EPOCH.")
        print("       Make sure the hyperparameters here match what was used during training.")
        sys.exit(1)

    model = deser_model.model
    model.eval()

    all_preds   = []
    all_targets = []
    with torch.no_grad():
        for batch in dl:
            batch_device = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
            # Squeeze trivial T=1 dim for non-temporal models (same as rmse_climate_hp)
            if batch_device["input"].dim() == 4:
                batch_device["input"]  = batch_device["input"].squeeze(1)
                batch_device["target"] = batch_device["target"].squeeze(1)
            out = model(batch_device)
            all_preds.append(out["logits_output"].cpu())
            all_targets.append(batch_device["target"].cpu())

    preds = torch.cat(all_preds,   dim=0)
    tgts  = torch.cat(all_targets, dim=0)

    print(f"\nOutput shape : {tuple(preds.shape)}")
    print(f"Target shape : {tuple(tgts.shape)}")

    if preds.dim() == 3:
        rmse = (preds - tgts).pow(2).mean(dim=(0, 2)).sqrt().mean()
        print(f"\nNon-temporal model (single position)")
        print(f"  RMSE: {rmse:.6f}")
        return

    if preds.dim() != 4:
        print(f"Unexpected output shape {tuple(preds.shape)}, expected (N, T, C, P) or (N, C, P).")
        sys.exit(1)

    N, T, C, P = preds.shape
    var_names = train_run.train_config.train_data_config.output_vars

    print(f"\n{'pos':>5}  {'overall':>10}  " + "  ".join(f"{v:>10}" for v in var_names))
    print("-" * (5 + 12 + 12 * C))

    rmse_per_t = []
    for t in range(T):
        sq          = (preds[:, t] - tgts[:, t]).pow(2)      # (N, C, P)
        per_channel = sq.mean(dim=(0, 2)).sqrt()              # (C,)
        overall_t   = per_channel.mean().item()
        rmse_per_t.append(overall_t)
        row = f"{t:>5}  {overall_t:>10.6f}  " + "  ".join(f"{v.item():>10.6f}" for v in per_channel)
        print(row)

    overall_flat = (preds - tgts).pow(2).mean(dim=(0, 2, 3)).sqrt().mean().item()
    print(f"\nFlattened average (current metric): {overall_flat:.6f}")
    print(f"Ratio pos{T-1}/pos0 : {rmse_per_t[-1] / rmse_per_t[0]:.4f}x  "
          f"({'better' if rmse_per_t[-1] < rmse_per_t[0] else 'worse'} at later positions)")


if __name__ == "__main__":
    main()
