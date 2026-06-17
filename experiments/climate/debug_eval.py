#!/usr/bin/env python3
"""
Diagnostic script for debugging evaluation RMSE values.

Usage:
    # Evaluate a specific checkpoint (e.g. epoch 50, climate model 0, seed 0):
    python experiments/climate/debug_eval.py --epoch 50

    # Check a specific model index and seed:
    python experiments/climate/debug_eval.py --epoch 50 --model-idx 0 --seed 0

    # Limit to N batches for speed:
    python experiments/climate/debug_eval.py --epoch 50 --max-batches 5
"""

from __future__ import annotations
import sys
import argparse
import copy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

import lib.data_factory as data_factory
import lib.model_factory as model_factory
from lib.ddp import ddp_setup
from lib.serialization import deserialize_model, DeserializeConfig

from experiments.climate.data.climateset_data_hp import (
    ClimatesetHPConfig,
    ClimatesetDataHP,
    load_training_stats_from_config,
)
from experiments.climate.models.climate_pear_temporal_atn_causal import (
    SwinHPClimatesetTemporalAtnCausalConfig,
    SwinHPClimatesetTemporalAtnCausal,
)
from experiments.climate.persisted_configs.train_climate_pear_temporal_atn_causal_multiseed import (
    create_config,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--epoch",      type=int, default=50)
    p.add_argument("--model-idx",  type=int, default=0,  dest="model_idx")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--max-batches",type=int, default=None, dest="max_batches",
                   help="Limit number of test batches (None = all)")
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


def sep(title=""):
    w = 70
    if title:
        print(f"\n{'─' * 3} {title} {'─' * max(0, w - len(title) - 5)}")
    else:
        print("─" * w)


def tensor_summary(name, t):
    t = t.float()
    print(f"  {name}: shape={tuple(t.shape)}  "
          f"min={t.min():.4f}  max={t.max():.4f}  "
          f"mean={t.mean():.4f}  std={t.std():.4f}  "
          f"nan={t.isnan().sum().item()}  inf={t.isinf().sum().item()}")


def rmse_per_timestep(preds, targets):
    """preds/targets: (N, T, C, P) — return (T,) RMSE averaged over N, C, P."""
    sq = (preds.float() - targets.float()) ** 2
    return sq.mean(dim=(0, 2, 3)).sqrt()  # (T,)


def main():
    args = parse_args()

    sep("Registration")
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetTemporalAtnCausalConfig, SwinHPClimatesetTemporalAtnCausal)
    print("OK")

    device_id = args.device

    sep("Config")
    train_run = create_config(
        ensemble_id=args.seed,
        epoch=args.epoch,
        climate_model_idx=args.model_idx,
    )
    train_run.epochs = args.epoch
    model_name = train_run.train_config.train_data_config.climate_model
    print(f"  climate model : {model_name}")
    print(f"  seed          : {args.seed}")
    print(f"  epoch         : {args.epoch}")

    sep("Normalization stats")
    stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    if stats["output_stats"] is None:
        print("  ERROR: output_stats not found on disk!")
        sys.exit(1)
    out_mean = stats["output_stats"]["mean"]  # (1, C, 1)
    out_std  = stats["output_stats"]["std"]
    print(f"  output mean shape: {out_mean.shape}")
    for c, (m, s) in enumerate(zip(out_mean[0, :, 0], out_std[0, :, 0])):
        var_name = train_run.train_config.train_data_config.output_vars[c]
        print(f"    {var_name}: mean={m:.6e}, std={s:.6e}")

    sep("Test dataset")
    test_data_config = copy.deepcopy(train_run.train_config.train_data_config)
    test_data_config.scenarios = ["ssp245"]
    test_data_config.split = "test"
    test_ds = ClimatesetDataHP(test_data_config)
    test_ds.set_normalization_stats(**stats)
    print(f"  test set size : {len(test_ds)} sequences")

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=4, shuffle=False, drop_last=False,
    )

    sep("Load checkpoint")
    deser_config = DeserializeConfig(train_run=train_run, device_id=device_id)
    deser_model = deserialize_model(deser_config)
    if deser_model is None:
        print(f"  ERROR: Cannot deserialize model (epoch={args.epoch}, seed={args.seed})")
        print("  Checkpoints available? Check the run directory.")
        sys.exit(1)
    print(f"  model_id : {deser_model.model_id}")
    model = deser_model.model
    model.eval()

    sep("Batch shape check (first batch)")
    first_batch = next(iter(test_dl))
    inp = first_batch["input"]
    tgt = first_batch["target"]
    print(f"  input  shape : {tuple(inp.shape)}")
    print(f"  target shape : {tuple(tgt.shape)}")
    with torch.no_grad():
        inp_dev = inp.to(device_id)
        out = model({"input": inp_dev, "target": tgt.to(device_id)})
    pred = out["logits_output"].cpu()
    print(f"  output shape : {tuple(pred.shape)}")
    assert pred.shape == tgt.shape, (
        f"SHAPE MISMATCH: pred {tuple(pred.shape)} vs target {tuple(tgt.shape)}"
    )
    print("  Shapes match ✓")

    sep("Value ranges (first batch, normalized)")
    tensor_summary("input ", inp)
    tensor_summary("target", tgt)
    tensor_summary("pred  ", pred)

    sep("Zero-predictor baseline RMSE (normalized, first batch)")
    zero_rmse = tgt.float().pow(2).mean().sqrt()
    print(f"  zero-pred RMSE : {zero_rmse:.6f}  (target is ~N(0,1) normalized → expect ~1.0)")
    actual_rmse_first = (pred.float() - tgt.float()).pow(2).mean().sqrt()
    print(f"  model RMSE     : {actual_rmse_first:.6f}")

    sep(f"Full evaluation over test set (max_batches={args.max_batches})")
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            if args.max_batches is not None and i >= args.max_batches:
                break
            batch_dev = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            p = model(batch_dev)["logits_output"].cpu()
            t = batch["target"].cpu()
            all_preds.append(p)
            all_tgts.append(t)
    all_preds = torch.cat(all_preds, dim=0)  # (N, T, C, P)
    all_tgts  = torch.cat(all_tgts,  dim=0)

    tensor_summary("all predictions", all_preds)
    tensor_summary("all targets    ", all_tgts)

    sep("Per-timestep RMSE (normalized)")
    ts_rmse = rmse_per_timestep(all_preds, all_tgts)
    for t, r in enumerate(ts_rmse):
        print(f"  t={t:2d}  RMSE={r:.6f}")
    print(f"  overall mean: {ts_rmse.mean():.6f}")

    sep("Per-channel RMSE (normalized)")
    N, T, C, P = all_preds.shape
    flat_preds = all_preds.reshape(N * T, C, P).float()
    flat_tgts  = all_tgts.reshape(N * T, C, P).float()
    sq = (flat_preds - flat_tgts) ** 2
    rmse_per_ch = sq.mean(dim=(0, 2)).sqrt()  # (C,)
    output_vars = train_run.train_config.train_data_config.output_vars
    for c, (var, r) in enumerate(zip(output_vars, rmse_per_ch)):
        print(f"  {var}: RMSE={r:.6f}")

    sep("Denormalized per-channel RMSE")
    mean_t = torch.tensor(out_mean, dtype=torch.float32)  # (1, C, 1)
    std_t  = torch.tensor(out_std,  dtype=torch.float32)
    preds_d = flat_preds * std_t + mean_t
    tgts_d  = flat_tgts  * std_t + mean_t
    sq_d = (preds_d - tgts_d) ** 2
    rmse_per_ch_d = sq_d.mean(dim=(0, 2)).sqrt()
    for var, r in zip(output_vars, rmse_per_ch_d):
        print(f"  {var}: RMSE={r:.6f} (physical units)")

    sep("Comparison to zero predictor (denormalized)")
    zero_d = (tgts_d ** 2).mean(dim=(0, 2)).sqrt() + 0  # √E[y²] ≈ RMS of targets
    for var, zr, mr in zip(output_vars, zero_d, rmse_per_ch_d):
        ratio = mr / zr if zr > 0 else float("inf")
        print(f"  {var}: model={mr:.6f}, zero-pred={zr:.6f}, ratio={ratio:.3f}"
              f"  {'WORSE than zero!' if ratio > 1.0 else 'better than zero'}")

    sep("Done")


if __name__ == "__main__":
    main()
