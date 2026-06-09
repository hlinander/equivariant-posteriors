#!/usr/bin/env python
"""Slurm sweep over checkpoints: compute mean top-FTLE on train + val at every
saved epoch checkpoint of the fraction=0.3 runs from large_p_sweep_configs.

One slurm task per (train_run, epoch) pair. Launch via:
    uv run python run_slurm_sweep.py experiments/grokking/eval_ftle_checkpoints.py
"""
import math
import os
from dataclasses import replace

import torch

import lib.data_factory as data_factory
from lib.checkpoint_step import resolve_step_for_epoch
from lib.ddp import ddp_setup
from lib.export import export_all
from lib.paths import get_checkpoint_path
from lib.render_duck import (
    ensure_duck,
    insert_checkpoint_sample_metric,
    insert_model_with_model_id,
    insert_or_update_train_run,
)
from lib.serialization import (
    DeserializeConfig,
    deserialize_model,
    list_checkpoint_epochs,
)
from lib.stable_hash import stable_hash_str

from experiments.grokking.finite_field_det import large_p_sweep_configs
from experiments.grokking.lyapunov_metric import (
    SUBSAMPLE_N,
    _compute_ftle,
)


METRIC_NAME = "ftle"
TARGET_FRAC = 0.3


def _frac_runs():
    runs = [factory() for factory in large_p_sweep_configs()]
    return [r for r in runs if r.train_config.train_data_config.frac == TARGET_FRAC]


def _filter_epochs(epochs):
    min_e = int(os.environ.get("MIN_EPOCH", "0"))
    max_e_env = os.environ.get("MAX_EPOCH")
    max_e = int(max_e_env) if max_e_env else None
    return [e for e in epochs if e >= min_e and (max_e is None or e <= max_e)]


def _make_factory(train_run, epoch):
    def _factory():
        return {"train_run": train_run, "epoch": epoch}
    return _factory


def create_configs():
    out = []
    for train_run in _frac_runs():
        ckpt_hash = stable_hash_str(train_run.train_config)
        epochs = _filter_epochs(list_checkpoint_epochs(ckpt_hash))
        for epoch in epochs:
            out.append(_make_factory(train_run, epoch))
    return out


def _make_dataset(data_config, device_id):
    ds = data_factory.get_factory().create(data_config)
    if hasattr(ds, "to"):
        ds.to(torch.device(device_id))
    return ds


def _subsample_indices(n_total, device, seed):
    n = min(SUBSAMPLE_N, n_total)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randperm(n_total, device=device, generator=gen)[:n]


def _eval_split(model, dataset, split, model_id, step, epoch, ensemble_id):
    n_total = len(dataset)
    if n_total == 0:
        return
    device = dataset.xs.device
    seed = epoch * 10007 + ensemble_id + (0 if split == "train" else 1)
    idx = _subsample_indices(n_total, device, seed)
    xs = dataset.xs[idx]
    with torch.enable_grad():
        lambdas = _compute_ftle(model, xs)
    mean = float(lambdas.detach().float().mean().item())
    insert_checkpoint_sample_metric(
        model_id,
        step,
        f"{METRIC_NAME}_{split}",
        type(dataset).__name__,
        [],
        mean,
        [],
    )
    print(f"[ftle-eval]   {split}: mean={mean:.4f} (n={xs.shape[0]})")


def run(config):
    device_id = ddp_setup()
    train_run = config["train_run"]
    epoch = config["epoch"]
    ckpt_hash = stable_hash_str(train_run.train_config)
    print(f"[ftle-eval] {ckpt_hash} epoch={epoch}")

    eval_train_run = replace(train_run, epochs=epoch)
    deser = deserialize_model(DeserializeConfig(eval_train_run, device_id))
    if deser is None:
        print(f"[ftle-eval] Failed to deserialize {ckpt_hash} epoch={epoch}; skipping")
        return

    ensure_duck(train_run)
    insert_model_with_model_id(train_run, deser.model_id)
    insert_or_update_train_run(train_run, deser.model_id)

    checkpoint_path = get_checkpoint_path(train_run.train_config)
    step = resolve_step_for_epoch(checkpoint_path, epoch)
    if step is None:
        batch_size = train_run.train_config.batch_size
        n_train = train_run.train_config.train_data_config.n_samples
        step = epoch * math.ceil(n_train / batch_size)
        print(f"[ftle-eval] WARNING: step from checkpoints table not found; using step={step}")

    model = deser.model
    model.eval()
    ensemble_id = train_run.train_config.ensemble_id

    train_ds = _make_dataset(train_run.train_config.train_data_config, device_id)
    _eval_split(model, train_ds, "train", deser.model_id, step, epoch, ensemble_id)

    val_cfg = train_run.train_config.val_data_config
    if val_cfg is not None:
        val_ds = _make_dataset(val_cfg, device_id)
        _eval_split(model, val_ds, "val", deser.model_id, step, epoch, ensemble_id)

    print(f"[ftle-eval] Exporting metrics for {ckpt_hash}…")
    exported = export_all(train_run)
    print(f"[ftle-eval] Exported {len(exported) if exported else 0} files")
