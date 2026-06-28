"""
Lightweight helper to extract raw predictions and targets from a trained HP climate model.
Unlike evaluate_climate_hp.py, this runs only n_batches of data for fast interactive use.

Returns denormalized arrays (physical units) alongside normalized ones.

Two entry points
----------------
get_sample_predictions()          — load the first n_batches (good for exploration)
get_predictions_for_timestamp()   — load one specific (year, month) sample
"""
import copy
import torch
import numpy as np

from experiments.climate.evaluation.timestamp_utils import (
    sample_id_to_timestamp,
    find_sample_for_timestamp,
    format_timestamp,
)

import lib.data_factory as data_factory
import lib.model_factory as model_factory
from lib.ddp import ddp_setup
from lib.serialization import deserialize_model, DeserializeConfig

from experiments.climate.data.climateset_data_hp import (
    ClimatesetHPConfig,
    ClimatesetDataHP,
    load_training_stats_from_config,
)
from experiments.climate.models.swin_hp_climateset import SwinHPClimatesetConfig, SwinHPClimateset
from experiments.climate.models.swin_hp_climateset_seq import SwinHPClimatesetSeqConfig, SwinHPClimatesetSeq
from experiments.climate.models.GRU_wrapper import GRUTemporalWrapperConfig, GRUTemporalWrapper
from experiments.climate.models.climate_pear_temporal_atn import (
    SwinHPClimatesetTemporalAtnConfig,
    SwinHPClimatesetTemporalAtn,
)


def _register_factories():
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)
    mf.register(SwinHPClimatesetSeqConfig, SwinHPClimatesetSeq)
    mf.register(GRUTemporalWrapperConfig, GRUTemporalWrapper)
    mf.register(SwinHPClimatesetTemporalAtnConfig, SwinHPClimatesetTemporalAtn)


def get_sample_predictions(create_config, epoch, variant_idx=0, n_batches=2):
    """
    Load a trained HP climate model and run inference on the first n_batches of test data.

    Args:
        create_config: Callable accepting ensemble_id keyword; returns a TrainRun.
        epoch:         Checkpoint epoch to load.
        variant_idx:   Ensemble / seed index (passed as ensemble_id).
        n_batches:     How many batches to process (batch_size=12 → 12*n_batches samples).

    Returns dict with:
        predictions       np.ndarray (N, C, P) — normalized
        targets           np.ndarray (N, C, P) — normalized
        predictions_denorm np.ndarray (N, C, P) — physical units
        targets_denorm    np.ndarray (N, C, P) — physical units
        errors            np.ndarray (N, C, P) — abs(predictions_denorm - targets_denorm)
        var_names         list[str]             — e.g. ['tas', 'pr']
        model_name        str                   — CMIP6 model name
        nside             int
        epoch             int
    """
    _register_factories()
    device_id = ddp_setup()

    train_run = create_config(ensemble_id=variant_idx)
    train_run.epochs = epoch

    # Build test dataloader (ssp245, test split)
    test_data_config = copy.deepcopy(train_run.train_config.train_data_config)
    test_data_config.scenarios = ["ssp245"]
    test_data_config.split = "test"
    test_ds = ClimatesetDataHP(test_data_config)

    stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    test_ds.set_normalization_stats(**stats)

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=12, shuffle=False, drop_last=False
    )

    deser_model = deserialize_model(DeserializeConfig(train_run=train_run, device_id=device_id))
    if deser_model is None:
        raise RuntimeError(
            f"Could not load checkpoint for epoch={epoch}, variant={variant_idx}. "
            "Check that the checkpoint exists."
        )

    model = deser_model.model
    model.eval()

    all_preds, all_tgts, all_sample_ids = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            if i >= n_batches:
                break
            all_sample_ids.extend(batch["sample_id"].tolist())
            batch_dev = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            # Squeeze leading time dim for single-step models
            if batch_dev["input"].dim() == 4:
                batch_dev["input"] = batch_dev["input"].squeeze(1)
                batch_dev["target"] = batch_dev["target"].squeeze(1)

            out = model(batch_dev)
            all_preds.append(out["logits_output"].cpu())
            all_tgts.append(batch_dev["target"].cpu())

    preds = torch.cat(all_preds, dim=0)  # (N, C, P)
    tgts  = torch.cat(all_tgts,  dim=0)

    # Handle seq-to-seq (N, T, C, P) → (N*T, C, P)
    if preds.dim() == 4:
        N, T, C, P = preds.shape
        preds = preds.reshape(N * T, C, P)
        tgts  = tgts.reshape(N * T, C, P)

    mean_t = torch.tensor(stats["output_stats"]["mean"], dtype=torch.float32)
    std_t  = torch.tensor(stats["output_stats"]["std"],  dtype=torch.float32)
    preds_denorm = preds.float() * std_t + mean_t
    tgts_denorm  = tgts.float()  * std_t + mean_t

    data_cfg = train_run.train_config.train_data_config
    return {
        "predictions":        preds.float().numpy(),
        "targets":            tgts.float().numpy(),
        "predictions_denorm": preds_denorm.numpy(),
        "targets_denorm":     tgts_denorm.numpy(),
        "errors":             np.abs(preds_denorm.numpy() - tgts_denorm.numpy()),
        "sample_ids":         all_sample_ids,   # raw array offsets, one per sample
        "var_names":          list(data_cfg.output_vars),
        "model_name":         data_cfg.climate_model,
        "nside":              data_cfg.nside,
        "test_data_config":   test_data_config,
        "epoch":              epoch,
    }


def _build_model_and_test_ds(create_config, epoch, variant_idx):
    """Shared setup: load model + test dataset. Returns (model, test_ds, stats, data_cfg, device_id)."""
    _register_factories()
    device_id = ddp_setup()

    train_run = create_config(ensemble_id=variant_idx)
    train_run.epochs = epoch

    test_data_config = copy.deepcopy(train_run.train_config.train_data_config)
    test_data_config.scenarios = ["ssp245"]
    test_data_config.split = "test"
    test_ds = ClimatesetDataHP(test_data_config)

    stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    test_ds.set_normalization_stats(**stats)

    deser_model = deserialize_model(DeserializeConfig(train_run=train_run, device_id=device_id))
    if deser_model is None:
        raise RuntimeError(
            f"Could not load checkpoint for epoch={epoch}, variant={variant_idx}."
        )
    deser_model.model.eval()
    return deser_model.model, test_ds, stats, test_data_config, device_id


def get_predictions_for_timestamp(create_config, epoch, year, month, variant_idx=0):
    """
    Load one specific (year, month) sample from the ssp245 test set.

    Use this to compare the exact same timestep across different models or
    between the HP notebook and the nohp notebook.

    Returns dict with:
        prediction        np.ndarray (C, P) — denormalized
        target            np.ndarray (C, P) — denormalized
        error             np.ndarray (C, P) — abs error
        var_names         list[str]
        model_name        str
        nside             int
        scenario          str   e.g. "ssp245"
        year              int
        month             int
        timestamp_label   str   e.g. "ssp245 · Apr 2040"
    """
    model, test_ds, stats, test_data_config, device_id = _build_model_and_test_ds(
        create_config, epoch, variant_idx
    )

    dataloader_idx, _, _ = find_sample_for_timestamp(year, month, test_data_config)
    subset = torch.utils.data.Subset(test_ds, [dataloader_idx])
    dl = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

    mean_t = torch.tensor(stats["output_stats"]["mean"], dtype=torch.float32)
    std_t  = torch.tensor(stats["output_stats"]["std"],  dtype=torch.float32)

    with torch.no_grad():
        batch = next(iter(dl))
        batch_dev = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        if batch_dev["input"].dim() == 4:
            batch_dev["input"]  = batch_dev["input"].squeeze(1)
            batch_dev["target"] = batch_dev["target"].squeeze(1)
        out  = model(batch_dev)
        pred = out["logits_output"].cpu()[0]   # (C, P)
        tgt  = batch_dev["target"].cpu()[0]    # (C, P)
    
    pred_denorm = pred.float() * std_t.squeeze(0) + mean_t.squeeze(0)
    tgt_denorm  = tgt.float()  * std_t.squeeze(0) + mean_t.squeeze(0)
    pred_norm   = pred.float()
    tgt_norm    = tgt.float()

    scenario, yr, mo = sample_id_to_timestamp(batch["sample_id"][0].item(), test_data_config)
    data_cfg = test_data_config
    return {
        "prediction":      pred_denorm.numpy(),
        "target":          tgt_denorm.numpy(),
        "error":           np.abs(pred_denorm.numpy() - tgt_denorm.numpy()),
        "prediction_norm": pred_norm.numpy(),
        "target_norm":     tgt_norm.numpy(),
        "error_norm":      np.abs(pred_norm.numpy() - tgt_norm.numpy()),
        "var_names":       list(data_cfg.output_vars),
        "model_name":      data_cfg.climate_model,
        "nside":           data_cfg.nside,
        "scenario":        scenario,
        "year":            yr,
        "month":           mo,
        "timestamp_label": format_timestamp(scenario, yr, mo),
    }
