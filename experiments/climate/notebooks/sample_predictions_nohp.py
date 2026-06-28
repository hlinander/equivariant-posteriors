"""
Lightweight helper to extract raw predictions and targets from a trained nohp (lat/lon grid)
climate model. Mirrors sample_predictions.py but registers nohp factories.

Data shape
----------
  predictions / targets: (N, T, C, H, W)  for seq_len > 1  (T = seq_len months)
                         (N, C, H, W)      for seq_len == 1
  H = longitude axis, W = latitude axis (matching ClimatesetData convention).
  Transpose spatially for standard lat-on-y plotting: arr.transpose(0, ..., -1, -2).

Two entry points
----------------
get_sample_predictions_nohp()          — load the first n_batches (exploration)
get_predictions_for_timestamp_nohp()   — load one specific (year, month) sample
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

from experiments.climate.data.climateset_data_no_hp import (
    ClimatesetConfig,
    ClimatesetData,
    load_training_stats_from_config,
)
from experiments.climate.adapted_climateset_baselines.unet import UNetConfig, UNet
from experiments.climate.adapted_climateset_baselines.cnn_lstm import (
    CNNLSTMConfig,
    CNNLSTM_ClimateBench,
)
from experiments.climate.adapted_climateset_baselines.climax.climax_module import (
    ClimaXConfig,
    ClimaX,
)
from experiments.climate.models.GRU_wrapper import GRUTemporalWrapperConfig, GRUTemporalWrapper


def _register_factories():
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetConfig, ClimatesetData)
    mf = model_factory.get_factory()
    mf.register(ClimaXConfig, ClimaX)
    mf.register(UNetConfig, UNet)
    mf.register(CNNLSTMConfig, CNNLSTM_ClimateBench)
    mf.register(GRUTemporalWrapperConfig, GRUTemporalWrapper)


def get_sample_predictions_nohp(create_config, epoch, variant_idx=0, n_batches=2):
    """
    Load a trained nohp climate model and run inference on the first n_batches of test data.

    Args:
        create_config: Callable accepting ensemble_id keyword; returns a TrainRun.
        epoch:         Checkpoint epoch to load.
        variant_idx:   Ensemble / seed index (passed as ensemble_id).
        n_batches:     How many batches to process (batch_size=4 → 4*n_batches sequences).

    Returns dict with:
        predictions        np.ndarray  (N, T, C, H, W) if seq_len>1 else (N, C, H, W) — normalized
        targets            np.ndarray  same shape — normalized
        predictions_denorm np.ndarray  same shape — physical units
        targets_denorm     np.ndarray  same shape — physical units
        errors             np.ndarray  same shape — abs error (denorm)
        sample_ids         list[int]   raw array offsets, one per sequence in N
        seq_len            int         months per sequence (12 for CNN-LSTM, 1 for single-step)
        lats               np.ndarray  latitude values from the dataset (or None)
        var_names          list[str]
        model_name         str
        test_data_config   ClimatesetConfig
        epoch              int
    """
    _register_factories()
    device_id = ddp_setup()

    train_run = create_config(ensemble_id=variant_idx)
    train_run.epochs = epoch

    test_data_config = copy.deepcopy(train_run.train_config.train_data_config)
    test_data_config.scenarios = ["ssp245"]
    test_data_config.split = "test"
    test_ds = ClimatesetData(test_data_config)

    stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    test_ds.set_normalization_stats(**stats)

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=4, shuffle=False, drop_last=False
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
            out = model(batch_dev)
            all_preds.append(out["logits_output"].cpu())
            all_tgts.append(batch_dev["target"].cpu())

    preds = torch.cat(all_preds, dim=0)
    tgts  = torch.cat(all_tgts,  dim=0)

    mean_t = torch.tensor(stats["output_stats"]["mean"], dtype=torch.float32)
    std_t  = torch.tensor(stats["output_stats"]["std"],  dtype=torch.float32)

    # mean_t/std_t shape: (1, C, 1, 1) — broadcasts with (N, C, H, W) or (N, T, C, H, W)
    preds_denorm = preds.float() * std_t + mean_t
    tgts_denorm  = tgts.float()  * std_t + mean_t

    data_cfg = train_run.train_config.train_data_config
    return {
        "predictions":        preds.float().numpy(),
        "targets":            tgts.float().numpy(),
        "predictions_denorm": preds_denorm.numpy(),
        "targets_denorm":     tgts_denorm.numpy(),
        "errors":             np.abs(preds_denorm.numpy() - tgts_denorm.numpy()),
        "sample_ids":         all_sample_ids,
        "seq_len":            test_data_config.seq_len,
        "lats":               getattr(test_ds, "lats", None),
        "var_names":          list(data_cfg.output_vars),
        "model_name":         data_cfg.climate_model,
        "test_data_config":   test_data_config,
        "epoch":              epoch,
    }


def _build_model_and_test_ds_nohp(create_config, epoch, variant_idx):
    """Shared setup: load model + test dataset."""
    _register_factories()
    device_id = ddp_setup()

    train_run = create_config(ensemble_id=variant_idx)
    train_run.epochs = epoch

    test_data_config = copy.deepcopy(train_run.train_config.train_data_config)
    test_data_config.scenarios = ["ssp245"]
    test_data_config.split = "test"
    test_ds = ClimatesetData(test_data_config)

    stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    test_ds.set_normalization_stats(**stats)

    deser_model = deserialize_model(DeserializeConfig(train_run=train_run, device_id=device_id))
    if deser_model is None:
        raise RuntimeError(
            f"Could not load checkpoint for epoch={epoch}, variant={variant_idx}."
        )
    deser_model.model.eval()
    return deser_model.model, test_ds, stats, test_data_config, device_id


def get_predictions_for_timestamp_nohp(create_config, epoch, year, month, variant_idx=0):
    """
    Load one specific (year, month) sample from the ssp245 test set.

    Use this to compare the exact same timestep against the HP notebook:
        # HP notebook
        r_hp   = get_predictions_for_timestamp(hp_curried,   epoch, year=2040, month=4)
        # nohp notebook
        r_nohp = get_predictions_for_timestamp_nohp(nohp_curried, epoch, year=2040, month=4)

    Returns dict with:
        prediction        np.ndarray (C, H, W) — denormalized, single timestep
        target            np.ndarray (C, H, W) — denormalized
        error             np.ndarray (C, H, W) — abs error
        lats              np.ndarray  latitude values (W axis), or None
        var_names         list[str]
        model_name        str
        scenario          str
        year              int
        month             int
        timestamp_label   str   e.g. "ssp245 · Apr 2040"
    """
    model, test_ds, stats, test_data_config, device_id = _build_model_and_test_ds_nohp(
        create_config, epoch, variant_idx
    )

    dataloader_idx, seq_pos, raw_offset = find_sample_for_timestamp(year, month, test_data_config)
    subset = torch.utils.data.Subset(test_ds, [dataloader_idx])
    dl = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

    mean_t = torch.tensor(stats["output_stats"]["mean"], dtype=torch.float32)
    std_t  = torch.tensor(stats["output_stats"]["std"],  dtype=torch.float32)

    with torch.no_grad():
        batch = next(iter(dl))
        batch_dev = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        out  = model(batch_dev)
        pred = out["logits_output"].cpu()[0]   # (T, C, H, W) or (C, H, W)
        tgt  = batch_dev["target"].cpu()[0]    # same

    # Verify the batch actually starts at the sequence we expect, then extract seq_pos.
    # batch["sample_id"] is the start index of the 12-month window; adding seq_pos gives
    # the absolute timestep, which must equal raw_offset.
    seq_start = batch["sample_id"][0].item()
    if int(seq_start) + seq_pos != raw_offset:
        raise RuntimeError(
            f"Timestep alignment error: sequence starts at {seq_start}, "
            f"seq_pos={seq_pos}, but expected raw_offset={raw_offset}. "
            f"Got absolute timestep {seq_start + seq_pos}."
        )

    # Extract the requested month from the sequence
    if pred.dim() == 4:    # (T, C, H, W)
        pred = pred[seq_pos]   # (C, H, W)
        tgt  = tgt[seq_pos]

    pred_denorm = pred.float() * std_t.squeeze(0) + mean_t.squeeze(0)
    tgt_denorm  = tgt.float()  * std_t.squeeze(0) + mean_t.squeeze(0)
    pred_norm   = pred.float()
    tgt_norm    = tgt.float()

    scenario, yr, mo = sample_id_to_timestamp(raw_offset, test_data_config)
    data_cfg = test_data_config
    return {
        "prediction":      pred_denorm.numpy(),
        "target":          tgt_denorm.numpy(),
        "error":           np.abs(pred_denorm.numpy() - tgt_denorm.numpy()),
        "prediction_norm": pred_norm.numpy(),
        "target_norm":     tgt_norm.numpy(),
        "error_norm":      np.abs(pred_norm.numpy() - tgt_norm.numpy()),
        "lats":            getattr(test_ds, "lats", None),
        "var_names":       list(data_cfg.output_vars),
        "model_name":      data_cfg.climate_model,
        "scenario":        scenario,
        "year":            yr,
        "month":           mo,
        "timestamp_label": format_timestamp(scenario, yr, mo),
    }
