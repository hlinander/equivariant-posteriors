"""
Evaluates a trained NoHP (grid-based) climate model on its **training split** to
diagnose overfitting.  Metrics are stored under dataset="ClimatesetData_trainset" so
they never collide with test-split results (dataset="ClimatesetData") in DuckDB.

Diagnostic metrics saved per model / epoch:
  - rmse_latw_{var}        : per-variable latitude-weighted RMSE (normalised units)
  - rmse_latw_overall      : mean lat-weighted RMSE across all output variables
  - rmse_latw_t{t}_overall : lat-weighted RMSE at each sequence timestep t=0..T-1
                             (only inserted for seq-to-seq models; reveals whether
                             error accumulates through the sequence)
  - pred_std_overall       : std of predictions (near 0 → model predicts climatological mean)
  - target_std_overall     : std of targets (reference scale)

Usage (mirrors evaluate_climate_nohp.py):
  CONFIG=<path/to/train_climax_nohp.py> python evaluate_climate_nohp_trainset.py
  SLURM_ARRAY_TASK_ID=3  # variant index
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

from lib.render_duck import (
    insert_or_update_train_run,
    insert_model_with_model_id,
    insert_checkpoint_sample_metric,
    ensure_duck,
)

from lib.export import export_all

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
from experiments.climate.adapted_climateset_baselines.adapted_models.climax.climax_module import (
    ClimaXConfig,
    ClimaX,
)
from experiments.climate.models.GRU_wrapper import GRUTemporalWrapperConfig, GRUTemporalWrapper
from experiments.climate.evaluation.metrics import rmse_climate_nohp, LLweighted_RMSE_Climax

sys.stdout.flush()

# Dataset name used in DuckDB — must differ from ClimatesetData.__name__
_DATASET_NAME = "ClimatesetData_trainset"


def _compute_per_timestep_rmse(model, dataloader, device_id):
    """
    Collect (N, T, C, H, W) predictions and targets and return per-timestep
    lat-weighted RMSE averaged over output channels.

    Returns None if the model output has no time dimension (non-seq-to-seq model).
    """
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
            output = model(batch_device)
            all_predictions.append(output["logits_output"].cpu())
            all_targets.append(batch_device["target"].cpu())

    preds = torch.cat(all_predictions, dim=0)
    tgts  = torch.cat(all_targets,     dim=0)

    if preds.dim() != 5:
        # Not seq-to-seq; per-timestep RMSE is not applicable.
        return None

    N, T, C, H, W = preds.shape
    preds_np = preds.numpy()
    tgts_np  = tgts.numpy()

    per_timestep = []
    for t in range(T):
        # Average lat-weighted RMSE across output channels at this timestep
        channel_rmses = [
            LLweighted_RMSE_Climax(preds_np[:, t, c, :, :], tgts_np[:, t, c, :, :])
            for c in range(C)
        ]
        per_timestep.append(float(np.mean(channel_rmses)))

    return per_timestep  # list of T floats


def _compute_pred_collapse(model, dataloader, device_id):
    """Return std of predictions and targets (collapsed over all dimensions)."""
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
            output = model(batch_device)
            all_predictions.append(output["logits_output"].cpu())
            all_targets.append(batch_device["target"].cpu())

    preds = torch.cat(all_predictions, dim=0)
    tgts  = torch.cat(all_targets,     dim=0)
    return preds.std().item(), tgts.std().item()


def evaluate_climate_trainset(create_config, epoch, variant_idx=0):
    device_id = ddp_setup()

    print("Registering datasets and models...")
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetConfig, ClimatesetData)
    mf = model_factory.get_factory()
    mf.register(ClimaXConfig, ClimaX)
    mf.register(UNetConfig, UNet)
    mf.register(CNNLSTMConfig, CNNLSTM_ClimateBench)
    mf.register(GRUTemporalWrapperConfig, GRUTemporalWrapper)

    print(f"[eval-trainset] Evaluating variant {variant_idx}, epoch {epoch} on TRAINING split")
    train_run = create_config(ensemble_id=variant_idx)
    train_run.epochs = epoch

    # Evaluate on the training split directly (config already has split="train")
    train_ds = ClimatesetData(train_run.train_config.train_data_config)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )

    # Normalization stats (always derived from training config)
    stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    output_stats = stats["output_stats"]

    # Deserialize checkpoint
    deser_config = DeserializeConfig(train_run=train_run, device_id=device_id)
    deser_model = deserialize_model(deser_config)
    if deser_model is None:
        print(f"[eval-trainset] Can't deserialize variant {variant_idx}, epoch {epoch}")
        return

    # Register in DuckDB
    ensure_duck(train_run)
    insert_or_update_train_run(train_run, deser_model.model_id)
    insert_model_with_model_id(train_run, deser_model.model_id)

    model = deser_model.model
    model.eval()

    step = epoch * len(train_ds)

    # ── Main lat-weighted RMSE ─────────────────────────────────────────────────
    print("[eval-trainset] Computing lat-weighted RMSE on training split...")
    rmse_results = rmse_climate_nohp(model, train_dl, device_id, output_stats)

    output_var_names = train_run.train_config.train_data_config.output_vars

    for var_idx, var_name in enumerate(output_var_names):
        rmse_value = float(rmse_results["rmse_per_channel"][var_idx])
        print(f"[eval-trainset]   RMSE latw {var_name}: {rmse_value:.6f}")
        insert_checkpoint_sample_metric(
            deser_model.model_id,
            step,
            f"rmse_latw_{var_name}",
            _DATASET_NAME,
            [],
            rmse_value,
            [],
        )

    overall_rmse = float(rmse_results["overall_rmse"])
    print(f"[eval-trainset]   Overall lat-weighted RMSE: {overall_rmse:.6f}")
    insert_checkpoint_sample_metric(
        deser_model.model_id,
        step,
        "rmse_latw_overall",
        _DATASET_NAME,
        [],
        overall_rmse,
        [],
    )

    # ── Per-timestep RMSE (seq-to-seq models only) ────────────────────────────
    print("[eval-trainset] Computing per-timestep RMSE...")
    per_timestep = _compute_per_timestep_rmse(model, train_dl, device_id)
    if per_timestep is not None:
        for t, val in enumerate(per_timestep):
            print(f"[eval-trainset]   t={t:02d}  RMSE latw: {val:.6f}")
            insert_checkpoint_sample_metric(
                deser_model.model_id,
                step,
                f"rmse_latw_t{t}_overall",
                _DATASET_NAME,
                [],
                val,
                [],
            )
    else:
        print("[eval-trainset]   (skipped — model is not seq-to-seq)")

    # ── Prediction collapse diagnostic ────────────────────────────────────────
    print("[eval-trainset] Computing prediction std (collapse check)...")
    pred_std, target_std = _compute_pred_collapse(model, train_dl, device_id)
    print(f"[eval-trainset]   pred_std={pred_std:.6f}  target_std={target_std:.6f}")
    insert_checkpoint_sample_metric(
        deser_model.model_id, step, "pred_std_overall", _DATASET_NAME, [], pred_std, []
    )
    insert_checkpoint_sample_metric(
        deser_model.model_id, step, "target_std_overall", _DATASET_NAME, [], target_std, []
    )

    print("[eval-trainset] Exporting metrics to staging...")
    export_all(train_run)
    print(f"[eval-trainset] Done for variant {variant_idx}, epoch {epoch}!")


def load_create_config(module_file_path):
    module_name = Path(module_file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    return config_file.create_config


if __name__ == "__main__":
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0").strip()
    variant_idx = int(task_id) if task_id else 0

    create_config = load_create_config(os.environ["CONFIG"])
    c = create_config(0)
    epochs = list(range(0, c.epochs + 1, c.keep_nth_epoch_checkpoints))

    for epoch in epochs:
        evaluate_climate_trainset(create_config, epoch, variant_idx=variant_idx)
