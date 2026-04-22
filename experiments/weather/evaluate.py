#!/usr/bin/env python
"""
Updated evaluation script using the new AnalyticsConfig-based export system.

Key changes from old version:
1. Removed db_prefix="pg." - metrics go to local DuckDB first
2. Removed attach_pg() and insert_checkpoint_pg() - use export instead
3. Added export_all() at the end to push to staging
4. Metrics flow: Local DuckDB → Staging (S3/filesystem) → Central DB

This allows the script to work in any environment (with or without Postgres access).
"""
import os
import sys
import importlib
import torch
import numpy as np
from pathlib import Path
from filelock import FileLock, Timeout

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.metric import create_metric
from lib.paths import get_lock_path, get_checkpoint_path

from experiments.weather.models.swin_hp_pangu import SwinHPPanguConfig

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.files import prepare_results
from lib.serialization import (
    deserialize_model,
    DeserializeConfig,
    load_model_from_checkpoint,
    load_checkpoint_data_config,
    load_checkpoint_train_run_json,
)

from lib.data_factory import get_factory as get_dataset_factory

# Updated imports - use new API
from lib.render_duck import (
    insert_or_update_train_run,
    insert_artifact,
    insert_model_with_model_id,
    insert_checkpoint_sample_metric,
    ensure_duck,
)

# NEW: Import export functionality
from lib.export import export_all

from lib.distributed_trainer import distributed_train

from experiments.weather.data import DataHPConfig, Climatology, DataHP
from experiments.weather.metrics import (
    anomaly_correlation_coefficient_hp,
    anomaly_correlation_coefficient_dh,
    anomaly_correlation_coefficient_dh_on_dh,
    rmse_hp,
    rmse_dh,
    rmse_dh_on_dh,
    start_gpu_keepalive,
    stop_gpu_keepalive,
    MeteorologicalData,
)


def _create_eval_data(data_config, lead_time_days):
    val_config = data_config.validation().with_lead_time_days(lead_time_days)
    ds_rmse = DataHP(val_config)
    dl_rmse = torch.utils.data.DataLoader(
        ds_rmse, batch_size=1, shuffle=False, drop_last=False,
    )
    ds_acc = Climatology(
        data_config.validation().with_lead_time_days(lead_time_days)
    )
    dl_acc = torch.utils.data.DataLoader(
        ds_acc, batch_size=1, shuffle=False, drop_last=False,
    )
    return val_config, dl_rmse, dl_acc


def _run_evaluation(model, model_id, epoch, ds_train, train_config,
                    ds_rmse_config, dl_rmse, dl_acc, device_id,
                    checkpoint_path_override=None):
    from lib.checkpoint_step import resolve_step_for_epoch
    checkpoint_path = checkpoint_path_override or get_checkpoint_path(train_config)
    step = resolve_step_for_epoch(checkpoint_path, epoch)
    if step is None:
        import math
        step = epoch * math.ceil(len(ds_train) / train_config.batch_size)
        print(f"[eval] WARNING: Could not resolve step from checkpoint analytics, "
              f"using computed step={step} for epoch {epoch}")
    else:
        print(f"[eval] Resolved epoch {epoch} -> step {step}")
    era5_meta = MeteorologicalData()
    model.eval()

    print("ACC")
    if ds_rmse_config.driscoll_healy:
        acc_res_on_dh = anomaly_correlation_coefficient_dh_on_dh(
            model, dl_acc, device_id
        )
        # start_gpu_keepalive()
        acc_res = anomaly_correlation_coefficient_dh(model, dl_acc, device_id)
        # stop_gpu_keepalive()

        for var_idx, var_data in enumerate(acc_res_on_dh.acc_surface):
            insert_checkpoint_sample_metric(
                model_id,
                step,
                f"dh$acc_surface_{era5_meta.surface.names[var_idx]}.{ds_rmse_config.lead_time_days}d",
                ds_rmse_config.short_name(),
                [],
                var_data.item(),
                [],
            )
        for var_idx, var_data in enumerate(acc_res_on_dh.acc_upper):
            for level, value in zip(era5_meta.upper.levels, var_data.cpu().numpy()):
                var_name = f"dh$acc_upper_{era5_meta.upper.names[var_idx]}_{int(level)}.{ds_rmse_config.lead_time_days}d"
                insert_checkpoint_sample_metric(
                    model_id,
                    step,
                    var_name,
                    ds_rmse_config.short_name(),
                    [],
                    value.item(),
                    [],
                )
    else:
        acc_res = anomaly_correlation_coefficient_hp(model, dl_acc, device_id)

    print("[eval] rmse")
    if ds_rmse_config.driscoll_healy:
        # start_gpu_keepalive()
        rmse_res = rmse_dh(model, dl_rmse, device_id)
        # stop_gpu_keepalive()
        rmse_res_on_dh = rmse_dh_on_dh(model, dl_rmse, device_id)

        for var_idx, var_data in enumerate(rmse_res_on_dh.mean_surface):
            insert_checkpoint_sample_metric(
                model_id,
                step,
                f"dh$rmse_surface_{era5_meta.surface.names[var_idx]}.{ds_rmse_config.lead_time_days}d",
                ds_rmse_config.short_name(),
                [],
                var_data.item(),
                [],
            )
        for var_idx, var_data in enumerate(rmse_res_on_dh.mean_upper):
            for level, value in zip(era5_meta.upper.levels, var_data.cpu().numpy()):
                var_name = f"dh$rmse_upper_{era5_meta.upper.names[var_idx]}_{int(level)}.{ds_rmse_config.lead_time_days}d"
                insert_checkpoint_sample_metric(
                    model_id,
                    step,
                    var_name,
                    ds_rmse_config.short_name(),
                    [],
                    value.item(),
                    [],
                )
    else:
        rmse_res = rmse_hp(model, dl_rmse, device_id)

    for var_idx, var_data in enumerate(rmse_res.mean_surface):
        insert_checkpoint_sample_metric(
            model_id,
            epoch * len(ds_train),
            f"rmse_surface_{era5_meta.surface.names[var_idx]}.{ds_rmse_config.lead_time_days}d",
            ds_rmse_config.short_name(),
            [],
            var_data.item(),
            [],
        )
    for var_idx, var_data in enumerate(rmse_res.mean_upper):
        for level, value in zip(era5_meta.upper.levels, var_data.cpu().numpy()):
            var_name = f"rmse_upper_{era5_meta.upper.names[var_idx]}_{int(level)}.{ds_rmse_config.lead_time_days}d"
            insert_checkpoint_sample_metric(
                model_id,
                step,
                var_name,
                ds_rmse_config.short_name(),
                [],
                value.item(),
                [],
            )
    for var_idx, var_data in enumerate(acc_res.acc_surface):
        insert_checkpoint_sample_metric(
            model_id,
            epoch * len(ds_train),
            f"acc_surface_{era5_meta.surface.names[var_idx]}.{ds_rmse_config.lead_time_days}d",
            ds_rmse_config.short_name(),
            [],
            var_data.item(),
            [],
        )
    for var_idx, var_data in enumerate(acc_res.acc_upper):
        for level, value in zip(era5_meta.upper.levels, var_data.cpu().numpy()):
            var_name = f"acc_upper_{era5_meta.upper.names[var_idx]}_{int(level)}.{ds_rmse_config.lead_time_days}d"
            insert_checkpoint_sample_metric(
                model_id,
                step,
                var_name,
                ds_rmse_config.short_name(),
                [],
                value.item(),
                [],
            )

    print("[eval] Evaluation complete!")


def _export(train_run):
    print("[eval] Exporting metrics to staging...")
    exported_paths = export_all(train_run)
    if exported_paths:
        print(f"[eval] ✓ Exported {len(exported_paths)} files to staging")
    else:
        print("[eval] No new metrics to export")


def evaluate_weather(create_config, epoch, lead_time_days, ensemble_id=0):
    device_id = ddp_setup()
    print(f"Lead time {lead_time_days}d")
    train_run = create_config(0, 10)

    ds_train = DataHP(train_run.train_config.train_data_config)
    ds_rmse_config, dl_rmse, dl_acc = _create_eval_data(
        train_run.train_config.train_data_config, lead_time_days
    )

    print(f"[eval] Epoch {epoch}")
    deser_config = DeserializeConfig(
        train_run=create_config(ensemble_id, epoch),
        device_id=device_id,
    )
    deser_model = deserialize_model(deser_config)
    if deser_model is None:
        print("Can't deserialize")
        exit(0)

    ensure_duck(train_run)
    insert_model_with_model_id(train_run, deser_model.model_id)
    insert_or_update_train_run(train_run, deser_model.model_id)

    _run_evaluation(
        deser_model.model, deser_model.model_id, epoch, ds_train,
        train_run.train_config,
        ds_rmse_config, dl_rmse, dl_acc, device_id,
    )
    _export(train_run)


def evaluate_weather_from_checkpoint(checkpoint_hash, epoch, lead_time_days):
    """Evaluate a model loaded by checkpoint hash.

    Reconstructs model, data config, and metric reporting from the saved
    checkpoint — no config .py file needed.
    """
    from lib.render_duck import setup_duck_from_checkpoint

    device_id = ddp_setup()
    print(f"Lead time {lead_time_days}d, checkpoint {checkpoint_hash}")

    data_config = load_checkpoint_data_config(checkpoint_hash)
    ds_train = DataHP(data_config)
    ds_rmse_config, dl_rmse, dl_acc = _create_eval_data(
        data_config, lead_time_days
    )

    print(f"[eval] Epoch {epoch}")
    deser_model = load_model_from_checkpoint(
        checkpoint_hash, device_id, epoch=epoch
    )
    if deser_model is None:
        print("Can't deserialize from checkpoint hash")
        exit(0)

    saved_json = load_checkpoint_train_run_json(checkpoint_hash)
    setup_duck_from_checkpoint(
        checkpoint_hash, deser_model.model_id, saved_json
    )

    from lib.compute_env import env as compute_env
    from types import SimpleNamespace
    batch_size = saved_json.get("train_config", {}).get("batch_size", 1)
    # Build a minimal train_config with batch_size for fallback step computation.
    # _run_evaluation will use the checkpoint_hash to find the checkpoint path.
    tc = SimpleNamespace(batch_size=batch_size, _checkpoint_hash=checkpoint_hash)
    _run_evaluation(
        deser_model.model, deser_model.model_id, epoch, ds_train,
        tc,
        ds_rmse_config, dl_rmse, dl_acc, device_id,
        checkpoint_path_override=compute_env().paths.checkpoints / f"checkpoint_{checkpoint_hash}",
    )


def load_create_config(module_file_path):
    module_name = Path(module_file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    return config_file.create_config


if __name__ == "__main__":
    device_id = ddp_setup()

    create_config = load_create_config(sys.argv[1])
    epoch = int(sys.argv[2])
    lead_time_days = int(os.environ.get("LEADTIME", "1"))
    evaluate_weather(create_config, epoch, lead_time_days)
