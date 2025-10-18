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
from lib.paths import get_lock_path

from experiments.weather.models.swin_hp_pangu import SwinHPPanguConfig

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.files import prepare_results
from lib.serialization import deserialize_model, DeserializeConfig

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
    MeteorologicalData,
)


if __name__ == "__main__":
    device_id = ddp_setup()

    module_name = Path(sys.argv[1]).stem
    spec = importlib.util.spec_from_file_location(module_name, sys.argv[1])
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    train_run = config_file.create_config(0, 10)

    lead_time_days = int(os.environ.get("LEADTIME", "1"))
    print(f"Lead time {lead_time_days}d")

    ds_train = DataHP(train_run.train_config.train_data_config)
    ds_rmse_config = (
        train_run.train_config.train_data_config.validation().with_lead_time_days(
            lead_time_days
        )
    )
    ds_rmse = DataHP(ds_rmse_config)
    dl_rmse = torch.utils.data.DataLoader(
        ds_rmse,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    ds_acc = Climatology(
        train_run.train_config.train_data_config.validation().with_lead_time_days(
            lead_time_days
        )
    )
    dl_acc = torch.utils.data.DataLoader(
        ds_acc,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    era5_meta = MeteorologicalData()

    epoch = int(sys.argv[2])

    print(f"[eval] Epoch {epoch}")
    deser_config = DeserializeConfig(
        train_run=create_ensemble_config(
            lambda eid: config_file.create_config(eid, epoch),
            1,
        ).members[0],
        device_id=device_id,
    )
    deser_model = deserialize_model(deser_config)
    if deser_model is None:
        print("Can't deserialize")
        exit(0)

    insert_model_with_model_id(train_run, deser_model.model_id)

    result_path = prepare_results(
        f"{train_run.serialize_human()["run_id"]}",
        train_run,
    )

    def save_and_register(name, array):
        path = result_path / f"{name}.npy"
        np.save(
            path,
            array.detach().cpu().float().numpy(),
        )
        insert_artifact(deser_model.model_id, name, path, ".npy")

    # Initialize local DuckDB (no Postgres connection needed)
    ensure_duck(train_run)

    model = deser_model.model
    model.eval()

    print("ACC")
    if ds_rmse_config.driscoll_healy:
        acc_res_on_dh = anomaly_correlation_coefficient_dh_on_dh(
            model, dl_acc, device_id
        )
        acc_res = anomaly_correlation_coefficient_dh(model, dl_acc, device_id)

        # CHANGED: Removed db_prefix="pg." - metrics go to local DuckDB
        for var_idx, var_data in enumerate(acc_res_on_dh.acc_surface):
            insert_checkpoint_sample_metric(
                deser_model.model_id,
                epoch * len(ds_train),
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
                    deser_model.model_id,
                    epoch * len(ds_train),
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
        rmse_res = rmse_dh(model, dl_rmse, device_id)
        rmse_res_on_dh = rmse_dh_on_dh(model, dl_rmse, device_id)

        # CHANGED: Removed db_prefix="pg."
        for var_idx, var_data in enumerate(rmse_res_on_dh.mean_surface):
            insert_checkpoint_sample_metric(
                deser_model.model_id,
                epoch * len(ds_train),
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
                    deser_model.model_id,
                    epoch * len(ds_train),
                    var_name,
                    ds_rmse_config.short_name(),
                    [],
                    value.item(),
                    [],
                )
    else:
        rmse_res = rmse_hp(model, dl_rmse, device_id)

    # CHANGED: Removed db_prefix="pg."
    for var_idx, var_data in enumerate(rmse_res.mean_surface):
        insert_checkpoint_sample_metric(
            deser_model.model_id,
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
                deser_model.model_id,
                epoch * len(ds_train),
                var_name,
                ds_rmse_config.short_name(),
                [],
                value.item(),
                [],
            )
    for var_idx, var_data in enumerate(acc_res.acc_surface):
        insert_checkpoint_sample_metric(
            deser_model.model_id,
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
                deser_model.model_id,
                epoch * len(ds_train),
                var_name,
                ds_rmse_config.short_name(),
                [],
                value.item(),
                [],
            )

    # NEW: Export metrics to staging using AnalyticsConfig
    # This replaces the old sync(train_run) call
    print("[eval] Exporting metrics to staging...")
    exported_paths = export_all(train_run)
    if exported_paths:
        print(f"[eval] ✓ Exported {len(exported_paths)} files to staging")
    else:
        print("[eval] No new metrics to export")

    print("[eval] Evaluation complete!")
