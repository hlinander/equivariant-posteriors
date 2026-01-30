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



from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.files import prepare_results
from lib.serialization import deserialize_model, DeserializeConfig

from lib.data_factory import get_factory as get_dataset_factory
import lib.data_factory as data_factory
import lib.model_factory as model_factory

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

from experiments.climate.climateset_data import ClimatesetHPConfig, ClimatesetDataHP
from experiments.climate.models.swin_hp_climateset import SwinHPClimatesetConfig, SwinHPClimateset
from experiments.climate.metrics import rmse_climate_hp

sys.stdout.flush()
if __name__ == "__main__":

    print("Registering datasets and models...")
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)

    # anv'nder mf ist'llet f;r att registrera modellen f;rst h'r, borde jag g;ra det?
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)

    device_id = ddp_setup()
    
    module_name = Path(sys.argv[1]).stem
    spec = importlib.util.spec_from_file_location(module_name, sys.argv[1])
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)

    epoch = int(sys.argv[2])
    print(f"[eval] Evaluating epoch {epoch}")
    train_run = config_file.create_config(0)
    train_run.epochs = epoch
    
    ds_train = ClimatesetDataHP(train_run.train_config.train_data_config)

    val_config = train_run.train_config.val_data_config
    if val_config is None:
        print("No validation config specified, using train config with validation split")
        val_config = train_run.train_config.train_data_config
    
    ds_val = ClimatesetDataHP(val_config)
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )


    # Deserialize with single train_run
    deser_config = DeserializeConfig(
        train_run=train_run,
        device_id=device_id,
    )

    # deser_config = DeserializeConfig(
    #     train_run=create_ensemble_config(
    #         lambda eid: config_file.create_config(eid),
    #         1,
    #     ).members[0],
    #     device_id=device_id,
    # )

    deser_model = deserialize_model(deser_config)
    if deser_model is None:
        print("Can't deserialize")
        exit(0)
    
    ensure_duck(train_run)
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

    model = deser_model.model
    model.eval()

    print("Computing RMSE...")
    rmse_results = rmse_climate_hp(model, dl_val, device_id)
    print("RMSE results", rmse_results)
    
    # Save results
    save_and_register(
        f"epoch_{deser_model.epoch:03d}_rmse_per_channel",
        rmse_results['rmse_per_channel']
    )
    
    # Get variable names from dataset metadata
    meta = ds_val.get_meta()
    output_var_names = meta['output']['names']
    
    # Get dataset class name for logging
    dataset_name = ClimatesetDataHP.__name__
    
    # Log metrics per variable
    for var_idx, var_name in enumerate(output_var_names):
        rmse_value = rmse_results['rmse_per_channel'][var_idx].item()
        print(f"  RMSE {var_name}: {rmse_value:.6f}")
        
        # insert_checkpoint_sample_metric requires:
        # model_id, step, name, dataset, sample_ids, mean, value_per_sample
        insert_checkpoint_sample_metric(
            model_id=deser_model.model_id,
            step=deser_model.epoch,  # Use epoch as step
            name=f"rmse_{var_name}",
            dataset=dataset_name,
            sample_ids=None,  # No per-sample IDs for aggregate metrics
            mean=rmse_value,
            value_per_sample=None,  # No per-sample values for aggregate metrics
        )
    
    # Log overall RMSE
    overall_rmse = rmse_results['overall_rmse'].item()
    print(f"  Overall RMSE: {overall_rmse:.6f}")
    insert_checkpoint_sample_metric(
        model_id=deser_model.model_id,
        step=deser_model.epoch,
        name="rmse_overall",
        dataset=dataset_name,
        sample_ids=None,
        mean=overall_rmse,
        value_per_sample=None,
    )
    
    # Export all metrics to staging
    print("Exporting metrics to staging...")
    export_all(train_run)
    print("Evaluation complete!")
