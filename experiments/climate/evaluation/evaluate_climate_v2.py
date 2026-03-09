#!/usr/bin/env python
"""
Updated evaluation script for climate model variants.

Key fix: Uses SLURM_ARRAY_TASK_ID (or command line arg) to select the right
model variant when calling create_config().
"""
import os
import sys
import importlib
import torch
import numpy as np
from pathlib import Path
from filelock import FileLock, Timeout
import copy

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

from lib.render_duck import (
    insert_or_update_train_run,
    insert_artifact,
    insert_model_with_model_id,
    insert_checkpoint_sample_metric,
    ensure_duck,
)
from experiments.climate.models.swin_hp_climateset_seq import (
    SwinHPClimatesetSeqConfig,
    SwinHPClimatesetSeq,
)

from lib.export import export_all
from lib.distributed_trainer import distributed_train

from experiments.climate.climateset_data_hp import ClimatesetHPConfig, ClimatesetDataHP
from experiments.climate.models.swin_hp_climateset import SwinHPClimatesetConfig, SwinHPClimateset
from experiments.climate.metrics import rmse_climate_hp
from experiments.climate.climateset_data_hp import load_training_stats_from_config

sys.stdout.flush()

if __name__ == "__main__":
    device_id = ddp_setup()

    print("Registering datasets and models...")
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)
    mf.register(SwinHPClimatesetSeqConfig, SwinHPClimatesetSeq)
    # Load config module
    module_name = Path(sys.argv[1]).stem
    spec = importlib.util.spec_from_file_location(module_name, sys.argv[1])
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    
    # Get variant index - try SLURM first, then command line, then default to 0
    if len(sys.argv) > 3:
        variant_idx = int(sys.argv[3])
        print(f"[eval] Using variant_idx from command line: {variant_idx}")
    else:
        variant_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        print(f"[eval] Using variant_idx from SLURM_ARRAY_TASK_ID: {variant_idx}")
    
    # Create config with the correct variant/ensemble_id
    train_run = config_file.create_config(ensemble_id=variant_idx)

    epoch = int(sys.argv[2])
    print(f"[eval] Evaluating variant {variant_idx}, epoch {epoch}")
    train_run.epochs = epoch
    print(f"Nside: {train_run.train_config.train_data_config.nside}, ")
    deser_config = DeserializeConfig(
        train_run=train_run,
        device_id=device_id,
    )

    # # working config
    # config = ClimatesetHPConfig(
    #     nside=32,
    #     climate_model="CAS-ESM2-0",
    #     ensemble="r3i1p1f1",
    #     input_vars=["CH4", "SO2", "CO2", "BC"],
    #     output_vars=["tas", "pr"],
    #     scenarios=["ssp126", "ssp370", "ssp585"],
    #     years="2015-2100",
    #     seq_len=1,
    #     normalized=True,
    #     split="train",   # fixed: was "full" (invalid)
    #     cache=True,
    #     val_fraction=0.1,
    #     random_seed=42,
    # )
    # bugtest = ClimatesetDataHP(config)
    # sample = bugtest[0]
    # #print("Sample keys:", sample.keys())
    # #print(f"Sample input shape: {sample['input'].shape}, target shape: {sample['target'].shape}")
    # #print(f"Sample input (norm): {sample['input']}, \n target (norm): {sample['target']}")
    # # Create test dataset

    # train_data_config = copy.deepcopy(train_run.train_config.train_data_config)
    # train_ds = ClimatesetDataHP(train_data_config)

    test_data_config = copy.deepcopy(train_run.train_config.train_data_config)
    test_data_config.scenarios = ["ssp245"]
    test_data_config.split = "test"
    test_ds = ClimatesetDataHP(test_data_config)
    #test_sample = test_ds[0]
    #print("Test sample keys:", test_sample.keys())
    #print(f"test sample input {test_sample['input']}, \n target {test_sample['target']}")
    stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    #print("Loaded training stats:", stats)
    test_ds.set_normalization_stats(**stats)
    test_sample = test_ds[0]
    #print("Test sample keys:", test_sample.keys())
    #print(f"test sample input {test_sample['input']}, \n target {test_sample['target']}")
    # Load normalization stats from training
    #stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    #test_ds.set_normalization_stats(**stats)
    
    # train_dl = torch.utils.data.DataLoader(
    #     train_ds,
    #     batch_size=12,  # Same as training batch size
    #     shuffle=False,
    #     drop_last=False,
    # )
    # Create dataloader for evaluation
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=12,  # Same as training batch size
        shuffle=False,
        drop_last=False,
    )

    # Deserialize model
    deser_model = deserialize_model(deser_config)
    if deser_model is None:
        print("[eval] ERROR: Can't deserialize model")
        print(f"[eval] Check that checkpoint exists for variant {variant_idx}, epoch {epoch}")
        exit(1)
    
    #print(f"[eval] Successfully loaded model_id={deser_model.model_id}")
    insert_model_with_model_id(train_run, deser_model.model_id)

    result_path = prepare_results(
        f"{train_run.serialize_human()['run_id']}",
        train_run,
    )

    def save_and_register(name, array):
        path = result_path / f"{name}.npy"
        np.save(
            path,
            array.detach().cpu().float().numpy(),
        )
        insert_artifact(deser_model.model_id, name, path, ".npy")

    #ensure_duck(train_run)

    model = deser_model.model
    model.eval()

    print("[eval] Computing RMSE...")
    #print(f"[eval] Loaded stats structure: {stats.keys()}")
    
    # Get the normalization stats for denormalization
    # Stats are nested under 'output_stats'
    if 'output_stats' in stats:
        output_stats = {
            'mean': stats['output_stats']['mean'],
            'std': stats['output_stats']['std']
        }
        #print(f"[eval] Extracted output_stats from nested structure")
    else:
        # Fallback for different structure
        output_stats = {
            'mean': stats.get('output_mean', stats.get('mean')),
            'std': stats.get('output_std', stats.get('std'))
        }
        #print(f"[eval] Extracted output_stats from flat structure")
    
    #print(f"[eval] Using normalization stats - mean shape: {np.array(output_stats['mean']).shape}, std shape: {np.array(output_stats['std']).shape}")
    
    #rmse_results_train = rmse_climate_hp(model, train_dl, device_id, output_stats)
    rmse_results = rmse_climate_hp(model, test_dl, device_id, output_stats)
      # Temporary exit to check RMSE computation
    #print(f"[eval] RMSE results on training set: {rmse_results_train['rmse_per_channel']}")
    print(f"[eval] RMSE results: {rmse_results['rmse_per_channel']}")
    
    # Save results
    save_and_register(
        f"epoch_{deser_model.epoch:03d}_rmse_per_channel",
        rmse_results['rmse_per_channel']
    )
    
    # Get dataset class name and output variable names
    dataset_name = ClimatesetDataHP.__name__
    output_var_names = train_run.train_config.train_data_config.output_vars
    
    # Log metrics per variable
    for var_idx, var_name in enumerate(output_var_names):
        rmse_value = rmse_results['rmse_per_channel'][var_idx].item()
        print(f"[eval]   RMSE {var_name}: {rmse_value:.6f}")
        
        insert_checkpoint_sample_metric(
            model_id=deser_model.model_id,
            step=deser_model.epoch,
            name=f"rmse_{var_name}",
            dataset=dataset_name,
            sample_ids=None,
            mean=rmse_value,
            value_per_sample=None,
        )
    
    # Log overall RMSE
    overall_rmse = rmse_results['overall_rmse'].item()
    print(f"[eval]   Overall RMSE: {overall_rmse:.6f}")
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
    print("[eval] Exporting metrics to staging...")
    export_all(train_run)
    print(f"[eval] Evaluation complete for variant {variant_idx}!")