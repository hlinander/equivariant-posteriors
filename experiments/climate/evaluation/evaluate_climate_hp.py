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

from experiments.climate.data.climateset_data_hp import ClimatesetHPConfig, ClimatesetDataHP
from experiments.climate.models.swin_hp_climateset import SwinHPClimatesetConfig, SwinHPClimateset
from experiments.climate.models.GRU_wrapper import GRUTemporalWrapperConfig, GRUTemporalWrapper
from experiments.climate.evaluation.metrics import rmse_climate_hp
from experiments.climate.data.climateset_data_hp import load_training_stats_from_config

sys.stdout.flush()

def evaluate_climate(create_config, epoch, variant_idx=0):
    device_id = ddp_setup()

    print("Registering datasets and models...")
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)
    mf.register(SwinHPClimatesetSeqConfig, SwinHPClimatesetSeq)
    mf.register(GRUTemporalWrapperConfig, GRUTemporalWrapper)

    print(f"[eval] Evaluating variant {variant_idx}, epoch {epoch}")
    train_run = create_config(ensemble_id=variant_idx)
    train_run.epochs = epoch

    train_ds = ClimatesetDataHP(train_run.train_config.train_data_config)
    test_data_config = copy.deepcopy(train_run.train_config.train_data_config)  # copy the config, not the dataset
    test_data_config.scenarios = ["ssp245"]
    test_data_config.split = "test"
    test_ds = ClimatesetDataHP(test_data_config)
        
    # Load normalization stats from training and set in test set
    stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    test_ds.set_normalization_stats(**stats)
    
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=12,
        shuffle=False,
        drop_last=False,
    )

    # Deserialize model
    deser_config = DeserializeConfig(train_run=train_run, device_id=device_id)
    deser_model = deserialize_model(deser_config)
    if deser_model is None:
        print(f"[eval] Can't deserialize variant {variant_idx}, epoch {epoch}")
        return

    # Register in DuckDB
    ensure_duck(train_run)
    insert_or_update_train_run(train_run, deser_model.model_id)
    insert_model_with_model_id(train_run, deser_model.model_id)

    model = deser_model.model
    model.eval()

    # Normalization stats for denormalization
    output_stats = {
        'mean': stats['output_stats']['mean'],
        'std': stats['output_stats']['std']
    }

    print("[eval] Computing RMSE...")
    rmse_results = rmse_climate_hp(model, test_dl, device_id, output_stats)
    print(f"[eval] RMSE results: {rmse_results['rmse_per_channel']}")

    dataset_name = ClimatesetDataHP.__name__
    output_var_names = train_run.train_config.train_data_config.output_vars

    for var_idx, var_name in enumerate(output_var_names):
        rmse_value = rmse_results['rmse_per_channel'][var_idx].item()
        print(f"[eval]   RMSE {var_name}: {rmse_value:.6f}")
        insert_checkpoint_sample_metric(
            deser_model.model_id,
            epoch * len(train_ds),
            f"rmse_{var_name}",
            dataset_name,
            [],
            rmse_value,
            [],
        )

    overall_rmse = rmse_results['overall_rmse'].item()
    print(f"[eval]   Overall RMSE: {overall_rmse:.6f}")
    insert_checkpoint_sample_metric(
        deser_model.model_id,
        epoch * len(train_ds),
        "rmse_overall",
        dataset_name,
        [],
        overall_rmse,
        [],
    )

    print("[eval] Exporting metrics to staging...")
    export_all(train_run)
    print(f"[eval] Evaluation complete for variant {variant_idx}, epoch {epoch}!")
    return overall_rmse


def load_create_config(module_file_path):
    module_name = Path(module_file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    return config_file.create_config