import sys
import importlib
import torch
import copy
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

from experiments.climate.data.climateset_data_no_hp import ClimatesetConfig, ClimatesetData
from experiments.climate.data.climateset_data_no_hp import load_training_stats_from_config

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
from experiments.climate.evaluation.metrics import rmse_climate_nohp

sys.stdout.flush()


def evaluate_climate(create_config, epoch, variant_idx=0):
    device_id = ddp_setup()

    print("Registering datasets and models...")
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetConfig, ClimatesetData)
    mf = model_factory.get_factory()
    mf.register(ClimaXConfig, ClimaX)
    mf.register(UNetConfig, UNet)
    mf.register(CNNLSTMConfig, CNNLSTM_ClimateBench)
    mf.register(GRUTemporalWrapperConfig, GRUTemporalWrapper)

    print(f"[eval] Evaluating variant {variant_idx}, epoch {epoch}")
    train_run = create_config(ensemble_id=variant_idx)
    train_run.epochs = epoch

    train_ds = ClimatesetData(train_run.train_config.train_data_config)
    test_data_config = copy.deepcopy(train_run.train_config.train_data_config)
    test_data_config.scenarios = ["ssp245"]
    test_data_config.split = "test"
    test_ds = ClimatesetData(test_data_config)

    # Load normalization stats from training and set in test set
    stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    test_ds.set_normalization_stats(**stats)

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=4,
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

    output_stats = stats["output_stats"]

    print("[eval] Computing latitude-weighted RMSE...")
    rmse_results = rmse_climate_nohp(model, test_dl, device_id, output_stats)
    print(f"[eval] RMSE results: {rmse_results['rmse_per_channel']}")

    dataset_name = ClimatesetData.__name__
    output_var_names = train_run.train_config.train_data_config.output_vars

    for var_idx, var_name in enumerate(output_var_names):
        rmse_value = rmse_results['rmse_per_channel'][var_idx].item()
        print(f"[eval]   RMSE {var_name}: {rmse_value:.6f}")
        insert_checkpoint_sample_metric(
            deser_model.model_id,
            epoch * len(train_ds),
            f"rmse_latw_{var_name}",
            dataset_name,
            [],
            rmse_value,
            [],
        )

    overall_rmse = rmse_results['overall_rmse'].item()
    print(f"[eval]   Overall lat-weighted RMSE: {overall_rmse:.6f}")
    insert_checkpoint_sample_metric(
        deser_model.model_id,
        epoch * len(train_ds),
        "rmse_latw_overall",
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
