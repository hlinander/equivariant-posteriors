#!/usr/bin/env python
import os
import sys
import importlib
import torch
import numpy as np
from pathlib import Path
from filelock import FileLock, Timeout

# import onnxruntime as ort

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.metric import create_metric
from lib.paths import get_lock_path


# from lib.models.healpix.swin_hp_transformer import SwinHPTransformerConfig
from experiments.weather.models.swin_hp_pangu import SwinHPPanguConfig

# from experiments.weather.models.swin_hp_pangu import SwinHPPangu

# from lib.models.mlp import MLPConfig
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.files import prepare_results
from lib.serialization import deserialize_model, DeserializeConfig


from lib.data_factory import get_factory as get_dataset_factory

from lib.render_duck import (
    # add_artifact,
    # has_artifact,
    # add_parameter,
    # connect_psql,
    # add_metric_epoch_values,
    insert_or_update_train_run,
    insert_model_with_model_id,
    insert_checkpoint_sample_metric,
    ensure_duck,
    attach_pg,
    insert_artifact,
    sync,
)

# get_parameter,
# insert_param,
# get_checkpoints,
# )

from lib.distributed_trainer import distributed_train

# from experiments.weather.data import DataHP
from experiments.weather.data import DataHPConfig, Climatology, DataHP
from experiments.weather.metrics import (
    anomaly_correlation_coefficient_hp,
    rmse_hp,
    rmse_dh,
    MeteorologicalData,
)


if __name__ == "__main__":
    device_id = ddp_setup()

    # config_file = importlib.import_module(sys.argv[1])
    module_name = Path(sys.argv[1]).stem
    spec = importlib.util.spec_from_file_location(module_name, sys.argv[1])
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    # ensemble_config = create_ensemble_config(config_file.create_config, 1)
    train_run = config_file.create_config(0, 10)
    # train_run = ensemble_config.members[0]
    ensure_duck(train_run, in_memory=True)
    attach_pg()
    result_path = prepare_results(
        f"{train_run.serialize_human()["run_id"]}",
        train_run,
    )

    # add_artifact(train_run, name, path)

    ds_train = DataHP(train_run.train_config.train_data_config)
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    ds_val_config = train_run.train_config.train_data_config.validation()
    ds_val = DataHP(ds_val_config)
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    era5_meta = MeteorologicalData()

    epoch = int(sys.argv[2])

    lock = FileLock(
        get_lock_path(train_config=train_run.train_config, lock_name=f"eval_{epoch}"),
        0.1,
    )
    try:
        lock.acquire(blocking=False)
    except Timeout:
        print("Already evaluating...")
        exit(0)

    try:
        print(f"[inspect output] Epoch {epoch}")
        print(f"[inspect output] Results at {result_path}")
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

        def save_and_register(name, array):
            path = result_path / f"{name}.npy"

            np.save(
                path,
                array.detach().cpu().float().numpy(),
            )
            insert_artifact(deser_model.model_id, name, path, ".npy")

        model = deser_model.model
        model.eval()
        print("[eval] rmse")
        for idx, batch in enumerate(dl_val):
            batch = {
                k: v.to(device_id) if hasattr(v, "to") else v for k, v in batch.items()
            }
            with torch.no_grad():
                output = model(batch)

            grid_str = "dh" if ds_val_config.driscoll_healy else "hp"
            save_and_register(
                f"{idx}_of_surface_{grid_str}.npy", output["logits_surface"]
            )
            save_and_register(
                f"{idx}_if_surface_{grid_str}.npy", batch["input_surface"]
            )
            save_and_register(
                f"{idx}_tf_surface_{grid_str}.npy", batch["target_surface"]
            )
            save_and_register(f"{idx}_of_upper_{grid_str}.npy", output["logits_upper"])
            save_and_register(f"{idx}_if_upper_{grid_str}.npy", batch["input_upper"])
            save_and_register(f"{idx}_tf_upper_{grid_str}.npy", batch["target_upper"])
            sync(train_run)
            del output
            break

    finally:
        lock.release()
