#!/usr/bin/env python
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
    add_artifact,
    # has_artifact,
    # add_parameter,
    connect_psql,
    # add_metric_epoch_values,
    insert_checkpoint_sample_metric,
    # get_parameter,
    insert_param,
    get_checkpoints,
)

from lib.distributed_trainer import distributed_train

# from experiments.weather.data import DataHP
from experiments.weather.data import DataHPConfig, Climatology, DataHP
from experiments.weather.metrics import (
    anomaly_correlation_coefficient_hp,
    rmse_hp,
    MeteorologicalData,
)


if __name__ == "__main__":
    device_id = ddp_setup()

    config_file = importlib.import_module(sys.argv[1])
    ensemble_config = create_ensemble_config(config_file.create_config, 1)
    train_run = ensemble_config.members[0]
    result_path = prepare_results(
        f"{train_run.serialize_human()["run_id"]}",
        ensemble_config,
    )

    def save_and_register(name, array):
        path = result_path / f"{name}.npy"

        np.save(
            path,
            array.detach().cpu().float().numpy(),
        )
        add_artifact(train_run, name, path)

    ds_rmse = DataHP(train_run.train_config.train_data_config.validation())
    dl_rmse = torch.utils.data.DataLoader(
        ds_rmse,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    ds_acc = Climatology(train_run.train_config.train_data_config.validation())
    dl_acc = torch.utils.data.DataLoader(
        ds_acc,
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
        eval_report_version = f"eval_log.epoch.{epoch:03d}_v2"
        # if get_parameter(train_run, eval_report_version) is not None:
        # continue
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

        model = deser_model.model
        model.eval()
        print("[eval] rmse")
        rmse_res = rmse_hp(model, dl_rmse, device_id)

        with connect_psql() as conn:
            for var_idx, var_data in enumerate(rmse_res.mean_surface):
                insert_checkpoint_sample_metric(
                    deser_model.model_id,
                )
                # add_metric_epoch_values(
                #     conn,
                #     deser_config.train_run,
                #     f"rmse_surface_{era5_meta.surface.names[var_idx]}",
                #     var_data.item(),
                # )

        print("[eval] acc")
        acc = anomaly_correlation_coefficient_hp(model, dl_acc, device_id)
        save_and_register(f"{epoch:03d}_rmse_surface.npy", rmse_res.surface)
        save_and_register(f"{epoch:03d}_rmse_upper.npy", rmse_res.upper)
        save_and_register(f"{epoch:03d}_acc_surface.npy", acc.acc_unnorm_surface)
        save_and_register(f"{epoch:03d}_acc_upper.npy", acc.acc_unnorm_upper)

        with connect_psql() as conn:
            for var_idx, var_data in enumerate(acc.acc_surface):
                add_metric_epoch_values(
                    conn,
                    deser_config.train_run,
                    f"acc_surface_{era5_meta.surface.names[var_idx]}",
                    var_data.item(),
                )
            train_run_serialized = train_run.serialize_human()
            train_id = train_run_serialized["train_id"]
            ensemble_id = train_run_serialized["ensemble_id"]
            insert_param(conn, train_id, ensemble_id, eval_report_version, "done")
    finally:
        lock.release()
