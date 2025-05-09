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
    # result_path = prepare_results(
    #     f"{train_run.serialize_human()["run_id"]}",
    #     train_run,
    # )

    # def save_and_register(name, array):
    #     path = result_path / f"{name}.npy"

    #     np.save(
    #         path,
    #         array.detach().cpu().float().numpy(),
    #     )
    # add_artifact(train_run, name, path)

    lead_time_days = int(os.environ.get("LEADTIME", "1"))
    print("Lead time {lead_time}d")
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

    # lock = FileLock(
    #     get_lock_path(train_config=train_run.train_config, lock_name=f"eval_{epoch}"),
    #     0.1,
    # )
    # try:
    #     lock.acquire(blocking=False)
    # except Timeout:
    #     print("Already evaluating...")
    #     exit(0)

    # try:

    # eval_report_version = f"eval_log.epoch.{epoch:03d}_v2"
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

    ensure_duck(train_run)
    attach_pg()
    # insert_model_with_model_id(train_run, deser_model.model_id)
    # insert_or_update_train_run(train_run, deser_model.model_id)

    model = deser_model.model
    model.eval()

    print("ACC")
    if ds_rmse_config.driscoll_healy:
        acc_res = rmse_dh(model, dl_rmse, device_id)
    else:
        acc_res = rmse_hp(model, dl_rmse, device_id)
    print("[eval] rmse")
    if ds_rmse_config.driscoll_healy:
        rmse_res = rmse_dh(model, dl_rmse, device_id)
    else:
        rmse_res = rmse_hp(model, dl_rmse, device_id)

    for var_idx, var_data in enumerate(rmse_res.mean_surface):
        insert_checkpoint_sample_metric(
            deser_model.model_id,
            epoch * len(ds_train),
            f"rmse_surface_{era5_meta.surface.names[var_idx]}.{ds_rmse_config.lead_time_days}d",
            ds_rmse_config.short_name(),
            [],
            var_data.item(),
            [],
            db_prefix="pg.",
        )
    for var_idx, var_data in enumerate(rmse_res.mean_upper):
        for level, value in zip(era5_meta.upper.levels, var_data.cpu().numpy()):
            # print(level, value)
            var_name = f"rmse_upper_{era5_meta.upper.names[var_idx]}_{int(level)}.{ds_rmse_config.lead_time_days}d"
            # print(var_name)
            # breakpoint()
            insert_checkpoint_sample_metric(
                deser_model.model_id,
                epoch * len(ds_train),
                var_name,
                ds_rmse_config.short_name(),
                [],
                value.item(),
                [],
                db_prefix="pg.",
            )
