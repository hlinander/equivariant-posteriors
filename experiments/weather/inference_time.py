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
    insert_artifact,
    insert_model_with_model_id,
    insert_checkpoint_sample_metric,
    ensure_duck,
    attach_pg,
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
    anomaly_correlation_coefficient_dh,
    anomaly_correlation_coefficient_dh_on_dh,
    rmse_hp,
    rmse_dh,
    rmse_dh_on_dh,
    MeteorologicalData,
)


if __name__ == "__main__":
    device_id = ddp_setup()

    # config_file = importlib.import_module(sys.argv[1])
    module_name = Path(sys.argv[1]).stem
    spec = importlib.util.spec_from_file_location(module_name, sys.argv[1])
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    train_run = config_file.create_config(0, 10)

    ds_train = DataHP(train_run.train_config.train_data_config)
    dl_rmse = torch.utils.data.DataLoader(
        ds_train,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    epoch = int(sys.argv[2])

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

    batch = next(iter(dl_rmse))

    surface_shape = batch["input_surface"].shape
    upper_shape = batch["input_upper"].shape

    # data_surface = torch.rand(surface_shape, device=device_id, dtype=torch.float32)
    # data_upper = torch.rand(upper_shape, device=device_id, dtype=torch.float32)
    # model = torch.jit.trace(
    #     model, dict(input_surface=data_surface, input_upper=data_upper), strict=False
    # )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(10):
        data_surface = torch.rand(surface_shape, device=device_id, dtype=torch.float32)
        data_upper = torch.rand(upper_shape, device=device_id, dtype=torch.float32)
        start.record()
        model(dict(input_surface=data_surface, input_upper=data_upper))
        end.record()
        torch.cuda.synchronize()
        # print(start.elapsed_time(end))

    # with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
    # model(dict(input_surface=data_surface, input_upper=data_upper))
    # print(prof)
    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    # print(prof)
    # exit(0)
    times = []
    for _ in range(100):
        data_surface = torch.rand(surface_shape, device=device_id, dtype=torch.float32)
        data_upper = torch.rand(upper_shape, device=device_id, dtype=torch.float32)
        start.record()
        model(dict(input_surface=data_surface, input_upper=data_upper))
        end.record()
        torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        times.append(start.elapsed_time(end))

    times = np.array(times)
    print(times.mean(), times.std())
