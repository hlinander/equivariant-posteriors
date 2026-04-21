#!/usr/bin/env python
"""Measure inference time and parameter count for weather models.

Usage:
    python experiments/weather/inference_time.py <config.py> <epoch>

Example:
    python experiments/weather/inference_time.py experiments/weather/persisted_configs/pear.py 200
    python experiments/weather/inference_time.py experiments/weather/persisted_configs/graphcast_physicsnemo.py 500
"""
import sys
import importlib
import torch
import numpy as np
from pathlib import Path

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.serialization import deserialize_model, DeserializeConfig
from lib.model_factory import get_factory as get_model_factory
from lib.data_factory import get_factory as get_dataset_factory
from experiments.weather.data import DataHP


def load_create_config(config_path):
    module_name = Path(config_path).stem
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    return config_file.create_config


if __name__ == "__main__":
    device_id = ddp_setup()

    config_path = sys.argv[1]
    create_config = load_create_config(config_path)
    train_run = create_config(0)

    # Parameter count
    data_spec = (
        get_dataset_factory()
        .get_class(train_run.train_config.train_data_config)
        .data_spec(train_run.train_config.train_data_config)
    )
    model_fresh = get_model_factory().create(train_run.train_config.model_config, data_spec)
    n_params = sum(p.numel() for p in model_fresh.parameters())
    print(f"Model: {train_run.train_config.model_config.__class__.__name__}")
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    del model_fresh

    # Load checkpoint for inference timing
    epoch = int(sys.argv[2])
    deser_config = DeserializeConfig(
        train_run=create_ensemble_config(
            lambda eid: create_config(eid, epoch),
            1,
        ).members[0],
        device_id=device_id,
    )
    deser_model = deserialize_model(deser_config)
    if deser_model is None:
        print("Can't deserialize checkpoint, skipping timing")
        sys.exit(0)

    model = deser_model.model
    model.eval()

    # Get input shapes from dataset
    ds = DataHP(train_run.train_config.train_data_config)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    batch = next(iter(dl))
    surface_shape = batch["input_surface"].shape
    upper_shape = batch["input_upper"].shape

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(10):
        data_surface = torch.rand(surface_shape, device=device_id, dtype=torch.float32)
        data_upper = torch.rand(upper_shape, device=device_id, dtype=torch.float32)
        start.record()
        model(dict(input_surface=data_surface, input_upper=data_upper))
        end.record()
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(100):
        data_surface = torch.rand(surface_shape, device=device_id, dtype=torch.float32)
        data_upper = torch.rand(upper_shape, device=device_id, dtype=torch.float32)
        start.record()
        model(dict(input_surface=data_surface, input_upper=data_upper))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)
    print(f"Inference time: {times.mean():.2f} ± {times.std():.2f} ms")
