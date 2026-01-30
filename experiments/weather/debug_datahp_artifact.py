#!/usr/bin/env python
"""
Debug script to aggregate DataHP samples and upload as artifacts.

Iterates through a range of DataHP indices and creates:
  - Upper artifact: [time, level, variable, healpix]
  - Surface artifact: [time, variable, healpix]

Usage:
    uv run python experiments/weather/debug_datahp_artifact.py

Then run ingestion:
    uv run python ingestion/ingest.py
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import lib.render_duck as duck
from lib.export import export_all
from lib.train_dataclasses import TrainConfig, TrainRun, OptimizerConfig, ComputeConfig
from lib.models.dense import DenseConfig
from lib.data_registry import DataSineConfig
from lib.regression_metrics import create_regression_metrics
from experiments.weather.data import DataHP, DataHPConfig


def create_debug_train_run():
    """Create a minimal TrainRun for debug purposes"""
    loss = torch.nn.functional.mse_loss

    def mse_loss(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=DenseConfig(d_hidden=10),
        train_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        val_data_config=None,
        loss=mse_loss,
        optimizer=OptimizerConfig(optimizer=torch.optim.Adam, kwargs=dict()),
        batch_size=1,
    )
    train_eval = create_regression_metrics(loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=1,
        save_nth_epoch=1,
        validate_nth_epoch=1,
        project="debug_datahp",
    )
    return train_run


def main():
    # Configuration
    nside = 64
    start_idx = 0
    end_idx = 10  # Aggregate 10 time steps

    n_pixels = 12 * nside**2
    n_time = end_idx - start_idx

    print(f"Aggregating DataHP indices {start_idx} to {end_idx - 1}")
    print(f"nside={nside}, n_pixels={n_pixels}")

    # Load dataset
    config = DataHPConfig(nside=nside, normalized=True)
    dataset = DataHP(config)
    print(f"Dataset length: {len(dataset)}")

    # Pre-allocate arrays
    # Upper: [time, level, variable, healpix]
    # Original input_upper shape: [n_upper=5, n_levels=13, n_pixels]
    upper_data = np.zeros((n_time, 13, 5, n_pixels), dtype=np.float32)

    # Surface: [time, variable, healpix]
    # Original input_surface shape: [n_surface=4, n_pixels]
    surface_data = np.zeros((n_time, 4, n_pixels), dtype=np.float32)

    # Collect data
    for i, idx in enumerate(tqdm(range(start_idx, end_idx), desc="Loading samples")):
        sample = dataset[idx]

        # input_upper: [5, 13, n_pixels] -> transpose to [13, 5, n_pixels]
        upper_data[i] = sample["input_upper"].transpose(1, 0, 2)

        # input_surface: [4, n_pixels] -> keep as is
        surface_data[i] = sample["input_surface"]

    print(f"Upper tensor shape: {upper_data.shape}")  # [time, level, variable, healpix]
    print(f"Surface tensor shape: {surface_data.shape}")  # [time, variable, healpix]

    # Create train run and initialize duck
    train_run = create_debug_train_run()
    duck.ensure_duck(train_run)

    # Insert model
    model_id = duck.insert_model(train_run)
    duck.insert_run(train_run.run_id, model_id)
    print(f"Created model_id: {model_id}, run_id: {train_run.run_id}")

    # Save tensors to files
    output_dir = Path(".local/debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    upper_path = output_dir / f"datahp_upper_nside{nside}.npy"
    surface_path = output_dir / f"datahp_surface_nside{nside}.npy"

    np.save(upper_path, upper_data)
    np.save(surface_path, surface_data)
    print(f"Saved upper tensor to: {upper_path}")
    print(f"Saved surface tensor to: {surface_path}")

    # Insert as artifacts
    upper_artifact_id = duck.insert_artifact(
        model_id, f"datahp_upper_nside{nside}", upper_path, "npy"
    )
    surface_artifact_id = duck.insert_artifact(
        model_id, f"datahp_surface_nside{nside}", surface_path, "npy"
    )
    print(f"Inserted upper artifact_id: {upper_artifact_id}")
    print(f"Inserted surface artifact_id: {surface_artifact_id}")

    # Export to staging
    print("\nExporting to staging...")
    paths = export_all(train_run)
    print(f"Exported {len(paths)} files:")
    for p in paths:
        print(f"  {p}")

    print("\nDone! Run ingestion with: uv run python ingestion/ingest.py")


if __name__ == "__main__":
    main()
