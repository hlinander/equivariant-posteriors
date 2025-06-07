#!/usr/bin/env python
"""
Utilities for testing experiment configurations.
"""
import torch
import traceback
from typing import Optional

from lib.train_dataclasses import TrainRun
import lib.data_factory as data_factory
import lib.model_factory as model_factory


def create_minimal_dataloader(data_config, batch_size: int = 2):
    """Create a minimal dataloader for testing."""
    dataset = data_factory.get_factory().create(data_config)
    # Use a very small batch size for testing
    test_batch_size = min(batch_size, 2)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,  # No multiprocessing for tests
    )
    return dataloader


def validate_single_batch_training(train_run: TrainRun, device: Optional[torch.device] = None) -> bool:
    """
    Test that a TrainRun can train for exactly one batch.
    This is a reusable function that can be used by any experiment.
    
    Args:
        train_run: The TrainRun configuration to test
        device: PyTorch device to use (defaults to CPU)
    
    Returns:
        bool: True if training succeeded, False otherwise
    """
    if device is None:
        device = torch.device("cpu")
    
    try:
        # Create dataset to get data spec
        train_ds = data_factory.get_factory().create(
            train_run.train_config.train_data_config
        )
        data_spec = train_ds.__class__.data_spec(
            train_run.train_config.train_data_config
        )

        # Create model
        model = model_factory.get_factory().create(
            train_run.train_config.model_config, data_spec
        )
        model = model.to(device)

        # Apply post-creation hooks if they exist
        if train_run.train_config.post_model_create_hook:
            try:
                model = train_run.train_config.post_model_create_hook(model, train_run)
            except Exception as e:
                print(f"Warning: post_model_create_hook failed: {e}")

        # Apply pre-training hooks if they exist
        if train_run.train_config.model_pre_train_hook:
            try:
                model = train_run.train_config.model_pre_train_hook(model, train_run)
            except Exception as e:
                print(f"Warning: model_pre_train_hook failed: {e}")

        # Create optimizer
        optimizer = train_run.train_config.optimizer.optimizer(
            model.parameters(), **train_run.train_config.optimizer.kwargs
        )

        # Create dataloader
        dataloader = create_minimal_dataloader(
            train_run.train_config.train_data_config, train_run.train_config.batch_size
        )

        # Get first batch
        batch = next(iter(dataloader))
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

        # Forward pass
        model.train()
        output = model(batch)

        # Compute loss
        loss = train_run.train_config.loss(output, batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping if specified
        if train_run.train_config.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_run.train_config.gradient_clipping
            )

        # Optimizer step
        optimizer.step()

        print(f"Successfully trained one batch. Loss: {loss.item():.4f}")
        return True

    except Exception as e:
        print(f"Single batch training failed: {e}")
        traceback.print_exc()
        return False