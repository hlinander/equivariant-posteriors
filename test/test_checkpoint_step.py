"""Test that resolve_step_for_epoch correctly recovers the training step from analytics."""
import os
import torch
import pytest
from pathlib import Path

from lib.train_dataclasses import TrainConfig, TrainEval, TrainRun, OptimizerConfig, ComputeConfig
from lib.models.dense import DenseConfig
from lib.data_registry import DataSineConfig
from lib.regression_metrics import create_regression_metrics
from lib.serialization import serialize, SerializeConfig
from lib.train import create_initial_state
from lib.stable_hash import stable_hash_str
from lib.paths import get_checkpoint_path
from lib.checkpoint_step import resolve_step_for_epoch
import lib.compute_env as compute_env
from lib.compute_env_config import ComputeEnvironment, Paths
import lib.render_duck as duck


def _make_train_run(keep_nth=2):
    loss = torch.nn.functional.mse_loss

    def mse_loss(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=DenseConfig(d_hidden=20),
        train_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        val_data_config=None,
        loss=mse_loss,
        optimizer=OptimizerConfig(optimizer=torch.optim.Adam, kwargs=dict()),
        batch_size=2,
    )
    train_eval = create_regression_metrics(loss, None)
    return TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=6,
        save_nth_epoch=1,
        validate_nth_epoch=1,
        keep_nth_epoch_checkpoints=keep_nth,
        keep_epoch_checkpoints=True,
        project="test",
    )


@pytest.fixture
def env_setup(tmp_path):
    """Redirect checkpoints to tmp, set up duck in-memory, skip code copy."""
    original_env = compute_env._current_env
    compute_env._current_env = ComputeEnvironment(
        paths=Paths(checkpoints=tmp_path / "checkpoints")
    )
    os.environ["NOCOPY"] = "1"

    # Reset duck for a fresh in-memory db
    if duck.CONN is not None:
        duck.CONN.close()
    duck.CONN = None
    duck.SCHEMA_ENSURED = False
    duck.ensure_duck(None, True)

    yield tmp_path

    os.environ.pop("NOCOPY", None)
    compute_env._current_env = original_env
    if duck.CONN is not None:
        duck.CONN.close()
    duck.CONN = None
    duck.SCHEMA_ENSURED = False


def _train_epochs(train_run, device="cpu"):
    """Simulate training for multiple epochs with serialization."""
    state = create_initial_state(train_run, None, device)

    for epoch in range(1, train_run.epochs + 1):
        # Simulate a training epoch by iterating the dataloader
        for batch in state.train_dataloader:
            outputs = state.model(batch)
            loss = train_run.train_config.loss(outputs, batch)
            loss.backward()
            state.optimizer.step()
            state.optimizer.zero_grad()
            state.batch += 1

        state.epoch = epoch
        serialize(SerializeConfig(train_run=train_run, train_epoch_state=state))

    return state


def _export_analytics(train_run):
    """Export duck tables to parquets in the checkpoint analytics dir."""
    from lib.staging_filesystem import flush_all_to_checkpoint
    checkpoint_path = get_checkpoint_path(train_run.train_config)
    flush_all_to_checkpoint(train_run, checkpoint_path, duck.CONN)


def test_resolve_step_for_epoch(env_setup):
    """Train a model, export analytics, and verify step resolution."""
    train_run = _make_train_run(keep_nth=2)
    state = _train_epochs(train_run)
    _export_analytics(train_run)

    checkpoint_path = get_checkpoint_path(train_run.train_config)

    # Epoch checkpoints should exist for epochs 2, 4, 6 (keep_nth=2)
    for epoch in [2, 4, 6]:
        step = resolve_step_for_epoch(checkpoint_path, epoch)
        assert step is not None, f"Should resolve step for epoch {epoch}"
        # Step should be epoch * batches_per_epoch
        # DataSineConfig has 100 samples, batch_size=2, drop_last=False -> 50 batches/epoch
        expected = epoch * 50
        assert step == expected, f"Epoch {epoch}: expected step {expected}, got {step}"

    # Epochs without checkpoints should return None
    for epoch in [1, 3, 5]:
        step = resolve_step_for_epoch(checkpoint_path, epoch)
        assert step is None, f"Epoch {epoch} should not have a checkpoint"


def test_fallback_matches_actual_step(env_setup):
    """Verify the fallback computation matches the actual training step."""
    import math
    train_run = _make_train_run(keep_nth=2)
    state = _train_epochs(train_run)
    _export_analytics(train_run)

    checkpoint_path = get_checkpoint_path(train_run.train_config)
    ds_len = len(state.train_dataloader.dataset)
    batch_size = train_run.train_config.batch_size
    batches_per_epoch = math.ceil(ds_len / batch_size)

    for epoch in [2, 4, 6]:
        actual_step = resolve_step_for_epoch(checkpoint_path, epoch)
        fallback_step = epoch * batches_per_epoch
        assert actual_step == fallback_step, (
            f"Epoch {epoch}: actual step {actual_step} != fallback {fallback_step} "
            f"(ds_len={ds_len}, batch_size={batch_size})"
        )


def test_resolve_step_no_analytics(env_setup):
    """Returns None when no analytics parquets exist."""
    train_run = _make_train_run()
    # Don't train or export, just get the path
    checkpoint_path = get_checkpoint_path(train_run.train_config)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    step = resolve_step_for_epoch(checkpoint_path, 2)
    assert step is None
