"""
Test roundtrip serialization and loading of checkpoints,
including loading from checkpoints where the config has changed.
"""
import os
import torch
import pytest
from pathlib import Path

from lib.train_dataclasses import TrainConfig, TrainEval, TrainRun, OptimizerConfig, ComputeConfig
from lib.models.dense import DenseConfig
from lib.data_registry import DataSineConfig
from lib.regression_metrics import create_regression_metrics
from lib.serialization import (
    serialize,
    SerializeConfig,
    deserialize_model,
    DeserializeConfig,
    load_model_from_checkpoint,
    load_checkpoint_data_config,
    load_checkpoint_train_run_json,
    list_checkpoint_epochs,
)
from lib.train import create_initial_state
from lib.stable_hash import stable_hash_str
import lib.compute_env as compute_env
from lib.compute_env_config import ComputeEnvironment, Paths
import lib.render_duck as duck


def make_train_run(d_hidden=100):
    loss = torch.nn.functional.mse_loss

    def mse_loss(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=DenseConfig(d_hidden=d_hidden),
        train_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        val_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        loss=mse_loss,
        optimizer=OptimizerConfig(optimizer=torch.optim.Adam, kwargs=dict()),
        batch_size=2,
    )
    train_eval = create_regression_metrics(loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=1,
        save_nth_epoch=1,
        validate_nth_epoch=1,
        project="test",
    )
    return train_run


@pytest.fixture
def checkpoint_env(tmp_path):
    """Redirect checkpoint storage to a temp directory, skip code copy."""
    original_env = compute_env._current_env
    compute_env._current_env = ComputeEnvironment(
        paths=Paths(checkpoints=tmp_path / "checkpoints")
    )
    os.environ["NOCOPY"] = "1"
    yield tmp_path
    os.environ.pop("NOCOPY", None)
    compute_env._current_env = original_env


@pytest.fixture
def checkpoint_env_with_code(tmp_path):
    """Redirect checkpoint storage to a temp directory, include code copy."""
    original_env = compute_env._current_env
    compute_env._current_env = ComputeEnvironment(
        paths=Paths(checkpoints=tmp_path / "checkpoints")
    )
    os.environ.pop("NOCOPY", None)
    yield tmp_path
    compute_env._current_env = original_env


def _train_and_serialize(train_run, device="cpu"):
    """Create initial state, do one forward/backward pass, and serialize."""
    state = create_initial_state(train_run, None, device)

    # Do a training step so the weights differ from init
    batch = next(iter(state.train_dataloader))
    outputs = state.model(batch)
    loss = train_run.train_config.loss(outputs, batch)
    loss.backward()
    state.optimizer.step()
    state.epoch = 1

    serialize(SerializeConfig(train_run=train_run, train_epoch_state=state))
    return state


def test_roundtrip_same_config(checkpoint_env):
    """Serialize a checkpoint and load it back with the same config."""
    train_run = make_train_run(d_hidden=50)
    state = _train_and_serialize(train_run)

    original_params = {
        k: v.clone() for k, v in state.model.state_dict().items()
    }

    deser = deserialize_model(DeserializeConfig(train_run, "cpu"), latest_ok=True)
    assert deser is not None
    for k, v in deser.model.state_dict().items():
        assert torch.equal(v, original_params[k]), f"Mismatch on {k}"


def test_load_from_checkpoint_same_config(checkpoint_env):
    """load_model_from_checkpoint works when config hasn't changed."""
    train_run = make_train_run(d_hidden=50)
    state = _train_and_serialize(train_run)

    original_params = {
        k: v.clone() for k, v in state.model.state_dict().items()
    }

    checkpoint_hash = stable_hash_str(train_run.train_config)
    deser = load_model_from_checkpoint(checkpoint_hash, "cpu")
    assert deser is not None
    for k, v in deser.model.state_dict().items():
        assert torch.equal(v, original_params[k]), f"Mismatch on {k}"


def test_load_from_checkpoint_changed_config(checkpoint_env):
    """Config changed (different d_hidden), but we can still load the old
    checkpoint by hash because load_model_from_checkpoint reconstructs the
    model config from the saved train_run.json."""
    # Train and serialize with d_hidden=50
    train_run_v1 = make_train_run(d_hidden=50)
    state = _train_and_serialize(train_run_v1)
    checkpoint_hash = stable_hash_str(train_run_v1.train_config)

    original_params = {
        k: v.clone() for k, v in state.model.state_dict().items()
    }

    # "Change the config" — now we'd use d_hidden=80.
    # The normal deserialize_model path would fail because the hash differs
    # and the architecture wouldn't match.
    train_run_v2 = make_train_run(d_hidden=80)
    new_hash = stable_hash_str(train_run_v2.train_config)
    assert new_hash != checkpoint_hash, "Hashes should differ"

    # Normal path: can't find checkpoint for the new config
    deser_normal = deserialize_model(
        DeserializeConfig(train_run_v2, "cpu"), latest_ok=True
    )
    assert deser_normal is None, "Should not find checkpoint for changed config"

    # New path: load by old hash, model is reconstructed from saved JSON
    deser = load_model_from_checkpoint(checkpoint_hash, "cpu")
    assert deser is not None

    # Verify it loaded the d_hidden=50 model with correct weights
    for k, v in deser.model.state_dict().items():
        assert torch.equal(v, original_params[k]), f"Mismatch on {k}"

    # Verify the loaded model actually has the old architecture (d_hidden=50)
    # Dense model has l1 (input -> d_hidden) and l2 (d_hidden -> output)
    l1_weight = deser.model.state_dict()["l1.weight"]
    assert l1_weight.shape[0] == 50


def test_load_checkpoint_data_config(checkpoint_env):
    """load_checkpoint_data_config reconstructs the data config from JSON."""
    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    data_config = load_checkpoint_data_config(checkpoint_hash)
    assert type(data_config).__name__ == "DataSineConfig"
    # torch.Size is serialized as a list in JSON
    assert list(data_config.input_shape) == [1]
    assert list(data_config.output_shape) == [1]


def test_load_checkpoint_train_run_json(checkpoint_env):
    """load_checkpoint_train_run_json returns the saved JSON with expected structure."""
    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    saved = load_checkpoint_train_run_json(checkpoint_hash)
    assert "__class__" in saved
    assert saved["__class__"] == "TrainRun"
    assert "__data__" in saved
    assert "train_config" in saved["__data__"]
    train_config = saved["__data__"]["train_config"]
    assert train_config["__data__"]["model_config"]["__class__"] == "DenseConfig"


def test_list_checkpoint_epochs(checkpoint_env):
    """list_checkpoint_epochs finds epoch-specific checkpoint files."""
    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    # No epoch checkpoints saved (keep_epoch_checkpoints=False by default)
    assert list_checkpoint_epochs(checkpoint_hash) == []

    # Create fake epoch checkpoint files
    from lib.compute_env import env
    checkpoint_path = env().paths.checkpoints / f"checkpoint_{checkpoint_hash}"
    for epoch in [0, 20, 40]:
        (checkpoint_path / f"model_epoch_{epoch:04d}").touch()

    epochs = list_checkpoint_epochs(checkpoint_hash)
    assert epochs == [0, 20, 40]


def test_load_epoch_checkpoint(checkpoint_env):
    """load_model_from_checkpoint with epoch parameter loads the right file."""
    train_run = make_train_run(d_hidden=50)
    state = _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    # Save an epoch-specific checkpoint
    from lib.compute_env import env
    checkpoint_path = env().paths.checkpoints / f"checkpoint_{checkpoint_hash}"
    torch.save(state.model.state_dict(), checkpoint_path / "model_epoch_0001")

    original_params = {
        k: v.clone() for k, v in state.model.state_dict().items()
    }

    deser = load_model_from_checkpoint(checkpoint_hash, "cpu", epoch=1)
    assert deser is not None
    assert deser.epoch == 1
    for k, v in deser.model.state_dict().items():
        assert torch.equal(v, original_params[k]), f"Mismatch on {k}"


def test_load_missing_epoch_returns_none(checkpoint_env):
    """load_model_from_checkpoint returns None for a non-existent epoch."""
    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    deser = load_model_from_checkpoint(checkpoint_hash, "cpu", epoch=999)
    assert deser is None


def test_setup_duck_from_checkpoint(checkpoint_env):
    """setup_duck_from_checkpoint records model parameters from saved JSON."""
    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    # Clean duck state
    if duck.CONN is not None:
        duck.CONN.close()
        duck.CONN = None
        duck.SCHEMA_ENSURED = False

    saved_json = load_checkpoint_train_run_json(checkpoint_hash)
    run_id = duck.setup_duck_from_checkpoint(checkpoint_hash, 42, saved_json)

    # Verify model was inserted
    result = duck.CONN.execute(
        "SELECT train_id FROM models WHERE id = 42"
    ).fetchone()
    assert result is not None
    assert result[0] == checkpoint_hash

    # Verify parameters were recorded
    params = duck.CONN.execute(
        "SELECT name FROM model_parameter WHERE model_id = 42 AND name = 'train_config_hash'"
    ).fetchone()
    assert params is not None

    # Cleanup
    duck.CONN.close()
    duck.CONN = None
    duck.SCHEMA_ENSURED = False


def test_code_saved_in_checkpoint(checkpoint_env_with_code):
    """Serialization saves a code snapshot into the checkpoint directory."""
    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    from lib.compute_env import env
    checkpoint_path = env().paths.checkpoints / f"checkpoint_{checkpoint_hash}"
    code_path = checkpoint_path / "code"

    assert code_path.is_dir(), "code/ directory should exist in checkpoint"
    python_files = list(code_path.rglob("*.py"))
    assert len(python_files) > 0, "code/ should contain Python files"
    # Verify key files are present
    assert (code_path / "lib" / "serialization.py").is_file()
    assert (code_path / "lib" / "train_dataclasses.py").is_file()


def test_code_saved_only_once(checkpoint_env_with_code):
    """Code snapshot is only saved on first serialization, not overwritten."""
    train_run = make_train_run(d_hidden=50)
    state = _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    from lib.compute_env import env
    checkpoint_path = env().paths.checkpoints / f"checkpoint_{checkpoint_hash}"
    code_path = checkpoint_path / "code"

    # Record modification time of a file in the code snapshot
    marker = code_path / "lib" / "serialization.py"
    mtime_first = marker.stat().st_mtime

    # Serialize again (epoch 2)
    state.epoch = 2
    serialize(SerializeConfig(train_run=train_run, train_epoch_state=state))

    # Code directory should not have been rewritten
    mtime_second = marker.stat().st_mtime
    assert mtime_first == mtime_second, "code/ should not be overwritten on second serialize"
