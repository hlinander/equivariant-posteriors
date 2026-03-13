"""
Test that a real train_run.json (with legacy short name callables like "LayerNorm")
can be used to instantiate a model via _deserialize_dataclass.
"""
import json
import torch
from pathlib import Path

from lib.serialization import _deserialize_dataclass, _get_config_class_registry
import lib.data_factory as data_factory
import lib.model_factory as model_factory

FIXTURE = Path(__file__).parent / "fixtures" / "train_run_pear.json"


def test_instantiate_model_from_train_run_json():
    """Deserialize model config from a saved train_run.json and instantiate the model."""
    saved = json.loads(FIXTURE.read_text())
    train_config_data = saved["__data__"]["train_config"]["__data__"]
    registry = _get_config_class_registry()

    model_config = _deserialize_dataclass(
        train_config_data["model_config"], registry
    )
    data_config = _deserialize_dataclass(
        train_config_data["train_data_config"], registry
    )

    # Verify norm_layer was resolved to actual class
    assert callable(model_config.norm_layer), (
        f"norm_layer should be callable, got {type(model_config.norm_layer)}: {model_config.norm_layer}"
    )
    assert model_config.norm_layer is torch.nn.LayerNorm

    # Instantiate the model
    data_spec = (
        data_factory.get_factory()
        .get_class(data_config)
        .data_spec(data_config)
    )
    model = model_factory.get_factory().create(model_config, data_spec)
    assert isinstance(model, torch.nn.Module)
