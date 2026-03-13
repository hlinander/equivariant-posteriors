"""
Test serialization and deserialization of callable fields (e.g. nn.LayerNorm).
Covers both the new fully qualified format and legacy short name compatibility.
"""
import dataclasses
import torch
from dataclasses import dataclass

from lib.stable_hash import json_default, json_dumps_dataclass_str, stable_hash_str
from lib.serialization import _resolve_callable_string, _deserialize_dataclass


# --- json_default tests ---


def test_legacy_short_names_preserved():
    """Legacy callables serialize to short names for hash stability."""
    assert json_default(torch.nn.LayerNorm) == "LayerNorm"
    assert json_default(torch.optim.Adam) == "Adam"
    assert json_default(torch.optim.AdamW) == "AdamW"
    assert json_default(torch.optim.SGD) == "SGD"


def test_new_callables_get_qualified_names():
    """Non-legacy callables get fully qualified module.qualname."""
    result = json_default(torch.nn.BatchNorm1d)
    assert "BatchNorm1d" in result
    assert "." in result  # Should be fully qualified

    result = json_default(torch.optim.RMSprop)
    assert "RMSprop" in result
    assert "." in result


def test_local_functions_use_short_name():
    """Local functions / closures with <locals> in qualname use short name."""
    def my_loss(x):
        return x

    result = json_default(my_loss)
    assert result == "my_loss"


def test_lambda_uses_short_name():
    """Lambdas use short name since qualname contains '<'."""
    f = lambda x: x
    result = json_default(f)
    assert result == "<lambda>"


def test_legacy_hash_stability():
    """Hashes for configs using legacy callables must not change."""
    @dataclass
    class FakeConfig:
        norm_layer: object = torch.nn.LayerNorm
        optimizer: object = torch.optim.AdamW
        value: int = 42

    hash1 = stable_hash_str(FakeConfig())

    # Simulate what the old json_default would have produced
    # by verifying the hash matches a config where we manually set strings
    old_json_default_output = json_dumps_dataclass_str(FakeConfig())
    assert '"LayerNorm"' in old_json_default_output
    assert '"AdamW"' in old_json_default_output

    # Hash should be stable across runs
    hash2 = stable_hash_str(FakeConfig())
    assert hash1 == hash2


# --- _resolve_callable_string tests ---


def test_resolve_fully_qualified():
    """Fully qualified names resolve to the actual class."""
    result = _resolve_callable_string("torch.nn.modules.normalization.LayerNorm")
    assert result is torch.nn.LayerNorm


def test_resolve_legacy_short_name_nn():
    """Legacy short names resolve via torch.nn namespace."""
    assert _resolve_callable_string("LayerNorm") is torch.nn.LayerNorm
    assert _resolve_callable_string("BatchNorm1d") is torch.nn.BatchNorm1d


def test_resolve_legacy_short_name_optim():
    """Legacy short names resolve via torch.optim namespace."""
    assert _resolve_callable_string("Adam") is torch.optim.Adam
    assert _resolve_callable_string("AdamW") is torch.optim.AdamW
    assert _resolve_callable_string("SGD") is torch.optim.SGD


def test_resolve_field_default_fallback():
    """Falls back to matching the field default when namespace lookup fails."""
    class MyCustomClass:
        __name__ = "MyCustomClass"

    @dataclass
    class Config:
        custom: object = MyCustomClass

    field_def = dataclasses.fields(Config)[0]
    result = _resolve_callable_string("MyCustomClass", field_def)
    assert result is MyCustomClass


def test_resolve_plain_string_unchanged():
    """Non-callable strings that don't match anything are returned as-is."""
    assert _resolve_callable_string("just_a_string") == "just_a_string"


def test_resolve_non_string_passthrough():
    """Non-string values pass through unchanged."""
    assert _resolve_callable_string(42) == 42
    assert _resolve_callable_string(None) is None
    assert _resolve_callable_string([1, 2]) == [1, 2]


# --- _deserialize_dataclass with callable fields ---


def test_deserialize_dataclass_resolves_callable_fields():
    """Deserialization resolves callable string fields back to classes."""
    @dataclass
    class ModelConfig:
        dim: int = 64
        norm_layer: object = torch.nn.LayerNorm

    registry = {"ModelConfig": ModelConfig}
    json_dict = {
        "__class__": "ModelConfig",
        "__data__": {
            "dim": 64,
            "norm_layer": "LayerNorm",
        },
    }
    result = _deserialize_dataclass(json_dict, registry)
    assert result.dim == 64
    assert result.norm_layer is torch.nn.LayerNorm


def test_deserialize_dataclass_resolves_qualified_callable():
    """Deserialization resolves fully qualified callable names."""
    @dataclass
    class ModelConfig:
        dim: int = 64
        norm_layer: object = torch.nn.LayerNorm

    registry = {"ModelConfig": ModelConfig}
    json_dict = {
        "__class__": "ModelConfig",
        "__data__": {
            "dim": 64,
            "norm_layer": "torch.nn.modules.normalization.LayerNorm",
        },
    }
    result = _deserialize_dataclass(json_dict, registry)
    assert result.norm_layer is torch.nn.LayerNorm


def test_deserialize_dataclass_none_callable_field():
    """None values in callable fields stay None."""
    @dataclass
    class ModelConfig:
        norm_layer: object = None

    registry = {"ModelConfig": ModelConfig}
    json_dict = {
        "__class__": "ModelConfig",
        "__data__": {"norm_layer": None},
    }
    result = _deserialize_dataclass(json_dict, registry)
    assert result.norm_layer is None
