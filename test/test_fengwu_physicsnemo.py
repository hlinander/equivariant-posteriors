import torch
import pytest

from experiments.weather.models.fengwu_physicsnemo import (
    FengwuPhysicsNemo,
    FengwuPhysicsNemoConfig,
    _DATA_TO_FENGWU,
    _FENGWU_TO_DATA,
)
from experiments.weather.data import DataSpecHP


@pytest.fixture
def config():
    return FengwuPhysicsNemoConfig(nside=64, embed_dim=192 // 4, patch_size=(4, 4))


@pytest.fixture
def data_spec():
    return DataSpecHP(nside=64, n_surface=4, n_upper=5)


@pytest.fixture
def model(config, data_spec):
    return FengwuPhysicsNemo(config, data_spec)


def make_batch(lat, lon, device="cpu"):
    return dict(
        input_surface=torch.randn(1, 4, lat, lon, device=device),
        input_upper=torch.randn(1, 5, 13, lat, lon, device=device),
        target_surface=torch.randn(1, 4, lat, lon, device=device),
        target_upper=torch.randn(1, 5, 13, lat, lon, device=device),
    )


def test_reorder_roundtrip():
    """Reordering data->fengwu->data should be identity."""
    idx = list(range(5))
    reordered = [idx[i] for i in _DATA_TO_FENGWU]
    restored = [reordered[i] for i in _FENGWU_TO_DATA]
    assert restored == idx


def test_instantiation(model, config):
    assert model is not None
    assert model.config == config


def test_forward_shapes(model):
    from experiments.weather.data import DataHP, DataHPConfig

    ds = DataHP(DataHPConfig(nside=64, driscoll_healy=True))
    res = ds.dh_resolution()
    lat, lon = res["lat"], res["lon"]

    batch = make_batch(lat, lon)
    with torch.no_grad():
        out = model(batch)

    assert "logits_surface" in out
    assert "logits_upper" in out
    assert out["logits_surface"].shape == batch["target_surface"].shape
    assert out["logits_upper"].shape == batch["target_upper"].shape


def test_forward_gradient(model):
    from experiments.weather.data import DataHP, DataHPConfig

    ds = DataHP(DataHPConfig(nside=64, driscoll_healy=True))
    res = ds.dh_resolution()
    lat, lon = res["lat"], res["lon"]

    batch = make_batch(lat, lon)
    out = model(batch)
    loss = out["logits_surface"].mean() + out["logits_upper"].mean()
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad
