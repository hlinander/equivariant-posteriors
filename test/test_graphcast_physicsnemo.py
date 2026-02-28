import torch
import pytest

try:
    from experiments.weather.models.graphcast_physicsnemo import (
        GraphCastPhysicsNemo,
        GraphCastPhysicsNemoConfig,
        N_SURFACE,
        N_UPPER_VARS,
        N_PRESSURE_LEVELS,
        N_TOTAL,
    )
    from experiments.weather.data import DataSpecHP

    _graphcast_available = True
except ImportError:
    _graphcast_available = False

pytestmark = pytest.mark.skipif(
    not _graphcast_available,
    reason="GraphCast dependencies not available (transformer_engine, sklearn, torch_geometric)",
)


@pytest.fixture
def config():
    return GraphCastPhysicsNemoConfig(nside=64, mesh_level=3, processor_layers=3, hidden_dim=64)


@pytest.fixture
def data_spec():
    return DataSpecHP(nside=64, n_surface=4, n_upper=5)


@pytest.fixture
def model(config, data_spec):
    return GraphCastPhysicsNemo(config, data_spec)


def make_batch(lat, lon, device="cpu"):
    return dict(
        input_surface=torch.randn(1, 4, lat, lon, device=device),
        input_upper=torch.randn(1, 5, 13, lat, lon, device=device),
        target_surface=torch.randn(1, 4, lat, lon, device=device),
        target_upper=torch.randn(1, 5, 13, lat, lon, device=device),
    )


def test_channel_constants():
    assert N_TOTAL == N_SURFACE + N_UPPER_VARS * N_PRESSURE_LEVELS
    assert N_TOTAL == 69


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
