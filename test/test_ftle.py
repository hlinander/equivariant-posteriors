"""Tests for the FTLE (top finite-time Lyapunov exponent) computation in
experiments.grokking.lyapunov_metric: analytic correctness on a linear model,
flat (MLP) and seq (transformer) input shapes, and chunking consistency."""
import torch

import experiments.grokking.lyapunov_metric as lyapunov_metric
from experiments.grokking.lyapunov_metric import _compute_ftle
from lib.datasets.finite_field_det import DataFiniteFieldDet, DataFiniteFieldDetConfig
from lib.models.grok_mlp import GrokMLP, GrokMLPConfig
from lib.models.transformer_encoder import TransformerEncoder, TransformerEncoderConfig


N_SAMPLES = 8


def _dataset(seq):
    config = DataFiniteFieldDetConfig(n=2, p=3, frac=0.5, seed=0, seq=seq)
    return DataFiniteFieldDet(config), DataFiniteFieldDet.data_spec(config)


def _reference_ftle(model, xs):
    """Per-sample log of the top singular value of the input->logits Jacobian,
    computed sample by sample with torch.autograd (independent of the
    vmap/jacrev + Gram-eigh route used by _compute_ftle)."""
    lambdas = []
    for xi in xs:
        jac = torch.autograd.functional.jacobian(
            lambda x: model({"input": x.unsqueeze(0)})["logits"].squeeze(0), xi
        )
        jac = jac.reshape(jac.shape[0], -1)
        lambdas.append(torch.linalg.svdvals(jac)[0].log())
    return torch.stack(lambdas)


class LinearModel(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = torch.nn.Linear(d_in, d_out, bias=True)

    def forward(self, batch):
        return dict(logits=self.linear(batch["input"]))


def test_ftle_linear_model_matches_top_singular_value():
    torch.manual_seed(0)
    model = LinearModel(6, 4)
    xs = torch.randn(N_SAMPLES, 6)
    expected = torch.linalg.svdvals(model.linear.weight)[0].log()
    ftle = _compute_ftle(model, xs)
    assert ftle.shape == (N_SAMPLES,)
    assert torch.allclose(ftle, expected.expand(N_SAMPLES), atol=1e-5)


def test_ftle_mlp_flat_input():
    torch.manual_seed(0)
    dataset, spec = _dataset(seq=False)
    model = GrokMLP(GrokMLPConfig(width=16, depth=2), spec).eval()
    xs = dataset.xs[:N_SAMPLES]
    assert xs.shape == (N_SAMPLES, 2 * 2 * 3)
    ftle = _compute_ftle(model, xs)
    assert ftle.shape == (N_SAMPLES,)
    assert torch.isfinite(ftle).all()
    assert torch.allclose(ftle, _reference_ftle(model, xs), atol=1e-4)


def test_ftle_transformer_seq_input():
    torch.manual_seed(0)
    dataset, spec = _dataset(seq=True)
    model_config = TransformerEncoderConfig(
        embed_d=8,
        mlp_dim=16,
        num_layers=2,
        num_heads=2,
        softmax=True,
        activation="relu",
    )
    model = TransformerEncoder(model_config, spec).eval()
    xs = dataset.xs[:N_SAMPLES]
    assert xs.shape == (N_SAMPLES, 2 * 2, 3)
    ftle = _compute_ftle(model, xs)
    assert ftle.shape == (N_SAMPLES,)
    assert torch.isfinite(ftle).all()
    assert torch.allclose(ftle, _reference_ftle(model, xs), atol=1e-4)


def test_ftle_chunking_consistent(monkeypatch):
    torch.manual_seed(0)
    model = LinearModel(5, 3)
    xs = torch.randn(7, 5)
    full = _compute_ftle(model, xs)
    monkeypatch.setattr(lyapunov_metric, "JACOBIAN_CHUNK", 3)
    chunked = _compute_ftle(model, xs)
    assert torch.allclose(full, chunked, atol=1e-6)
