"""Tests for lib.optimizers.adamw_no_decay_norm_bias: normalization params and
biases are exempt from weight decay, while unrelated 1-d parameters still
decay (the filter is by module type / name, not tensor dimensionality)."""
import torch

from lib.optimizers import adamw_no_decay_norm_bias


class ModelWithGain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.gain = torch.nn.Parameter(torch.ones(4))  # 1-d, not a norm param


def _weight_decay_by_param(opt):
    return {
        id(p): group["weight_decay"]
        for group in opt.param_groups
        for p in group["params"]
    }


def test_norm_and_bias_exempt_other_1d_decays():
    model = ModelWithGain()
    opt = adamw_no_decay_norm_bias(model, lr=1e-3, weight_decay=0.5)
    wd = _weight_decay_by_param(opt)
    assert wd[id(model.linear.weight)] == 0.5
    assert wd[id(model.linear.bias)] == 0.0
    assert wd[id(model.norm.weight)] == 0.0
    assert wd[id(model.norm.bias)] == 0.0
    assert wd[id(model.gain)] == 0.5
    assert len(wd) == sum(1 for _ in model.parameters())


def test_takes_model_marker():
    assert adamw_no_decay_norm_bias.takes_model is True


def test_attention_in_proj_bias_exempt():
    model = torch.nn.TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=True
    )
    opt = adamw_no_decay_norm_bias(model, lr=1e-3, weight_decay=0.5)
    wd = _weight_decay_by_param(opt)
    assert wd[id(model.self_attn.in_proj_bias)] == 0.0
    assert wd[id(model.self_attn.in_proj_weight)] == 0.5
    decay_n = sum(1 for v in wd.values() if v == 0.5)
    # exactly the four matmul weights decay: in_proj, out_proj, linear1, linear2
    assert decay_n == 4
