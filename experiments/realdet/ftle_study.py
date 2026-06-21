#!/usr/bin/env python
"""Terminal tool: FTLE / Jacobian study for the matrix-operator transformer.

The model computes log|det| by a free operator rollout; its sensitivity
d log|det| / dA should equal Jacobi's analytic gradient A^{-T}. We compute that
gradient by backprop through the (grad-enabled) rollout and compare to A^{-T}:
cosine alignment (mean) and relative Frobenius error (median). Because this
model runs the actual elimination, we expect far better alignment than the MLP
regression baseline (which plateaued at cos 0.87 at N=4, ~0.1 at N=5).

Usage: uv run python experiments/realdet/ftle_study.py [--n 4] [--n-eval 256]
"""
import argparse

import torch
import numpy as np

from lib.paths import get_checkpoint_path
from lib.serialization import instantiate_model
from lib.datasets.real_det_matrix import DataRealDetMatrix
from experiments.realdet.operator_transformer import det_configs


def _load(tr):
    d = get_checkpoint_path(tr.train_config)
    if not (d / "model").is_file():
        return None
    m = instantiate_model(tr.train_config)
    m.load_state_dict(torch.load(d / "model", map_location="cpu"))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("d log|det|/dA  vs  Jacobi A^{-T}   (operator transformer, free rollout)")
    print(f"{'N':>2} {'free_logdet_mae':>15} {'jac_cos':>9} {'jac_relerr':>11}")
    for c in det_configs():
        tr = c() if callable(c) else c
        dc = tr.train_config.train_data_config
        if (args.n is not None and dc.n != args.n) or dc.seed != args.seed:
            continue
        m = _load(tr)
        if m is None:
            continue
        n = dc.n
        true_ops = 2 * (n - 1)
        val = DataRealDetMatrix(type(dc)(**{**dc.__dict__, "validation": True}))
        a = val.xs[: args.n_eval].clone().requires_grad_(True)
        tgt = val.ys[: args.n_eval].squeeze(-1)

        logdet = m.rollout_logdet(a, true_ops)
        (grad,) = torch.autograd.grad(logdet.sum(), a)  # (B, n, n) = d logdet_i / dA_i
        with torch.no_grad():
            ainv_t = torch.linalg.inv(a.detach()).transpose(-1, -2)
            g = grad.flatten(1)
            r = ainv_t.flatten(1)
            cos = (g * r).sum(1) / (g.norm(dim=1).clamp_min(1e-30) * r.norm(dim=1).clamp_min(1e-30))
            relerr = (g - r).norm(dim=1) / r.norm(dim=1).clamp_min(1e-30)
            mae = (logdet.detach() - tgt).abs().mean().item()
        print(f"{n:>2} {mae:>15.4f} {cos.mean().item():>9.3f} {relerr.median().item():>11.3f}")


if __name__ == "__main__":
    main()
