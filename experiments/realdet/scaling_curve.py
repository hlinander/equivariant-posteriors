#!/usr/bin/env python
"""Terminal tool: post-hoc test-time-compute scaling curve for elimination
checkpoints. For each finished run it loads the final model and evaluates
free-rollout logdet_mae across a refinement R-grid (0,1,2,4,8 x base) on the
held-out set, plus the per-step mult_loss. Groups by (N, step_arch) so the
mlp-vs-transformer comparison and the refinement scaling are read off directly.

Usage:
    uv run python experiments/realdet/scaling_curve.py --n 3
    uv run python experiments/realdet/scaling_curve.py            # all N
"""
import argparse
from collections import defaultdict

import torch
import numpy as np

from lib.paths import get_checkpoint_path
from lib.serialization import instantiate_model
from lib.datasets.real_det_matrix import DataRealDetMatrix
from experiments.realdet.elimination import arch_configs, refine_configs


def _load(tr):
    d = get_checkpoint_path(tr.train_config)
    if not (d / "model").is_file():
        return None
    try:
        ep = torch.load(d / "epoch", map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not (isinstance(ep, int) and ep >= tr.epochs):
        return None
    m = instantiate_model(tr.train_config)
    m.load_state_dict(torch.load(d / "model", map_location="cpu"))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=2048)
    args = ap.parse_args()

    cfgs = arch_configs() + refine_configs()
    seen = set()
    rows = defaultdict(lambda: defaultdict(list))
    mults = [0, 1, 2, 4, 8]
    for c in cfgs:
        tr = c() if callable(c) else c
        dc = tr.train_config.train_data_config
        mc = tr.train_config.model_config
        h = (dc.n, mc.step_arch, dc.seed)
        if h in seen:
            continue
        seen.add(h)
        if args.n is not None and dc.n != args.n:
            continue
        m = _load(tr)
        if m is None:
            continue
        val = DataRealDetMatrix(type(dc)(**{**dc.__dict__, "validation": True}))
        a = val.xs[: args.n_eval]
        tgt = val.ys[: args.n_eval].squeeze(-1)
        base = dc.n * (dc.n - 1) // 2
        key = (dc.n, mc.step_arch)
        with torch.no_grad():
            for mult in mults:
                ld = m.rollout(a, refine_steps=base * mult)[0]
                rows[key][mult].append((ld - tgt).abs().mean().item())
            rows[key]["mult_loss"].append(float(m.rollout(a)[4]))

    if not rows:
        print("No finished checkpoints found.")
        return
    print("logdet_mae vs refinement R (xbase), mean over seeds, + per-step mult_loss")
    print(f"{'N':>2} {'arch':>11} | " + "  ".join(f"R{m}x" for m in mults) + " | mult_loss  seeds")
    for (n, arch) in sorted(rows):
        r = rows[(n, arch)]
        cells = "  ".join(f"{np.mean(r[m]):.3f}" for m in mults)
        ml = np.mean(r["mult_loss"])
        print(f"{n:>2} {arch:>11} | {cells} | {ml:9.4f}  {len(r[0])}")


if __name__ == "__main__":
    main()
