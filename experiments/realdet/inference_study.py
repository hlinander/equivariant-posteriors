#!/usr/bin/env python
"""Terminal tool: test-time inference study for the matrix-operator transformer.

For each trained DET checkpoint it forces a long operator rollout (ignoring
STOP) and reports, per step:
  - logdet_mae   : running free-rollout error  -> does more compute help / plateau / hurt?
  - stop_prob    : the model's halt probability -> when does it want to stop?
  - ||M-I||      : operator magnitude           -> does it identity-pad past the natural stop?
It also reports the natural (STOP-halted) op count distribution and whether it
adapts to instance difficulty (correlation of n_ops with |log|det||).

Usage:
    uv run python experiments/realdet/inference_study.py
    uv run python experiments/realdet/inference_study.py --n 4 --extra 6
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
    ap.add_argument("--extra", type=int, default=6, help="steps to force past the true op count")
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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
        a = val.xs[: args.n_eval]
        tgt = val.ys[: args.n_eval].squeeze(-1)

        # model has only max_ops positional slots (= true_ops here), so we can't
        # force past the natural length without retraining with slack.
        n_steps = min(true_ops + args.extra, m.max_ops)
        stop_probs, op_norms, logdets = m.free_rollout_trace(a, n_steps)
        mae = [float((ld - tgt).abs().mean()) for ld in logdets]

        print(f"\n===== N={n} (true_ops={true_ops}, seed={dc.seed}) =====")
        print(f"{'step':>4} {'logdet_mae':>11} {'stop_prob':>10} {'||M-I||':>9}")
        for t in range(n_steps):
            mark = " <- true op count" if t + 1 == true_ops else ""
            print(f"{t+1:>4} {mae[t]:>11.4f} {stop_probs[t]:>10.3f} {op_norms[t]:>9.3f}{mark}")

        # natural halting + adaptivity
        ld, n_ops, _ = m.free_rollout(a)
        absld = tgt.abs()
        c = float(np.corrcoef(n_ops.cpu().numpy(), absld.cpu().numpy())[0, 1])
        print(f"natural halt: avg_ops={n_ops.float().mean():.2f} (true {true_ops}), "
              f"std={n_ops.float().std():.2f}, corr(n_ops, |log|det||)={c:+.3f}")


if __name__ == "__main__":
    main()
