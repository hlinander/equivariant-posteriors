#!/usr/bin/env python
"""Terminal tool: inspect matrix-operator-transformer checkpoints.

Loads an operator-transformer DET checkpoint (best by val free_logdet_mae, or
chosen by --hash) and prints step-by-step free rollouts on example matrices:
the starting matrix, each predicted operator (||M-I||, stop prob, the residual
||tril(A,-1)|| shrinking), STOP firing, the final triangular form, and the
predicted vs true log|det|.

Usage:
    uv run python experiments/realdet/inspect_operator.py --n 8
    uv run python experiments/realdet/inspect_operator.py --n 4 --examples 2 --show-ops
    uv run python experiments/realdet/inspect_operator.py --hash <config_hash>
"""
import argparse

import torch
import duckdb

from lib.paths import get_checkpoint_path
from lib.serialization import instantiate_model
from lib.stable_hash import stable_hash_str
from lib.datasets.real_det_matrix import DataRealDetMatrix
from experiments.realdet.operator_transformer import (
    det_configs, data_scaling_configs, n8_configs, n8_huge_configs,
)


def _candidates():
    out = []
    for fn in (det_configs, data_scaling_configs, n8_configs, n8_huge_configs):
        for c in fn():
            out.append(c() if callable(c) else c)
    return out


def _val_free_mae(train_config):
    d = get_checkpoint_path(train_config) / "analytics" / "checkpoint_sample_metric"
    g = str(d / "*.parquet")
    try:
        con = duckdb.connect()
        r = con.execute(
            f"SELECT mean_float FROM read_parquet('{g}') WHERE name='free_logdet_mae_val' "
            f"AND mean_float IS NOT NULL ORDER BY step DESC LIMIT 1"
        ).fetchone()
        return r[0] if r else None
    except Exception:
        return None


def _fmt(m, indent="    "):
    return "\n".join(indent + " ".join(f"{v:7.2f}" for v in row) for row in m.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--hash", type=str, default=None)
    ap.add_argument("--examples", type=int, default=2)
    ap.add_argument("--show-ops", action="store_true", help="print each operator matrix M")
    args = ap.parse_args()

    ranked = []
    seen = set()
    for tr in _candidates():
        h = stable_hash_str(tr.train_config)
        if h in seen:
            continue
        seen.add(h)
        if args.hash and h != args.hash:
            continue
        if args.n is not None and tr.train_config.train_data_config.n != args.n:
            continue
        if not (get_checkpoint_path(tr.train_config) / "model").is_file():
            continue
        mae = _val_free_mae(tr.train_config)
        if mae is None:
            continue
        ranked.append((mae, h, tr))
    if not ranked:
        print("No matching operator-transformer checkpoints found.")
        return
    ranked.sort(key=lambda x: x[0])
    mae, h, tr = ranked[0]
    dc = tr.train_config.train_data_config
    n = dc.n
    print(f"Inspecting {h}: N={n} n_train={dc.n_train} seed={dc.seed} "
          f"val_free_logdet_mae={mae:.3f}\n")

    model = instantiate_model(tr.train_config)
    model.load_state_dict(torch.load(get_checkpoint_path(tr.train_config) / "model", map_location="cpu"))
    model.eval()
    val = DataRealDetMatrix(type(dc)(**{**dc.__dict__, "validation": True}))

    for e in range(min(args.examples, len(val))):
        a = val.xs[e : e + 1]
        true_ld = float(torch.linalg.slogdet(a)[1])
        tr_out = model.trace_rollout(a)
        print(f"================ N={n} example {e + 1} (true_ops={2*(n-1)}) ================")
        print("starting matrix A:")
        print(_fmt(tr_out["initial"]))
        print(f"start ||tril(A,-1)|| = {torch.tril(tr_out['initial'], -1).norm():.3f}")
        print(f"{'step':>4} {'stop_p':>7} {'||M-I||':>8} {'lower_resid':>12}")
        for i, s in enumerate(tr_out["steps"]):
            if s["stopped"]:
                print(f"{i+1:>4} {s['stop_prob']:>7.3f} {s['op_norm']:>8.3f}   <- STOP")
                continue
            print(f"{i+1:>4} {s['stop_prob']:>7.3f} {s['op_norm']:>8.3f} {s['lower_resid']:>12.3f}")
            if args.show_ops:
                print(_fmt(s["M"], indent="        op M="))
        print("final state (≈ upper-triangular):")
        print(_fmt(tr_out["final"]))
        err = abs(tr_out["pred_logdet"] - true_ld)
        print(f"pred log|det| = sum log|diag| = {tr_out['pred_logdet']:.4f}   "
              f"true = {true_ld:.4f}   |err| = {err:.4f}\n")


if __name__ == "__main__":
    main()
