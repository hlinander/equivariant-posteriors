#!/usr/bin/env python
"""Terminal tool: inspect elimination-rollout checkpoints.

Ranks the elimination sweep configs by held-out logdet_mae, loads the best
checkpoint (or one chosen by --hash), and prints step-by-step rollout traces on
a few example matrices: the starting matrix, every predicted row operation
(pivot choice + multiplier, vs the oracle), the intermediate matrices, and the
final triangular form with the predicted vs true log|det|.

Usage:
    uv run python experiments/realdet/inspect_rollout.py
    uv run python experiments/realdet/inspect_rollout.py --n 4 --examples 2
    uv run python experiments/realdet/inspect_rollout.py --hash <config_hash>
"""
import argparse

import torch
import duckdb

from lib.paths import get_checkpoint_path
from lib.serialization import instantiate_model
from lib.stable_hash import stable_hash_str
from lib.datasets.real_det_matrix import DataRealDetMatrix
from experiments.realdet.elimination import (
    elimination_configs,
    teacher_configs,
    pivoting_configs,
)


def _candidates():
    out = []
    for fn in (pivoting_configs, teacher_configs, elimination_configs):
        for c in fn():
            tr = c() if callable(c) else c
            out.append(tr)
    return out


def _val_mae(train_config, k=10):
    d = get_checkpoint_path(train_config) / "analytics" / "train_epoch_metric"
    g = str(d / "*.parquet")
    try:
        con = duckdb.connect()
        r = con.execute(
            f"SELECT median(mean) FROM (SELECT mean FROM read_parquet('{g}') "
            f"WHERE dataset_split='val' AND name='logdet_mae' ORDER BY epoch DESC LIMIT {k})"
        ).fetchone()
        return r[0] if r and r[0] is not None else None
    except Exception:
        return None


def _fmt_matrix(m, indent="    "):
    return "\n".join(
        indent + " ".join(f"{v:8.3f}" for v in row) for row in m.tolist()
    )


def _cfg_summary(tr):
    mc = tr.train_config.model_config
    dc = tr.train_config.train_data_config
    return (f"N={dc.n} pivot={mc.pivot} mult={mc.multiplier_param} "
            f"teacher={mc.teacher_mode} feat={mc.input_features} seed={dc.seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="filter to matrix size N")
    ap.add_argument("--hash", type=str, default=None, help="inspect a specific config hash")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--full-matrices", action="store_true",
                    help="print the matrix after every op, not just per-column")
    args = ap.parse_args()

    cands = _candidates()
    ranked = []
    seen = set()
    for tr in cands:
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
        mae = _val_mae(tr.train_config)
        if mae is None:
            continue
        ranked.append((mae, h, tr))
    if not ranked:
        print("No matching checkpoints found.")
        return
    ranked.sort(key=lambda x: x[0])

    print("Ranked candidates (val logdet_mae, lower=better):")
    for mae, h, tr in ranked[:10]:
        print(f"  {mae:7.3f}  {h}  {_cfg_summary(tr)}")
    mae, h, tr = ranked[0]
    print(f"\nInspecting best: {h}  ({_cfg_summary(tr)})  val_logdet_mae={mae:.3f}\n")

    model = instantiate_model(tr.train_config)
    sd = torch.load(get_checkpoint_path(tr.train_config) / "model", map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    dc = tr.train_config.train_data_config
    val = DataRealDetMatrix(type(dc)(**{**dc.__dict__, "validation": True}))

    for e in range(min(args.examples, len(val))):
        a = val.xs[e : e + 1]
        true_ld = float(torch.linalg.slogdet(a)[1].item())
        _, _, _, _, _, _ = model.rollout(a, oracle=True)  # warm (no-op)
        oracle_ld = float(model.rollout(a, oracle=True)[0].item())
        tr_out = model.trace_rollout(a)

        print(f"================ Example {e + 1} (N={dc.n}) ================")
        print("starting matrix A:")
        print(_fmt_matrix(tr_out["initial"]))
        print(f"true log|det| = {true_ld:.4f}   oracle rollout = {oracle_ld:.4f}")
        cur_col = -1
        for s in tr_out["steps"]:
            if s["type"] == "pivot":
                if s["k"] != cur_col:
                    cur_col = s["k"]
                    print(f"-- column {s['k']} --")
                ok = "OK" if s["p"] == s["target_p"] else "MISMATCH"
                if s["p"] == s["k"]:
                    print(f"  pivot: keep row {s['k']} (partial={s['target_p']}) {ok}")
                else:
                    print(f"  pivot: swap row {s['k']} <-> row {s['p']} "
                          f"(partial={s['target_p']}) {ok}")
            else:
                if s["k"] != cur_col:
                    cur_col = s["k"]
                    print(f"-- column {s['k']} --")
                print(f"  row{s['i']} -= {s['c_model']:+.4f} * row{s['k']}   "
                      f"(oracle {s['c_oracle']:+.4f}, |resid| {s['resid']:.2e})")
                if args.full_matrices:
                    print(_fmt_matrix(s["matrix"], indent="        "))
            # per-column snapshot: print matrix after the last elim of each column
            nxt = tr_out["steps"].index(s) + 1
            is_last_of_col = (
                s["type"] == "elim"
                and (nxt >= len(tr_out["steps"]) or tr_out["steps"][nxt]["k"] != s["k"])
            )
            if is_last_of_col and not args.full_matrices:
                print(f"  matrix after column {s['k']}:")
                print(_fmt_matrix(s["matrix"], indent="        "))
        print("final U:")
        print(_fmt_matrix(tr_out["final"]))
        err = abs(tr_out["pred_logdet"] - true_ld)
        print(f"pred log|det| = sum log|diag| = {tr_out['pred_logdet']:.4f}   "
              f"(|err vs true| = {err:.4f})\n")


if __name__ == "__main__":
    main()
