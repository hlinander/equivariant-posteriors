#!/usr/bin/env python
"""
Diagnose where HP model errors are concentrated relative to a nohp baseline.

Runs full ssp245 test-set inference for ONE HP model and ONE nohp baseline,
then computes per-sample RMSE (temporal analysis) and per-pixel spatial RMSE
(geographic analysis). Results are saved to a .npz file for fast notebook
re-analysis without re-running inference.

Edit the CONFIG block below, then run from the repo root:
    python experiments/climate/evaluation/diagnose_error_patterns.py

Optional epoch overrides (use best-val epoch from DuckDB by default):
    python ... --hp-epoch 150 --nohp-epoch 200
"""
import sys
import argparse
import copy
import glob
import io
import contextlib
from pathlib import Path

import numpy as np
import torch
import healpix
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ─── CONFIG ──────────────────────────────────────────────────────────────────
HP_CONFIG   = "/home/x_tagty/equivariant-posteriors/experiments/climate/persisted_configs/train_climate_pear_multiseed.py"
NOHP_CONFIG = "/home/x_tagty/equivariant-posteriors/experiments/climate/persisted_configs/train_unet_nohp_multiseed.py"
CLIMATE_MODEL_IDX = 1    # index into CLIMATE_MODELS list (1 = BCC-CSM2-MR)
SEED              = 0
HP_LR             = 2e-4
HP_EPOCH          = None  # None → best-val epoch from DuckDB
NOHP_EPOCH        = None  # None → best-val epoch from DuckDB
_REPO_ROOT        = Path(__file__).resolve().parents[3]
OUTPUT_DIR        = _REPO_ROOT / "results"
# ─────────────────────────────────────────────────────────────────────────────

from experiments.climate.evaluation.evaluate_climate_hp import load_create_config as _load_hp_config
from experiments.climate.evaluation.evaluate_climate_nohp import load_create_config as _load_nohp_config
from experiments.climate.evaluation.sample_predictions import _build_model_and_test_ds, _register_factories as _reg_hp
from experiments.climate.evaluation.sample_predictions_nohp import _build_model_and_test_ds_nohp, _register_factories as _reg_nohp
from experiments.climate.evaluation.timestamp_utils import sample_id_to_timestamp
from experiments.climate.data.climateset_data_hp import ClimatesetDataHP
from experiments.climate.data.climateset_data_no_hp import ClimatesetData
from lib.paths import get_checkpoint_path


# ─── Epoch auto-selection (mirrors notebook helpers) ─────────────────────────

def _query_all_steps(checkpoint_dir, metric_name):
    import duckdb
    rows = []
    for db_path in sorted(glob.glob(str(checkpoint_dir / "duck_*.db"))):
        try:
            con = duckdb.connect(db_path, read_only=True)
            rows.extend(con.execute(
                "SELECT step, mean_float FROM checkpoint_sample_metric "
                "WHERE name = ? AND mean_float IS NOT NULL",
                [metric_name],
            ).fetchall())
            con.close()
        except Exception:
            pass
    return rows


def get_best_epoch_hp(curried, seed=0):
    train_run = curried(ensemble_id=seed)
    checkpoint_dir = get_checkpoint_path(train_run.train_config)
    rows = _query_all_steps(checkpoint_dir, "rmse_overall")
    if not rows:
        return None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_ds = ClimatesetDataHP(train_run.train_config.train_data_config)
    n_train = len(train_ds)
    best_epoch, best_val = None, float("inf")
    for step, val in rows:
        epoch = round(step / n_train)
        if (checkpoint_dir / f"model_epoch_{epoch:04d}").is_file() and val < best_val:
            best_val, best_epoch = val, epoch
    print(f"  HP   best epoch: {best_epoch}  (val rmse={best_val:.4e})")
    return best_epoch


def get_best_epoch_nohp(curried, seed=0):
    train_run = curried(ensemble_id=seed)
    checkpoint_dir = get_checkpoint_path(train_run.train_config)
    rows = _query_all_steps(checkpoint_dir, "rmse_latw_overall")
    if not rows:
        return None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_ds = ClimatesetData(train_run.train_config.train_data_config)
    n_train = len(train_ds)
    best_epoch, best_val = None, float("inf")
    for step, val in rows:
        epoch = round(step / n_train)
        if (checkpoint_dir / f"model_epoch_{epoch:04d}").is_file() and val < best_val:
            best_val, best_epoch = val, epoch
    print(f"  nohp best epoch: {best_epoch}  (val rmse={best_val:.4e})")
    return best_epoch


# ─── Inference helpers ────────────────────────────────────────────────────────

def _run_hp_inference(create_config, epoch, seed):
    """Run HP model on full test set. Returns raw preds/tgts and sample IDs."""
    _reg_hp()
    model, test_ds, stats, data_cfg, device_id = _build_model_and_test_ds(create_config, epoch, seed)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=12, shuffle=False, drop_last=False)

    all_preds, all_tgts, all_sample_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="HP inference"):
            batch_dev = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            if batch_dev["input"].dim() == 4:
                batch_dev["input"]  = batch_dev["input"].squeeze(1)
                batch_dev["target"] = batch_dev["target"].squeeze(1)
            out = model(batch_dev)
            all_preds.append(out["logits_output"].cpu())
            all_tgts.append(batch_dev["target"].cpu())
            all_sample_ids.extend(batch["sample_id"].tolist())

    preds = torch.cat(all_preds, dim=0)   # (N, C, P) or (N, T, C, P)
    tgts  = torch.cat(all_tgts,  dim=0)

    # Seq-to-seq: flatten time into batch and expand sample IDs accordingly
    if preds.dim() == 4:
        N, T, C, P = preds.shape
        preds = preds.reshape(N * T, C, P)
        tgts  = tgts.reshape(N * T, C, P)
        expanded = []
        for sid in all_sample_ids:
            expanded.extend(range(int(sid), int(sid) + T))
        all_sample_ids = expanded

    return preds.float(), tgts.float(), all_sample_ids, data_cfg, stats


def _run_nohp_inference(create_config, epoch, seed):
    """Run nohp model on full test set. Returns raw preds/tgts and sample IDs."""
    _reg_nohp()
    model, test_ds, stats, data_cfg, device_id = _build_model_and_test_ds_nohp(create_config, epoch, seed)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False, drop_last=False)
    lats = getattr(test_ds, "lats", None)

    all_preds, all_tgts, all_sample_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="nohp inference"):
            batch_dev = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            out = model(batch_dev)
            all_preds.append(out["logits_output"].cpu())
            all_tgts.append(batch_dev["target"].cpu())
            all_sample_ids.extend(batch["sample_id"].tolist())

    preds = torch.cat(all_preds, dim=0)   # (N, C, H, W) or (N, T, C, H, W)
    tgts  = torch.cat(all_tgts,  dim=0)

    seq_len = getattr(data_cfg, "seq_len", 1)
    if preds.dim() == 5:
        N, T, C, H, W = preds.shape
        preds = preds.reshape(N * T, C, H, W)
        tgts  = tgts.reshape(N * T, C, H, W)
        expanded = []
        for sid in all_sample_ids:
            expanded.extend(range(int(sid), int(sid) + T))
        all_sample_ids = expanded

    return preds.float(), tgts.float(), all_sample_ids, data_cfg, lats


# ─── Metrics ─────────────────────────────────────────────────────────────────

def _per_sample_rmse(preds, tgts):
    """Unweighted per-sample RMSE. preds/tgts: (N, C, ...). Returns (N, C)."""
    sq = (preds - tgts) ** 2
    # flatten spatial dims
    N, C = preds.shape[0], preds.shape[1]
    sq_flat = sq.reshape(N, C, -1)   # (N, C, pixels)
    return torch.sqrt(sq_flat.mean(dim=-1)).numpy()   # (N, C)


def _spatial_rmse(preds, tgts):
    """Per-pixel RMSE averaged over all samples. Returns (C, pixels...)."""
    sq = (preds - tgts) ** 2
    return torch.sqrt(sq.mean(dim=0)).numpy()   # (C, P) or (C, H, W)


def _hp_to_latlon(hp_map, nside, target_lats, target_lons):
    """Nearest-neighbour reproject (P,) HEALPix array to (n_lat, n_lon) grid."""
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
    pix = healpix.ang2pix(nside, lon_grid.ravel(), lat_grid.ravel(), lonlat=True, nest=True)
    return hp_map[pix].reshape(len(target_lats), len(target_lons))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hp-epoch",   type=int, default=None)
    parser.add_argument("--nohp-epoch", type=int, default=None)
    args = parser.parse_args()

    # ── Build curried configs ──
    hp_base   = _load_hp_config(HP_CONFIG)
    nohp_base = _load_nohp_config(NOHP_CONFIG)

    hp_curried   = lambda ensemble_id, **kw: hp_base(
        ensemble_id=ensemble_id, climate_model_idx=CLIMATE_MODEL_IDX, lr=HP_LR, **kw)
    nohp_curried = lambda ensemble_id, **kw: nohp_base(
        ensemble_id=ensemble_id, climate_model_idx=CLIMATE_MODEL_IDX, **kw)

    hp_epoch   = args.hp_epoch   or HP_EPOCH   or get_best_epoch_hp(hp_curried,   seed=SEED)
    nohp_epoch = args.nohp_epoch or NOHP_EPOCH or get_best_epoch_nohp(nohp_curried, seed=SEED)

    if hp_epoch is None:
        raise RuntimeError("Could not determine HP epoch. Set HP_EPOCH or --hp-epoch.")
    if nohp_epoch is None:
        raise RuntimeError("Could not determine nohp epoch. Set NOHP_EPOCH or --nohp-epoch.")

    print(f"\nRunning with HP epoch={hp_epoch}, nohp epoch={nohp_epoch}, "
          f"climate_model_idx={CLIMATE_MODEL_IDX}, seed={SEED}\n")

    # ── HP inference ──
    print("=== HP model ===")
    hp_preds, hp_tgts, hp_ids, hp_data_cfg, hp_stats = _run_hp_inference(
        hp_curried, hp_epoch, SEED)
    N_hp = hp_preds.shape[0]
    nside = hp_data_cfg.nside
    cm_name = hp_data_cfg.climate_model

    hp_per_sample = _per_sample_rmse(hp_preds, hp_tgts)    # (N, C)
    hp_spatial    = _spatial_rmse(hp_preds, hp_tgts)        # (C, P)
    var_names = list(hp_data_cfg.output_vars)

    hp_years  = np.zeros(N_hp, dtype=int)
    hp_months = np.zeros(N_hp, dtype=int)
    for i, sid in enumerate(hp_ids):
        _, yr, mo = sample_id_to_timestamp(sid, hp_data_cfg)
        hp_years[i]  = yr
        hp_months[i] = mo

    # Reproject HP spatial RMSE to lat/lon
    target_lons = np.linspace(0, 360, 144, endpoint=False)
    target_lats = np.linspace(-90, 90, 96)   # fallback; will be overwritten from nohp lats

    # ── nohp inference ──
    print("\n=== nohp model ===")
    nohp_preds, nohp_tgts, nohp_ids, nohp_data_cfg, nohp_lats = _run_nohp_inference(
        nohp_curried, nohp_epoch, SEED)
    N_nohp = nohp_preds.shape[0]

    nohp_per_sample = _per_sample_rmse(nohp_preds, nohp_tgts)   # (N, C)
    nohp_spatial    = _spatial_rmse(nohp_preds, nohp_tgts)       # (C, H, W)

    nohp_years  = np.zeros(N_nohp, dtype=int)
    nohp_months = np.zeros(N_nohp, dtype=int)
    for i, sid in enumerate(nohp_ids):
        _, yr, mo = sample_id_to_timestamp(sid, nohp_data_cfg)
        nohp_years[i]  = yr
        nohp_months[i] = mo

    # Use lats from nohp dataset if available
    if nohp_lats is not None:
        target_lats = np.array(nohp_lats)

    # Reproject HP spatial RMSE → lat/lon
    n_lat = nohp_spatial.shape[-2] if nohp_spatial.ndim == 3 else 96
    n_lon = nohp_spatial.shape[-1] if nohp_spatial.ndim == 3 else 144
    target_lats_use = target_lats if len(target_lats) == n_lat else np.linspace(-90, 90, n_lat)
    target_lons_use = np.linspace(0, 360, n_lon, endpoint=False)

    C = len(var_names)
    hp_spatial_ll = np.zeros((C, n_lat, n_lon), dtype=np.float32)
    for c in range(C):
        hp_spatial_ll[c] = _hp_to_latlon(hp_spatial[c], nside, target_lats_use, target_lons_use)

    # ── Align per-sample arrays by timestamp ─────────────────────────────────
    # Both test sets cover the same scenario/years but may have different N if
    # seq_len differs. Build a common index by (year, month).
    hp_key   = {(yr, mo): i for i, (yr, mo) in enumerate(zip(hp_years, hp_months))}
    nohp_key = {(yr, mo): i for i, (yr, mo) in enumerate(zip(nohp_years, nohp_months))}
    common_keys = sorted(set(hp_key) & set(nohp_key))

    aligned_years   = np.array([k[0] for k in common_keys], dtype=int)
    aligned_months  = np.array([k[1] for k in common_keys], dtype=int)
    aligned_hp      = np.array([hp_per_sample[hp_key[k]]   for k in common_keys])   # (M, C)
    aligned_nohp    = np.array([nohp_per_sample[nohp_key[k]] for k in common_keys]) # (M, C)

    # ── Save ─────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"error_diagnosis_{cm_name}_seed{SEED}.npz"
    np.savez(
        out_path,
        hp_per_sample_rmse   = aligned_hp,           # (M, C)
        nohp_per_sample_rmse = aligned_nohp,         # (M, C)
        years                = aligned_years,         # (M,)
        months               = aligned_months,        # (M,)
        hp_spatial_rmse_ll   = hp_spatial_ll,         # (C, n_lat, n_lon) lat/lon grid
        nohp_spatial_rmse_ll = nohp_spatial,          # (C, n_lat, n_lon)
        var_names            = np.array(var_names),
        target_lats          = target_lats_use,
        target_lons          = target_lons_use,
        cm_name              = cm_name,
        hp_epoch             = hp_epoch,
        nohp_epoch           = nohp_epoch,
        seed                 = SEED,
    )
    print(f"\nSaved → {out_path}")

    # ── Diagnostic summary ───────────────────────────────────────────────────
    _print_summary(aligned_hp, aligned_nohp, aligned_years, aligned_months, var_names)


def _print_summary(hp, nohp, years, months, var_names):
    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for ci, var in enumerate(var_names):
        hp_c    = hp[:, ci]
        nohp_c  = nohp[:, ci]
        delta   = hp_c - nohp_c   # positive = HP is worse

        print(f"\n{'='*60}")
        print(f"  Variable: {var}")
        print(f"  Overall mean  HP RMSE={hp_c.mean():.4f}  nohp RMSE={nohp_c.mean():.4f}"
              f"  Δ={delta.mean():.4f}")
        print(f"  Samples HP>nohp: {(delta>0).sum()} / {len(delta)}")

        print(f"\n  Top 15 samples where HP is worst relative to nohp:")
        print(f"  {'Rank':>4}  {'Year':>4}  {'Mon':>3}  {'HP':>8}  {'nohp':>8}  {'Δ':>8}")
        for rank, idx in enumerate(np.argsort(delta)[::-1][:15], 1):
            print(f"  {rank:>4}  {years[idx]:>4}  {MONTH_NAMES[months[idx]-1]:>3}"
                  f"  {hp_c[idx]:>8.4f}  {nohp_c[idx]:>8.4f}  {delta[idx]:>8.4f}")

        print(f"\n  Seasonal breakdown (mean Δ per calendar month):")
        print(f"  {'Mon':>3}  {'HP':>8}  {'nohp':>8}  {'Δ':>8}  {'n':>4}")
        for mo in range(1, 13):
            mask = months == mo
            if mask.sum() == 0:
                continue
            print(f"  {MONTH_NAMES[mo-1]:>3}  {hp_c[mask].mean():>8.4f}"
                  f"  {nohp_c[mask].mean():>8.4f}  {delta[mask].mean():>8.4f}"
                  f"  {mask.sum():>4}")

        decades = sorted(set((y // 10) * 10 for y in years))
        print(f"\n  Per-decade breakdown:")
        print(f"  {'Decade':>7}  {'HP':>8}  {'nohp':>8}  {'Δ':>8}  {'n':>4}")
        for dec in decades:
            mask = (years >= dec) & (years < dec + 10)
            print(f"  {dec}s  {hp_c[mask].mean():>8.4f}"
                  f"  {nohp_c[mask].mean():>8.4f}  {delta[mask].mean():>8.4f}"
                  f"  {mask.sum():>4}")


if __name__ == "__main__":
    main()
