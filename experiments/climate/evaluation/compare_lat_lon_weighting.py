"""
Compare latitude-weighted vs longitude-weighted vs unweighted RMSE.

Usage (mirrors evaluate_all_baselines_all_checkpoints_multiseed.py):
    CONFIG=path/to/config.py EPOCH=50 python -m experiments.climate.evaluation.compare_lat_lon_weighting
    CONFIG=path/to/config.py N_SEEDS=2 NUM_VARIANTS=3 python -m experiments.climate.evaluation.compare_lat_lon_weighting

Env vars:
    CONFIG         (required) path to the train config .py file
    EPOCH          single epoch; if omitted sweeps 0..epochs step keep_nth_epoch_checkpoints
    EVAL_EVERY     override step between evaluated epochs
    N_SEEDS        seeds to sweep (default 5)
    NUM_VARIANTS   climate model variants to sweep (default 15)
    CLIMATE_MODEL_IDX  first variant index (default 0)

Output: terminal only — nothing is written to DuckDB.
"""

import os
import copy
import sys
import importlib
import importlib.util
from pathlib import Path

# Ensure the project root is first in sys.path so `import env` inside
# lib/compute_env.py always finds the real env.py (with the correct checkpoint
# paths) regardless of what directory this script is invoked from.
_project_root = Path(__file__).resolve().parents[3]  # .../equivariant-posteriors/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch

import lib.compute_env as _compute_env
_compute_env.env().paths.checkpoints = Path("/proj/heal_pangu/eqp_climate/checkpoints")

from lib.ddp import ddp_setup
from lib.serialization import deserialize_model, DeserializeConfig
from lib.generic_ablation import get_config_grid

import lib.data_factory as data_factory
import lib.model_factory as model_factory

from experiments.climate.data.climateset_data_no_hp import ClimatesetConfig, ClimatesetData
from experiments.climate.data.climateset_data_no_hp import load_training_stats_from_config
from experiments.climate.adapted_climateset_baselines.adapted_models.unet import UNetConfig, UNet
from experiments.climate.adapted_climateset_baselines.adapted_models.cnn_lstm import (
    CNNLSTMConfig,
    CNNLSTM_ClimateBench,
)
from experiments.climate.adapted_climateset_baselines.adapted_models.climax.climax_module import (
    ClimaXConfig,
    ClimaX,
)
from experiments.climate.models.GRU_wrapper import GRUTemporalWrapperConfig, GRUTemporalWrapper


# ---------------------------------------------------------------------------
# Weighting functions
# ---------------------------------------------------------------------------
# Data layout coming in from the dataset: (N, lat, lon)
#   axis -2 = lat (size 96 for 250 km grid)
#   axis -1 = lon (size 144 for 250 km grid)

def correct_lat_weighted_rmse(preds: np.ndarray, y: np.ndarray, lats: np.ndarray) -> float:
    """Correct latitude-weighted RMSE.
    Applies cos(lat) weights to the lat axis (axis -2) using real lat values.
    This is what the metric should have been computing all along.
    """
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.mean()
    weights = weights[np.newaxis, :, np.newaxis]  # (1, lat, 1) → broadcasts over (N, lat, lon)
    error = (((preds - y) ** 2) * weights).mean()
    return float(np.sqrt(error))


def old_buggy_lat_weighted_rmse(preds: np.ndarray, y: np.ndarray) -> float:
    """OLD (buggy) latitude-weighted RMSE — what was computed before the fix.
    Applied cos(linspace(-90, 90, n_lon)) weights to the lon axis (axis -1)
    instead of the lat axis, because y.shape[-1] was used instead of y.shape[-2].
    """
    n_lon = y.shape[-1]
    angles = np.linspace(-90, 90, n_lon)
    weights = np.cos(np.deg2rad(angles))
    weights = weights / weights.mean()
    # shape (1, 1, lon) → broadcasts over (N, lat, lon), weighting lon not lat
    weights = weights[np.newaxis, np.newaxis, :]
    error = (((preds - y) ** 2) * weights).mean()
    return float(np.sqrt(error))


def unweighted_rmse(preds: np.ndarray, y: np.ndarray) -> float:
    """Plain RMSE with no spatial weighting."""
    return float(np.sqrt(((preds - y) ** 2).mean()))


# ---------------------------------------------------------------------------
# Model loading / prediction collection (no DB writes)
# ---------------------------------------------------------------------------

def _collect_predictions(model, dataloader, device_id):
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch_device = {
                k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            output = model(batch_device)
            all_preds.append(output["logits_output"].cpu())
            all_tgts.append(batch_device["target"].cpu())

    preds = torch.cat(all_preds, dim=0)
    tgts = torch.cat(all_tgts, dim=0)

    # Seq-to-seq: (N, T, C, lon, lat) → (N*T, C, lon, lat)
    if preds.dim() == 5:
        N, T, C, H, W = preds.shape
        preds = preds.reshape(N * T, C, H, W)
        tgts = tgts.reshape(N * T, C, H, W)

    return preds.numpy(), tgts.numpy()


def compute_weighting_comparison(create_config, epoch, variant_idx=0):
    """Load model, collect predictions, return comparison dict. No DB writes."""
    device_id = ddp_setup()

    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetConfig, ClimatesetData)
    mf = model_factory.get_factory()
    mf.register(ClimaXConfig, ClimaX)
    mf.register(UNetConfig, UNet)
    mf.register(CNNLSTMConfig, CNNLSTM_ClimateBench)
    mf.register(GRUTemporalWrapperConfig, GRUTemporalWrapper)

    train_run = create_config(ensemble_id=variant_idx)
    train_run.epochs = epoch

    train_ds = ClimatesetData(train_run.train_config.train_data_config)
    test_cfg = copy.deepcopy(train_run.train_config.train_data_config)
    test_cfg.scenarios = ["ssp245"]
    test_cfg.split = "test"
    test_ds = ClimatesetData(test_cfg)

    stats = load_training_stats_from_config(train_run.train_config.train_data_config)
    test_ds.set_normalization_stats(**stats)

    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False, drop_last=False)

    deser_model = deserialize_model(DeserializeConfig(train_run=train_run, device_id=device_id))
    if deser_model is None:
        print(f"  [skip] no checkpoint at epoch {epoch} for variant {variant_idx}")
        return None

    preds_np, tgts_np = _collect_predictions(deser_model.model, test_dl, device_id)
    lats = test_ds.lats  # 1-D array of real latitude values (degrees)

    output_vars = train_run.train_config.train_data_config.output_vars
    results = {}
    for c in range(preds_np.shape[1]):
        p, t = preds_np[:, c], tgts_np[:, c]
        var = output_vars[c]
        results.setdefault("correct",    {"per_channel": {}, "overall": None})
        results.setdefault("old_buggy",  {"per_channel": {}, "overall": None})
        results.setdefault("unweighted", {"per_channel": {}, "overall": None})
        results["correct"]["per_channel"][var]    = correct_lat_weighted_rmse(p, t, lats)
        results["old_buggy"]["per_channel"][var]  = old_buggy_lat_weighted_rmse(p, t)
        results["unweighted"]["per_channel"][var] = unweighted_rmse(p, t)

    for scheme in results:
        vals = list(results[scheme]["per_channel"].values())
        results[scheme]["overall"] = float(np.mean(vals))
    return results


def _pct(a, b):
    """Percentage difference of a relative to b: (a-b)/b * 100."""
    return (a - b) / b * 100 if b != 0 else float("nan")


def _print_comparison(model_name, seed, epoch, results):
    correct = results["correct"]["overall"]
    buggy   = results["old_buggy"]["overall"]
    unw     = results["unweighted"]["overall"]

    header = f"\n{'='*75}"
    print(header)
    print(f"  {model_name}  |  seed={seed}  |  epoch={epoch}")
    print(f"{'='*75}")
    print(f"  {'Scheme':<22}  {'Overall RMSE':>14}  {'vs correct':>12}  {'% diff vs correct':>18}")
    print(f"  {'-'*70}")
    rows = [
        ("correct (fixed)",    correct, True),
        ("old_buggy (pre-fix)", buggy,  False),
        ("unweighted",          unw,    False),
    ]
    for label, overall, is_ref in rows:
        pct_str = "---  (reference)" if is_ref else f"{_pct(overall, correct):+.2f}%"
        ratio_str = "---" if is_ref else f"{overall / correct:.4f}x"
        print(f"  {label:<22}  {overall:>14.6f}  {ratio_str:>12}  {pct_str:>18}")

    print(f"\n  Per-channel breakdown:")
    vars_ = list(results["correct"]["per_channel"].keys())
    col_w = max(len(v) for v in vars_)
    print(f"  {'Variable':<{col_w}}  {'correct':>14}  {'old_buggy':>14}  {'unweighted':>12}  {'bug % diff':>11}  {'unw % diff':>11}")
    print(f"  {'-'*90}")
    for var in vars_:
        r_cor = results["correct"]["per_channel"][var]
        r_bug = results["old_buggy"]["per_channel"][var]
        r_unw = results["unweighted"]["per_channel"][var]
        print(f"  {var:<{col_w}}  {r_cor:>14.6f}  {r_bug:>14.6f}  {r_unw:>12.6f}  {_pct(r_bug, r_cor):>+10.2f}%  {_pct(r_unw, r_cor):>+10.2f}%")
    print(header)


# ---------------------------------------------------------------------------
# Config grid / sweep logic (mirrors the multiseed script)
# ---------------------------------------------------------------------------

def _load_create_config(module_file_path):
    name = Path(module_file_path).stem
    spec = importlib.util.spec_from_file_location(name, module_file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.create_config


def _get_epochs(create_config):
    if "EPOCH" in os.environ:
        return [int(os.environ["EPOCH"])]
    try:
        c = create_config(0, climate_model_idx=0)
    except TypeError:
        c = create_config(0)
    step = int(os.environ.get("EVAL_EVERY", str(c.keep_nth_epoch_checkpoints)))
    return list(range(0, c.epochs + 1, step))


def create_configs():
    n_seeds = int(os.environ.get("N_SEEDS", "5"))
    n_variants = int(os.environ.get("NUM_VARIANTS", "15"))
    start = int(os.environ.get("CLIMATE_MODEL_IDX", "0"))
    create_config = _load_create_config(os.environ["CONFIG"])
    epochs = _get_epochs(create_config)
    print(
        f"Sweep: {n_variants} climate model(s) × {n_seeds} seed(s) = "
        f"{n_variants * n_seeds} job(s), each evaluating {len(epochs)} epoch(s)"
    )
    return get_config_grid(
        lambda **x: dict(**x),
        dict(
            seed=list(range(n_seeds)),
            climate_model_idx=list(range(start, start + n_variants)),
        ),
    )


def run(config):
    base_create_config = _load_create_config(os.environ["CONFIG"])
    climate_model_idx = config["climate_model_idx"]
    seed = config["seed"]
    epochs = _get_epochs(base_create_config)

    try:
        sample = base_create_config(ensemble_id=seed, climate_model_idx=climate_model_idx)
    except TypeError:
        sample = base_create_config(ensemble_id=seed)
    model_name = sample.train_config.train_data_config.climate_model

    curried = lambda ensemble_id, **kw: base_create_config(
        ensemble_id=ensemble_id, climate_model_idx=climate_model_idx, **kw
    )

    print(f"\n>>> {model_name} (idx={climate_model_idx})  seed={seed}  epochs={epochs}")

    for epoch in epochs:
        results = compute_weighting_comparison(curried, epoch, variant_idx=seed)
        if results is not None:
            _print_comparison(model_name, seed, epoch, results)

    sys.stdout.flush()


if __name__ == "__main__":
    configs = create_configs()
    for cfg_fn in configs:
        run(cfg_fn())
