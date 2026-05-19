#!/usr/bin/env python
"""
Plot per-channel validation curves across training checkpoints for one or more models.

Usage:
    uv run python experiments/weather/plot_training_curves.py \\
        config1.py config2.py ... \\
        [--metric rmse|acc] [--lead-time-days 1] \\
        [--epochs 10,20,...,200] [--labels ModelA,ModelB] \\
        [--upper] [--out-dir experiments/weather/plots/validation]

    # Compute RMSE and/or ACC directly from checkpoints (bypasses DB, uses GPU if available)
    uv run python experiments/weather/plot_training_curves.py \\
        config1.py config2.py ... \\
        --compute-rmse [--compute-acc] \\
        [--reduction-factor 0.2] [--consecutive-samples 10] \\
        [--device cuda|cpu]

Reads all data from the DuckDB remote database (via secrets.sql) by default.
With --compute-rmse, loads model checkpoints and runs inference locally.
Produces a grid of subplots — one per channel — with one line per model,
x-axis = epoch, y-axis = the chosen metric.
"""

import argparse
import importlib
from pathlib import Path

import torch
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.weather.data import (
    DataHP,
    DataHPSubset,
    DataHPSubsetConfig,
    Climatology,
)
from experiments.weather.metrics import (
    MeteorologicalData,
    rmse_hp,
    anomaly_correlation_coefficient_hp,
)
from lib.paths import get_checkpoint_path
from lib.serialization import deserialize_model, DeserializeConfig
from lib.stable_hash import stable_hash_str

era5_meta = MeteorologicalData()


# cache helpers

def _val_cache_path(cache_dir: str, config_hash: str, epoch: int,
                    lead_time_days: int, metric: str) -> Path:
    tag = f"{config_hash}_ep{epoch}_lt{lead_time_days}_{metric}"
    return Path(cache_dir) / f"{tag}.pkl"


def _load_cache(path: Path) -> dict | None:
    import pickle
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  Cache read failed ({path.name}): {e} — ignoring.")
    return None


def _save_cache(path: Path, data: dict) -> None:
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _config_hash(create_config, epoch: int) -> str:
    train_run = create_config(0, epoch)
    return stable_hash_str(train_run.train_config)[:16]


# helpers

def load_create_config(module_file_path: str):
    module_name = Path(module_file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    return config_file.create_config


def resolve_model_id_and_ds_len(create_config, epochs, db=None):
    """
    Return (model_id, ds_train_len) without loading model weights.

    Strategy (in order):
    1. Query the remote DB by config hash (works even without local checkpoints).
    2. Fall back to reading the model_id file from the local checkpoint directory.
    """
    train_run = create_config(0, epochs[0])

    # PEAR original config didn't include the param delta_t in the train_data_config,
    # so we need to handle that case for a correct retrieval of the model_id.
    if train_run.project == "weather":
        del train_run.train_config.train_data_config.delta_t


    ds_len = len(DataHP(train_run.train_config.train_data_config))

    # 1. DB lookup 
    if db is not None:
        from lib.stable_hash import stable_hash_str
        train_id = stable_hash_str(train_run.train_config)
        try:
            row = db.sql(f"""
                SELECT id FROM eqp.models
                WHERE train_id = '{train_id}'
                ORDER BY timestamp DESC
                LIMIT 1
            """).df()
            if not row.empty:
                model_id = int(row["id"].iloc[0])
                print(f"  model_id={model_id} (from DB, train_id={train_id})")
                return model_id, ds_len
            else:
                print(f"  train_id {train_id} not found in DB — trying local checkpoint.")
        except Exception as e:
            print(f"  DB model lookup failed ({e}) — trying local checkpoint.")

    # 2. Local checkpoint fallback 
    checkpoint_path = get_checkpoint_path(train_run.train_config)
    model_id_file = checkpoint_path / "model_id"
    if model_id_file.is_file():
        model_id = int(torch.load(model_id_file, map_location="cpu"))
        print(f"  model_id={model_id} (from local checkpoint)")
        return model_id, ds_len

    print(f"  model_id not found in DB or at {model_id_file}")
    return None, None


# DB 

def open_db():
    db = duckdb.connect()
    db.sql(open("secrets.sql").read())
    return db


def fetch_surface_metric(db, model_id: int, metric: str, lead_time_days: int) -> pd.DataFrame:
    """
    Fetch surface-channel rows from eqp.checkpoint_sample_metric_float.
    Name pattern: {metric}_surface_{channel}.{lead_time_days}d
    Returns: step | name | mean
    """
    return db.sql(f"""
        SELECT step, name, mean
        FROM eqp.checkpoint_sample_metric_float
        WHERE name LIKE '{metric}_surface_%.{lead_time_days}d'
          AND model_id = {model_id}
        ORDER BY step
    """).df()


def fetch_upper_metric(db, model_id: int, metric: str, lead_time_days: int) -> pd.DataFrame:
    """
    Fetch upper-level rows from eqp.checkpoint_sample_metric_float.
    Name pattern: {metric}_upper_{var}_{level}.{lead_time_days}d
    Returns: step | name | mean
    """
    return db.sql(f"""
        SELECT step, name, mean
        FROM eqp.checkpoint_sample_metric_float
        WHERE name LIKE '{metric}_upper_%.{lead_time_days}d'
          AND model_id = {model_id}
        ORDER BY step
    """).df()


def fetch_train_loss(db, model_id: int) -> pd.DataFrame:
    """
    Fetch training loss from eqp.train_step_metric_float.
    Returns: step | value  (downsampled to at most 1000 rows)
    """
    return db.sql(f"""
        SELECT step, value_float AS value
        FROM eqp.train_step_metric
        WHERE model_id = {model_id}
          AND name = 'loss'
          AND type = 'float'
        ORDER BY step
        USING SAMPLE 1000 ROWS
    """).df()


def to_tidy_surface(raw: pd.DataFrame, metric: str, lead_time_days: int, ds_len: int) -> pd.DataFrame:
    """
    Convert raw DB surface rows to:  epoch | channel | value
    """
    df = raw.copy()
    df["epoch"] = df["step"] / ds_len
    pattern = rf"{metric}_surface_([A-Za-z0-9_]+)\.{lead_time_days}d"
    df["channel"] = df["name"].str.extract(pattern)[0]
    df = df.dropna(subset=["channel"])
    return df[["epoch", "channel", "mean"]].rename(columns={"mean": "value"})


def to_tidy_upper(raw: pd.DataFrame, metric: str, lead_time_days: int, ds_len: int) -> pd.DataFrame:
    """
    Convert raw DB upper rows to:  epoch | channel | value
    channel is formatted as "{var}_{level}hPa".
    """
    df = raw.copy()
    df["epoch"] = df["step"] / ds_len
    pattern = rf"{metric}_upper_([A-Za-z0-9_]+)_([0-9]+)\.{lead_time_days}d"
    extracted = df["name"].str.extract(pattern)
    df["channel"] = extracted[0] + "_" + extracted[1] + "hPa"
    df = df.dropna(subset=["channel"])
    return df[["epoch", "channel", "mean"]].rename(columns={"mean": "value"})


# manual RMSE computation 

def _pick_device(requested: str) -> torch.device:
    """Resolve device string, falling back to CPU gracefully."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def compute_rmse_for_epoch(
    create_config,
    epoch: int,
    lead_time_days: int,
    device: torch.device,
    reduction_factor: float,
    consecutive_samples: int,
    is_pear: bool = False,
) -> dict:
    """
    Load checkpoint at *epoch*, run inference on the validation subset, and
    return a dict with keys 'surface' and 'upper', each a 1-D tensor of
    per-channel mean RMSE (already in physical units, averaged over pixels
    and samples).

    Delegates to rmse_hp() from metrics.py for consistency with the
    evaluate.py evaluation pipeline.

    Returns None if the checkpoint is missing.
    """
    train_run = create_config(0, epoch)

    data_cfg = train_run.train_config.train_data_config
    # DataHP24hConfig stores delta_t inside .base; DataHPConfig has it directly.
    _delta_t = getattr(data_cfg, "delta_t", None) or getattr(getattr(data_cfg, "base", None), "delta_t", None)
    if _delta_t == 24:
        reduction_factor = 1.0  # PEAR config was trained on the full dataset, so use all validation samples for a stable RMSE estimate
        consecutive_samples = 1  # PEAR config was trained with batch_size=1, so no need to group samples into blocks

    if is_pear:
        if hasattr(data_cfg, "delta_t"):
            del train_run.train_config.train_data_config.delta_t
        elif hasattr(getattr(data_cfg, "base", None), "delta_t"):
            del train_run.train_config.train_data_config.base.delta_t


    deser_config = DeserializeConfig(train_run=train_run, device_id=device)
    deser = deserialize_model(deser_config)
    if deser is None:
        print(f"  epoch={epoch}: checkpoint not found — skipping.")
        return None

    model = deser.model
    model.eval()

    val_data_config = (
        train_run.train_config.train_data_config
        .validation()
        .with_lead_time_days(lead_time_days)
    )
    # DataHP24hConfig wraps the real DataHPConfig in .base; DataHPSubset needs the inner config
    hp_val_config = getattr(val_data_config, "base", val_data_config)
    ds_val = DataHPSubset(
        DataHPSubsetConfig(
            data_config=hp_val_config,
            reduction_factor=reduction_factor,
            consecutive_samples=consecutive_samples,
        )
    )
    # ds_val = DataHP(val_data_config)
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=DataHPSubset.collate_fn,
    )

    result = rmse_hp(model, dl_val, device)
    return {
        "surface": result.mean_surface.cpu(),   # (C,)
        "upper":   result.mean_upper.cpu(),     # (C, L)
    }


def build_tidy_from_computed(
    rmse_by_epoch: dict,   # epoch -> {"surface": tensor(C,), "upper": tensor(C, L)}
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert the dict produced by compute_rmse_for_epoch calls into two tidy
    DataFrames:  (surface_df, upper_df)  each with columns epoch|channel|value.
    """
    surface_rows = []
    upper_rows   = []

    for epoch, result in sorted(rmse_by_epoch.items()):
        if result is None:
            continue
        for ch_idx, ch_name in enumerate(era5_meta.surface.names):
            surface_rows.append({
                "epoch":   epoch,
                "channel": ch_name,
                "value":   result["surface"][ch_idx].item(),
            })
        for var_idx, var_name in enumerate(era5_meta.upper.names):
            for lv_idx, level in enumerate(era5_meta.upper.levels):
                upper_rows.append({
                    "epoch":   epoch,
                    "channel": f"{var_name}_{int(level)}hPa",
                    "value":   result["upper"][var_idx, lv_idx].item(),
                })

    return pd.DataFrame(surface_rows), pd.DataFrame(upper_rows)


# manual ACC computation

def compute_acc_for_epoch(
    create_config,
    epoch: int,
    lead_time_days: int,
    device: torch.device,
    is_pear: bool = False,
) -> dict:
    """
    Load checkpoint at *epoch*, run ACC inference on the full validation Climatology,
    and return a dict with keys 'surface' and 'upper'.
    Returns None if the checkpoint is missing.
    """
    train_run = create_config(0, epoch)

    if is_pear:
        del train_run.train_config.train_data_config.delta_t

    deser_config = DeserializeConfig(train_run=train_run, device_id=device)
    deser = deserialize_model(deser_config)
    if deser is None:
        print(f"  epoch={epoch}: checkpoint not found — skipping.")
        return None

    model = deser.model
    model.eval()

    val_data_config = (
        train_run.train_config.train_data_config
        .validation()
        .with_lead_time_days(lead_time_days)
    )
    ds_acc = Climatology(val_data_config, use_wb2_clim=True)
    dl_acc = torch.utils.data.DataLoader(
        ds_acc,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=DataHPSubset.collate_fn,
    )

    result = anomaly_correlation_coefficient_hp(model, dl_acc, device)
    return {
        "surface": result.acc_surface.cpu(),  # (C,)
        "upper":   result.acc_upper.cpu(),    # (C, L)
    }


def build_tidy_from_computed_acc(
    acc_by_epoch: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert the dict produced by compute_acc_for_epoch calls into two tidy
    DataFrames: (surface_df, upper_df) each with columns epoch|channel|value.
    """
    surface_rows = []
    upper_rows   = []

    for epoch, result in sorted(acc_by_epoch.items()):
        if result is None:
            continue
        for ch_idx, ch_name in enumerate(era5_meta.surface.names):
            surface_rows.append({
                "epoch":   epoch,
                "channel": ch_name,
                "value":   result["surface"][ch_idx].item(),
            })
        for var_idx, var_name in enumerate(era5_meta.upper.names):
            for lv_idx, level in enumerate(era5_meta.upper.levels):
                upper_rows.append({
                    "epoch":   epoch,
                    "channel": f"{var_name}_{int(level)}hPa",
                    "value":   result["upper"][var_idx, lv_idx].item(),
                })

    return pd.DataFrame(surface_rows), pd.DataFrame(upper_rows)


# plotting

def _channel_order(channels: list[str], reference: list[str]) -> list[str]:
    """Sort channels so that known names appear first in reference order."""
    ordered = [c for c in reference if c in set(channels)]
    extra = sorted(c for c in channels if c not in set(reference))
    return ordered + extra

y_lim_for_channel_rmse_surface = {
    "msl": (20, 150),
    "u10": (0.35, 1.5),
    "v10": (0.35, 1.5),
    "t2m": (0, 2.5),
}


def plot_validation_grid(
    model_data: list[tuple[str, pd.DataFrame]],
    metric: str,
    lead_time_days: int,
    out_dir: str,
    tag: str = "",
):
    """
    model_data: list of (label, tidy_df) where tidy_df has columns epoch|channel|value.
    Produces a grid with one subplot per channel, one line per model.
    """
    all_channels: set[str] = set()
    for _, df in model_data:
        all_channels.update(df["channel"].unique())

    # Surface channels first (in ERA5 order), then upper-level channels
    surface_ref = era5_meta.surface.names
    upper_ref = [
        f"{v}_{int(lv)}hPa"
        for v in era5_meta.upper.names
        for lv in era5_meta.upper.levels
    ]
    channels = _channel_order(list(all_channels), surface_ref + upper_ref)

    n = len(channels)
    if n == 0:
        print("No channels to plot.")
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]

    for idx, ch in enumerate(channels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        for m_idx, (label, df) in enumerate(model_data):
            ch_df = df[df["channel"] == ch].sort_values("epoch")
            if ch_df.empty:
                continue
            color = colors[m_idx % len(colors)]
            ax.plot(ch_df["epoch"], ch_df["value"], marker="o", markersize=3,
                    linewidth=1.2, color=color, label=label)
        ax.set_title(ch, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(metric.upper(), fontsize=8)
        ax.legend(fontsize=7, ncol=1)
        ax.grid(True, linewidth=0.5)
        ax.tick_params(labelsize=7)
        # if metric == "rmse" and not 'upper' in out_dir:
        #     ax.set_ylim(y_lim_for_channel_rmse_surface[ch])


    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    title = f"{metric.upper()} — lead time {lead_time_days}d"
    if tag:
        title = f"{tag} — {title}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"validation_{metric}_{lead_time_days}d{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_train_loss(
    loss_data: list[tuple[str, pd.DataFrame]],
    ds_lens: list[int],
    out_dir: str,
    tag: str = "",
):
    """
    loss_data: list of (label, df) where df has columns step | value.
    Converts step → epoch using ds_lens, plots all models on one axis.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]

    for m_idx, ((label, df), ds_len) in enumerate(zip(loss_data, ds_lens)):
        if df.empty:
            continue
        color = colors[m_idx % len(colors)]
        epoch = df["step"] / ds_len
        ax.plot(epoch, df["value"], linewidth=0.8, color=color, alpha=0.8, label=label)

    ax.set_title("Training loss" + (f" — {tag}" if tag else ""))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"train_loss{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-channel validation curves for one or more trained models."
    )
    parser.add_argument(
        "configs", nargs="+",
        help="One or more experiment config .py files.",
    )
    parser.add_argument(
        "--metric", default="rmse", choices=["rmse", "acc"],
        help="Validation metric to plot (default: rmse).",
    )
    parser.add_argument("--lead-time-days", type=int, default=1)
    parser.add_argument(
        "--epochs", default=None,
        help="Comma-separated epochs used to probe for a valid checkpoint and resolve "
             "model_id, e.g. '10,20,...,200'. Default: 10,20,...,200.",
    )
    parser.add_argument(
        "--labels", default=None,
        help="Comma-separated display labels for each config. "
             "Default: stem of each config file.",
    )
    parser.add_argument(
        "--upper", action="store_true",
        help="Also plot upper-level pressure channels.",
    )
    parser.add_argument(
        "--train-loss", action="store_true",
        help="Also produce a training-loss plot.",
    )
    parser.add_argument(
        "--tag", default="",
        help="Optional string appended to output filenames and plot titles.",
    )
    parser.add_argument(
        "--out-dir", default="experiments/weather/plots/validation",
    )

    # manual RMSE/ACC computation
    parser.add_argument(
        "--compute-rmse", action="store_true",
        help="Compute RMSE directly from checkpoints (bypasses DB). "
             "Runs inference on the validation subset on the selected device.",
    )
    parser.add_argument(
        "--compute-acc", action="store_true",
        help="Compute ACC directly from checkpoints (bypasses DB). "
             "Runs inference on the full validation Climatology dataset. "
             "Can be combined with --compute-rmse.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device for --compute-rmse: 'cuda', 'cpu', or 'auto' (default). "
             "'auto' picks CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--reduction-factor", type=float, default=0.2,
        help="Fraction of validation samples to use when --compute-rmse is set "
             "(default: 0.2).",
    )
    parser.add_argument(
        "--consecutive-samples", type=int, default=10,
        help="Number of consecutive samples per block in the validation subset "
             "when --compute-rmse is set (default: 10).",
    )
    parser.add_argument(
        "--cache-dir", default=".local/validation_cache",
        help="Directory for local result cache (default: .local/validation_cache). "
             "Each (config, epoch, lead-time, metric) tuple gets its own .pkl file.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable the local file cache; always recompute from checkpoints.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    epochs = (
        [int(e.strip()) for e in args.epochs.split(",")]
        if args.epochs is not None
        else list(range(10, 201, 10))
    )

    n_configs = len(args.configs)
    if args.labels is not None:
        labels = [l.strip() for l in args.labels.split(",")]
        if len(labels) != n_configs:
            raise ValueError(
                f"--labels has {len(labels)} entries but {n_configs} configs were given."
            )
    else:
        labels = [Path(c).stem for c in args.configs]

    surface_data: list[tuple[str, pd.DataFrame]] = []
    upper_data:   list[tuple[str, pd.DataFrame]] = []
    acc_surface_data: list[tuple[str, pd.DataFrame]] = []
    acc_upper_data:   list[tuple[str, pd.DataFrame]] = []
    loss_data:    list[tuple[str, pd.DataFrame]] = []
    ds_lens:      list[int] = []

    # branch: compute RMSE/ACC manually from checkpoints
    if args.compute_rmse or args.compute_acc:
        device = _pick_device(args.device)
        print(f"[compute] Device: {device}")
        cache_dir = None if args.no_cache else args.cache_dir

        for config_path, label in zip(args.configs, labels):
            print(f"\n── {label} ({config_path}) ──")
            create_config = load_create_config(config_path)

            # Re-use the same delta_t stripping logic as resolve_model_id_and_ds_len:
            # all "weather" project configs had delta_t added later, so strip it.
            probe_run = create_config(0, epochs[0])
            is_pear = probe_run.project == "weather"

            cfg_hash = _config_hash(create_config, epochs[0])
            print(f"  config hash: {cfg_hash}")

            if args.compute_rmse:
                rmse_by_epoch = {}
                for epoch in epochs:
                    cached = None
                    if cache_dir is not None:
                        cp = _val_cache_path(cache_dir, cfg_hash, epoch, args.lead_time_days, "rmse")
                        cached = _load_cache(cp)
                        if cached is not None:
                            print(f"  [RMSE] epoch {epoch}: cache hit ({cp.name})")
                    if cached is not None:
                        rmse_by_epoch[epoch] = cached
                    else:
                        print(f"  [RMSE] epoch {epoch} …")
                        result = compute_rmse_for_epoch(
                            create_config=create_config,
                            epoch=epoch,
                            lead_time_days=args.lead_time_days,
                            device=device,
                            reduction_factor=args.reduction_factor,
                            consecutive_samples=args.consecutive_samples,
                            is_pear=is_pear,
                        )
                        rmse_by_epoch[epoch] = result
                        if result is not None and cache_dir is not None:
                            cp = _val_cache_path(cache_dir, cfg_hash, epoch, args.lead_time_days, "rmse")
                            _save_cache(cp, result)
                            print(f"  [RMSE] epoch {epoch}: cache write ({cp.name})")

                surf_df, upper_df = build_tidy_from_computed(rmse_by_epoch)
                if not surf_df.empty:
                    surface_data.append((label, surf_df))
                if args.upper and not upper_df.empty:
                    upper_data.append((label, upper_df))

            if args.compute_acc:
                acc_by_epoch = {}
                for epoch in epochs:
                    cached = None
                    if cache_dir is not None:
                        cp = _val_cache_path(cache_dir, cfg_hash, epoch, args.lead_time_days, "acc")
                        cached = _load_cache(cp)
                        if cached is not None:
                            print(f"  [ACC] epoch {epoch}: cache hit ({cp.name})")
                    if cached is not None:
                        acc_by_epoch[epoch] = cached
                    else:
                        print(f"  [ACC] epoch {epoch} …")
                        result = compute_acc_for_epoch(
                            create_config=create_config,
                            epoch=epoch,
                            lead_time_days=args.lead_time_days,
                            device=device,
                            is_pear=is_pear,
                        )
                        acc_by_epoch[epoch] = result
                        if result is not None and cache_dir is not None:
                            cp = _val_cache_path(cache_dir, cfg_hash, epoch, args.lead_time_days, "acc")
                            _save_cache(cp, result)
                            print(f"  [ACC] epoch {epoch}: cache write ({cp.name})")

                acc_surf_df, acc_upper_df = build_tidy_from_computed_acc(acc_by_epoch)
                if not acc_surf_df.empty:
                    acc_surface_data.append((label, acc_surf_df))
                if args.upper and not acc_upper_df.empty:
                    acc_upper_data.append((label, acc_upper_df))

    # branch: read from DuckDB 
    else:
        db = open_db()

        for config_path, label in zip(args.configs, labels):
            print(f"\n── {label} ({config_path}) ──")
            create_config = load_create_config(config_path)
            model_id, ds_len = resolve_model_id_and_ds_len(create_config, epochs, db=db)
            if model_id is None:
                print("  No checkpoint found — skipping.")
                continue
            print(f"  model_id={model_id}  ds_len={ds_len}")

            # Surface metric
            raw_surface = fetch_surface_metric(db, model_id, args.metric, args.lead_time_days)
            print(f"  Surface rows in DB: {len(raw_surface)}")
            if not raw_surface.empty:
                tidy = to_tidy_surface(raw_surface, args.metric, args.lead_time_days, ds_len)
                surface_data.append((label, tidy))

            # Upper metric (optional)
            if args.upper:
                raw_upper = fetch_upper_metric(db, model_id, args.metric, args.lead_time_days)
                print(f"  Upper rows in DB : {len(raw_upper)}")
                if not raw_upper.empty:
                    tidy_u = to_tidy_upper(raw_upper, args.metric, args.lead_time_days, ds_len)
                    upper_data.append((label, tidy_u))

            # Training loss (optional)
            if args.train_loss:
                loss_df = fetch_train_loss(db, model_id)
                print(f"  Train loss rows  : {len(loss_df)}")
                loss_data.append((label, loss_df))
                ds_lens.append(ds_len)

    if not surface_data and not upper_data and not acc_surface_data and not acc_upper_data:
        print("\nNo validation data found. Exiting.")
        raise SystemExit(1)

    # save raw data
    out_dir_path = Path(args.out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    if surface_data or upper_data:
        val_frames = []
        for label, df in surface_data:
            val_frames.append(df.assign(label=label, level="surface"))
        for label, df in upper_data:
            val_frames.append(df.assign(label=label, level="upper"))
        if val_frames:
            metric_tag = "rmse" if args.compute_rmse else args.metric
            val_csv = out_dir_path / f"validation_{metric_tag}_{args.lead_time_days}d{suffix}.csv"
            pd.concat(val_frames, ignore_index=True).to_csv(val_csv, index=False)
            print(f"Saved data: {val_csv}")

    if acc_surface_data or acc_upper_data:
        acc_frames = []
        for label, df in acc_surface_data:
            acc_frames.append(df.assign(label=label, level="surface"))
        for label, df in acc_upper_data:
            acc_frames.append(df.assign(label=label, level="upper"))
        if acc_frames:
            acc_csv = out_dir_path / f"validation_acc_{args.lead_time_days}d{suffix}.csv"
            pd.concat(acc_frames, ignore_index=True).to_csv(acc_csv, index=False)
            print(f"Saved data: {acc_csv}")

    if loss_data:
        loss_frames = []
        for (label, df), ds_len in zip(loss_data, ds_lens):
            loss_frames.append(df.assign(label=label, epoch=df["step"] / ds_len))
        loss_csv = out_dir_path / f"train_loss{suffix}.csv"
        pd.concat(loss_frames, ignore_index=True).to_csv(loss_csv, index=False)
        print(f"Saved data: {loss_csv}")

    plot_metric = "rmse" if args.compute_rmse else args.metric

    if surface_data:
        print("\nPlotting surface RMSE validation grid …")
        plot_validation_grid(
            surface_data, plot_metric, args.lead_time_days,
            args.out_dir, tag=args.tag,
        )

    if args.upper and upper_data:
        print("Plotting upper-level RMSE validation grid …")
        plot_validation_grid(
            upper_data, plot_metric, args.lead_time_days,
            args.out_dir, tag=("upper_" + args.tag).strip("_"),
        )

    if acc_surface_data:
        print("\nPlotting surface ACC validation grid …")
        plot_validation_grid(
            acc_surface_data, "acc", args.lead_time_days,
            args.out_dir, tag=("acc_" + args.tag).strip("_"),
        )

    if args.upper and acc_upper_data:
        print("Plotting upper-level ACC validation grid …")
        plot_validation_grid(
            acc_upper_data, "acc", args.lead_time_days,
            args.out_dir, tag=("acc_upper_" + args.tag).strip("_"),
        )

    if args.train_loss and loss_data:
        print("Plotting training loss …")
        plot_train_loss(loss_data, ds_lens, args.out_dir, tag=args.tag)

    print("\nDone.")
