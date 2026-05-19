#!/usr/bin/env python
"""
Evaluate equivariance error across training checkpoints for one or more configs.

Usage (single config):
    uv run python run.py experiments/weather/evaluate_equivariance.py \\
        <config.py> [--epochs 0,100,200] [--lead-time-days 1] [--max-batches 40]

Usage (multi-config comparison):
    uv run python run.py experiments/weather/evaluate_equivariance.py \\
        config_a.py config_b.py config_c.py \\
        [--labels ModelA,ModelB,ModelC] [--epochs 0,50,100,150,200]

For each requested epoch the script checks the DB first (via secrets.sql).
Epochs already present in the DB are read directly; the rest are submitted as
SLURM jobs and waited on.  Results from both sources are merged before plotting.

Output: 3 plots divided by training phase (early / mid / late).
Each plot has one subplot per surface channel; one line per config.
Plots are saved to experiments/weather/plots/<run_name>/.
"""

import argparse
import importlib
import os
from pathlib import Path
import sys

import duckdb
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import submitit

from experiments.weather.data import DataHP
from experiments.weather.metrics import MeteorologicalData, equivariance_error, EquivarianceError
from experiments.weather.models.hp_autoregressive import AutoregressiveWrapper, AutoregressiveWrapperConfig
from lib.ddp import ddp_setup
from lib.serialization import DeserializeConfig, deserialize_model
from lib.slurm import load_slurm_config_from_env

era5_meta = MeteorologicalData()


# ── helpers ───────────────────────────────────────────────────────────────────

def load_create_config(module_file_path: str):
    module_name = Path(module_file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    return config_file.create_config


def get_dl(create_config, epoch, lead_time_days):
    train_run = create_config(0, epoch)
    ds_config = (
        train_run.train_config.train_data_config
        .validation()
        .with_lead_time_days(lead_time_days)
    )
    # DataHP24hConfig wraps the real DataHPConfig in .base
    hp_config = getattr(ds_config, "base", ds_config)
    ds = DataHP(hp_config)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, drop_last=False, collate_fn=DataHP.collate_fn)


def get_model(create_config, epoch, device_id, model_name="model", return_model_id=False, autoregressive=False, ar_target_hours=24):
    train_run = create_config(0, epoch)
    if model_name == "pear":
        del train_run.train_config.train_data_config.delta_t
    deser_config = DeserializeConfig(train_run=train_run, device_id=device_id)
    deser_model = deserialize_model(deser_config)
    if deser_model is None:
        print(f"  Epoch {epoch}: no checkpoint found, skipping.")
        return (None, None) if return_model_id else None
    model = deser_model.model
    if autoregressive:
        delta_t = train_run.train_config.train_data_config.delta_t
        ar_config = AutoregressiveWrapperConfig(model_config=None, delta_t=delta_t, target_hours=ar_target_hours)
        model = AutoregressiveWrapper(ar_config, model)
        print(f"  Epoch {epoch}: wrapped autoregressively ({ar_target_hours}h / {delta_t}h = {ar_target_hours // delta_t} steps).")
    return (model, deser_model.model_id) if return_model_id else model


# ── single-epoch eval (runs on cluster worker) ────────────────────────────────

def evaluate_single_epoch(
    create_config, epoch, device_id,
    lead_time_days=1, max_batches=10, sensitivity=120, model_name="model",
    autoregressive=False, ar_target_hours=24,
):
    """Compute equivariance error for one epoch checkpoint. Runs on a SLURM worker.

    Returns a dict with keys:
      "surface": DataFrame, rows=channels, cols=angles + "<angle>_std" + "epoch"
      "upper":   DataFrame, rows=(channel, level) pairs, cols=angles + "<angle>_std"
                 + "epoch" + "channel_idx" + "level_idx"
    """
    dl = get_dl(create_config, epoch, lead_time_days)
    model = get_model(create_config, epoch, device_id, model_name=model_name, autoregressive=autoregressive, ar_target_hours=ar_target_hours)
    if model is None:
        return None
    print(f"Evaluating epoch {epoch} on cluster with Device {device_id}...")
    equiv_err: EquivarianceError = equivariance_error(
        model, dl, device_id, max_batches=max_batches, sensitivity=sensitivity
    )
    print(f"Epoch {epoch}: surface={equiv_err.surface}")

    # ── surface DataFrame ────────────────────────────────────────────────────
    df_surface = pd.DataFrame(equiv_err.surface)
    std_df = pd.DataFrame(
        {angle: vals for angle, vals in equiv_err.surface_std.items()}
    ).rename(columns=lambda c: f"{c}_std")
    df_surface = pd.concat([df_surface, std_df], axis=1)
    df_surface["epoch"] = epoch

    # ── upper DataFrame ──────────────────────────────────────────────────────
    angles = list(equiv_err.upper.keys())
    first_upper = next(iter(equiv_err.upper.values()))
    C, L = first_upper.shape

    rows = []
    for ch_idx in range(C):
        for lev_idx in range(L):
            row = {"channel_idx": ch_idx, "level_idx": lev_idx, "epoch": epoch}
            for angle in angles:
                row[angle] = float(equiv_err.upper[angle][ch_idx, lev_idx])
                row[f"{angle}_std"] = float(equiv_err.upper_std[angle][ch_idx, lev_idx])
            rows.append(row)
    df_upper = pd.DataFrame(rows)

    return {"surface": df_surface, "upper": df_upper}


# ── DB helpers ────────────────────────────────────────────────────────────────

def _open_db():
    db = duckdb.connect()
    db.sql(open("secrets.sql").read())
    return db


def _resolve_model_id_and_ds_len(create_config, epochs, device_id, model_name):
    """Return (model_id, ds_train_len) from the first available checkpoint."""
    for ep in sorted(epochs):
        model, model_id = get_model(
            create_config, ep, device_id, model_name=model_name, return_model_id=True
        )
        if model is None:
            continue
        train_run = create_config(0, ep)
        data_cfg = train_run.train_config.train_data_config
        # DataHP24hConfig wraps the real DataHPConfig in .base
        hp_cfg = getattr(data_cfg, "base", data_cfg)
        ds_len = len(DataHP(hp_cfg))
        return model_id, ds_len
    return None, None


def _db_rows_to_epoch_df(db_df, epoch, ds_len, lead_time_days):
    """
    Convert DB rows for one epoch into the same DataFrame format as
    evaluate_single_epoch: rows = channels, columns = rotation angles (float)
    + "epoch".  Returns None if there are no rows for this epoch.
    """
    step = epoch * ds_len
    ep_rows = db_df[db_df["step"] == step].copy()
    if ep_rows.empty:
        return None

    parsed = ep_rows["name"].str.extract(
        rf"equiv_error_surface_([A-Za-z0-9]+)_([0-9]+(?:\.[0-9]+)?)deg\.{lead_time_days}d"
    )
    ep_rows["channel"] = parsed[0]
    ep_rows["angle"] = parsed[1].astype(float)
    ep_rows = ep_rows.dropna(subset=["channel", "angle"])
    if ep_rows.empty:
        return None

    pivot = ep_rows.pivot_table(index="channel", columns="angle", values="mean", aggfunc="first")
    pivot.columns.name = None

    # Align row order to era5_meta.surface.names (best-effort)
    ordered = [c for c in era5_meta.surface.names if c in pivot.index]
    pivot = pivot.loc[ordered] if ordered else pivot
    pivot = pivot.reset_index(drop=True)
    pivot["epoch"] = epoch
    return {"surface": pivot, "upper": None}


# ── unified query: DB first, cluster for the rest ────────────────────────────

def _cache_path(cache_dir: str, model_id: int, epoch: int, lead_time_days: int,
                max_batches: int, sensitivity: int, autoregressive: bool,
                ar_target_hours: int) -> Path:
    tag = (f"model{model_id}_ep{epoch}_lt{lead_time_days}"
           f"_mb{max_batches}_sens{sensitivity}"
           f"_ar{int(autoregressive)}_arh{ar_target_hours}")
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


def query_equivariance(
    create_config, epochs, device_id,
    lead_time_days=1, max_batches=10, sensitivity=120,
    log_dir="submitit_logs", model_name="model", check_db=True,
    autoregressive=False, ar_target_hours=24,
    cache_dir: str | None = ".local/equiv_cache",
    run_local: bool = False,
):
    """
    For each epoch in `epochs`:
      - read from local file cache if available
      - read from DB if data is already present
      - otherwise compute locally (run_local=True) or submit a SLURM job and wait

    Results are written to the local cache after computation so subsequent
    runs skip the GPU entirely.  Pass cache_dir=None to disable caching.

    Returns a list of per-epoch dicts sorted by epoch.
    """
    model_id, ds_len = _resolve_model_id_and_ds_len(
        create_config, epochs, device_id, model_name
    )

    # ── file cache check ──────────────────────────────────────────────────────
    found_dfs: dict[int, dict] = {}
    missing_epochs: list[int] = []

    for ep in epochs:
        cached = None
        if cache_dir is not None and model_id is not None:
            cp = _cache_path(cache_dir, model_id, ep, lead_time_days,
                             max_batches, sensitivity, autoregressive, ar_target_hours)
            cached = _load_cache(cp)
            if cached is not None:
                print(f"  Cache hit  : epoch {ep} ({cp.name})")
        if cached is not None:
            found_dfs[ep] = cached
        else:
            missing_epochs.append(ep)

    # ── DB check for epochs still missing ────────────────────────────────────
    if check_db and missing_epochs:
        db_df = pd.DataFrame()
        if model_id is not None:
            try:
                db = _open_db()
                db_df = db.sql(f"""
                    SELECT step, name, mean
                    FROM eqp.checkpoint_sample_metric_float
                    WHERE name LIKE '%equiv_error_surface%'
                    AND name LIKE '%.{lead_time_days}d'
                    AND model_id = {model_id}
                """).df()
                print(f"DB: fetched {len(db_df)} rows for model_id={model_id}")
            except Exception as e:
                print(f"DB unavailable ({e}) — remaining epochs go to cluster.")

        db_steps = set(db_df["step"].unique()) if not db_df.empty else set()
        still_missing: list[int] = []
        for ep in missing_epochs:
            step = ep * ds_len if ds_len is not None else -1
            if step in db_steps:
                df = _db_rows_to_epoch_df(db_df, ep, ds_len, lead_time_days)
                if df is not None:
                    found_dfs[ep] = df
                    if cache_dir is not None and model_id is not None:
                        cp = _cache_path(cache_dir, model_id, ep, lead_time_days,
                                         max_batches, sensitivity, autoregressive, ar_target_hours)
                        _save_cache(cp, df)
                    continue
            still_missing.append(ep)
        missing_epochs = still_missing
    elif not check_db:
        pass  # missing_epochs unchanged, all go to cluster

    print(f"  From cache/DB: {sorted(found_dfs.keys())}")
    mode_label = "local" if run_local else "cluster"
    print(f"  On {mode_label}     : {missing_epochs}")

    # ── compute missing epochs locally or via SLURM ───────────────────────────
    cluster_dfs: dict[int, dict] = {}
    if missing_epochs:
        if run_local:
            for ep in missing_epochs:
                print(f"  Computing epoch {ep} locally...")
                df = evaluate_single_epoch(
                    create_config, int(ep), device_id,
                    lead_time_days, max_batches, sensitivity, model_name,
                    autoregressive, ar_target_hours,
                )
                if df is not None:
                    cluster_dfs[ep] = df
                    if cache_dir is not None and model_id is not None:
                        cp = _cache_path(cache_dir, model_id, ep, lead_time_days,
                                         max_batches, sensitivity, autoregressive, ar_target_hours)
                        _save_cache(cp, df)
                        print(f"  Cache write: epoch {ep} ({cp.name})")
        else:
            os.makedirs(log_dir, exist_ok=True)
            slurm_cfg = load_slurm_config_from_env()
            executor = submitit.AutoExecutor(folder=log_dir)
            executor.update_parameters(
                timeout_min=120,
                slurm_partition=slurm_cfg.partition,
                gpus_per_node=slurm_cfg.gpus,
                cpus_per_task=slurm_cfg.cpus_per_task,
                mem_gb=slurm_cfg.mem,
                slurm_job_name="equiv_eval",
            )
            jobs: dict[int, submitit.Job] = {}
            with executor.batch():
                for ep in missing_epochs:
                    jobs[ep] = executor.submit(
                        evaluate_single_epoch,
                        create_config, int(ep), device_id,
                        lead_time_days, max_batches, sensitivity, model_name,
                        autoregressive, ar_target_hours,
                    )
            for ep, job in jobs.items():
                df = job.result()
                if df is not None:
                    cluster_dfs[ep] = df
                    if cache_dir is not None and model_id is not None:
                        cp = _cache_path(cache_dir, model_id, ep, lead_time_days,
                                         max_batches, sensitivity, autoregressive, ar_target_hours)
                        _save_cache(cp, df)
                        print(f"  Cache write: epoch {ep} ({cp.name})")

    # ── merge and return sorted by epoch ──────────────────────────────────────
    all_dfs = {**found_dfs, **cluster_dfs}
    return [all_dfs[ep] for ep in sorted(all_dfs.keys())]


# ── plotting ──────────────────────────────────────────────────────────────────

_GROUP_NAMES = ["early_training", "mid_training", "late_training"]
_GROUP_TITLES = ["Early training", "Mid training", "Late training"]


def _angle_cols(df: pd.DataFrame) -> list:
    """Return sorted angle columns (floats), excluding epoch / _std / metadata cols."""
    skip = {"epoch", "channel_idx", "level_idx"}
    return sorted(
        [c for c in df.columns if c not in skip and not str(c).endswith("_std")],
        key=float,
    )


def _surface_df(epoch_entry) -> pd.DataFrame | None:
    """Extract the surface DataFrame from either the new dict format or legacy plain df."""
    if isinstance(epoch_entry, dict):
        return epoch_entry.get("surface")
    return epoch_entry  # legacy plain DataFrame


def _upper_df(epoch_entry) -> pd.DataFrame | None:
    if isinstance(epoch_entry, dict):
        return epoch_entry.get("upper")
    return None


def _collect_channel_vals(epoch_dfs, epoch_set, ch_idx, acols, angles, surface=True):
    """
    For each epoch in epoch_set, extract the (mean, per-sample-std) arrays for
    channel ch_idx.  Returns (values_per_epoch, sample_stds_per_epoch).
    """
    values_per_epoch: list[np.ndarray] = []
    sample_stds_per_epoch: list[np.ndarray] = []

    for entry in epoch_dfs:
        df = _surface_df(entry) if surface else None
        if df is None:
            continue
        ep = int(df["epoch"].iloc[0])
        if ep not in epoch_set or ch_idx >= len(df):
            continue
        row = df.iloc[ch_idx]
        vals = np.array([row[c] for c in acols if c in row.index], dtype=float)
        stds = np.array(
            [row.get(f"{c}_std", np.nan) for c in acols], dtype=float
        )
        if len(vals) == len(angles):
            values_per_epoch.append(vals)
            sample_stds_per_epoch.append(stds)

    return values_per_epoch, sample_stds_per_epoch


def _draw_line_with_bands(ax, angles, values_per_epoch, sample_stds_per_epoch, color, ls, mk, label):
    """Plot mean line + per-sample-std inner band + across-epoch-std outer band."""
    if not values_per_epoch:
        return
    mean_v = np.mean(values_per_epoch, axis=0)
    ax.plot(
        angles, mean_v,
        linestyle=ls, marker=mk, markersize=4,
        color=color, linewidth=1.5, label=label,
    )
    # Inner band: mean per-sample measurement std (always shown when available).
    if sample_stds_per_epoch:
        mean_sample_std = np.nanmean(sample_stds_per_epoch, axis=0)
        if not np.all(np.isnan(mean_sample_std)):
            ax.fill_between(
                angles,
                mean_v - mean_sample_std, mean_v + mean_sample_std,
                alpha=0.20, color=color, linewidth=0,
            )
    # Outer band: across-epoch variability (only when multiple epochs in group).
    if len(values_per_epoch) > 1:
        epoch_std = np.std(values_per_epoch, axis=0)
        ax.fill_between(
            angles,
            mean_v - epoch_std, mean_v + epoch_std,
            alpha=0.10, color=color, linewidth=0,
        )


def _epoch_groups(config_results, n_groups):
    """Return (all_epochs, epoch_groups) split into n_groups."""
    all_epochs: list[int] = sorted({
        int(_surface_df(entry)["epoch"].iloc[0])
        for dfs in config_results.values()
        for entry in dfs
        if _surface_df(entry) is not None
    })
    return all_epochs, np.array_split(all_epochs, n_groups)


def plot_multi_config_by_epoch_group(
    config_results: dict[str, list],
    n_groups: int = 3,
    run_name: str = "comparison",
    lead_time_days: int = 1,
) -> None:
    """
    Produce n_groups figures, one per training phase (early / mid / late).

    Each figure has one subplot per surface channel.  Within each subplot,
    one line per config is drawn — the line is the per-epoch mean over all
    epochs that fall in that training phase.

    Shading shows two bands:
      - inner (darker): mean ±1 per-sample measurement std (from equivariance_error)
      - outer (lighter): ±1 std across epochs in the group (when >1 epoch present)

    Args:
        config_results: mapping from label → list of per-epoch dicts
                        (each dict has keys "surface" and "upper").
        n_groups:       how many training-phase groups to split epochs into.
        run_name:       sub-directory under experiments/weather/plots/.
        lead_time_days: used only in the figure super-title.
    """
    out_dir = Path(f"experiments/weather/plots/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_epochs, epoch_groups = _epoch_groups(config_results, n_groups)
    if not all_epochs:
        print("No data to plot.")
        return

    # Infer geometry from first available surface df.
    first_surface = next(
        (_surface_df(e) for dfs in config_results.values() for e in dfs if _surface_df(e) is not None),
        None,
    )
    if first_surface is None:
        return
    acols = _angle_cols(first_surface)
    angles = [float(c) for c in acols]
    num_channels = len(first_surface.drop(columns=["epoch"], errors="ignore").pipe(
        lambda df: df[[c for c in df.columns if not str(c).endswith("_std")]]
    ))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]

    for g_idx, group_epochs in enumerate(epoch_groups):
        if len(group_epochs) == 0:
            continue
        epoch_set = set(int(e) for e in group_epochs)
        label_str = f"epochs {int(group_epochs[0])}–{int(group_epochs[-1])}"

        ncols = 2
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
        for ch_idx in range(num_channels):
            row, col = divmod(ch_idx, ncols)
            ax = axes[row][col]
            ch_name = (
                era5_meta.surface.names[ch_idx]
                if ch_idx < len(era5_meta.surface.names)
                else str(ch_idx)
            )
            ax.set_title(ch_name, fontsize=20, pad=4)

            for c_idx, (label, epoch_dfs) in enumerate(config_results.items()):
                color = colors[c_idx % len(colors)]
                ls = linestyles[c_idx % len(linestyles)]
                mk = markers[c_idx % len(markers)]

                vals, stds = _collect_channel_vals(
                    epoch_dfs, epoch_set, ch_idx, acols, angles, surface=True
                )
                _draw_line_with_bands(ax, angles, vals, stds, color, ls, mk, label)

            if row == nrows - 1:
                ax.set_xlabel("Rotation angle (°)", fontsize=20)
            if col == 0:
                ax.set_ylabel("Equivariance error", fontsize=20)
            ax.tick_params(labelsize=18)
            ax.grid(True, linewidth=0.5, alpha=0.6)

        for ch_idx in range(num_channels, nrows * ncols):
            row, col = divmod(ch_idx, ncols)
            axes[row][col].set_visible(False)

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                       bbox_to_anchor=(0.5, 0.0), fontsize=18, framealpha=0.8)
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        group_tag = _GROUP_NAMES[g_idx % len(_GROUP_NAMES)]
        out_path = out_dir / f"equivariance_surface_{group_tag}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")

    print(f"Surface plots saved to {out_dir}/")


def plot_upper_by_epoch_group(
    config_results: dict[str, list],
    n_groups: int = 3,
    run_name: str = "comparison",
    lead_time_days: int = 1,
) -> None:
    """
    Produce n_groups figures (one per training phase) for upper-level channels.

    Each figure has one subplot per (variable, pressure-level) pair, e.g.
    z_500hPa, q_850hPa — matching the layout of plot_training_curves.py.
    Shading follows the same two-band convention as the surface plots.

    Only epochs with upper data (cluster-computed, not DB-sourced) are used.
    """
    out_dir = Path(f"experiments/weather/plots/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    upper_epochs: list[int] = sorted({
        int(entry["upper"]["epoch"].iloc[0])
        for dfs in config_results.values()
        for entry in dfs
        if isinstance(entry, dict) and entry.get("upper") is not None
    })
    if not upper_epochs:
        print("No upper-level data available — skipping upper plots.")
        return

    epoch_groups = np.array_split(upper_epochs, n_groups)

    first_upper = next(
        (entry["upper"] for dfs in config_results.values()
         for entry in dfs
         if isinstance(entry, dict) and entry.get("upper") is not None),
        None,
    )
    if first_upper is None:
        return
    acols = _angle_cols(first_upper)
    angles = [float(c) for c in acols]
    n_channels = first_upper["channel_idx"].nunique()
    n_levels = first_upper["level_idx"].nunique()

    # Columns: variables in user-specified order [q, t, u, v, z].
    var_col_order = ["q", "t", "u", "v", "z"]
    all_names = era5_meta.upper.names  # ["z", "q", "t", "u", "v"]
    col_ch_indices = [all_names.index(v) for v in var_col_order if v in all_names]

    # Rows: pressure levels high→low (levels list is already 1000→... descending).
    row_lv_indices = list(range(n_levels))

    # Build grid: (lv_idx, ch_idx, title) row-major so axes[row][col] is natural.
    subplots: list[tuple[int, int, str]] = []
    for lv_idx in row_lv_indices:
        lv_label = (
            f"{int(era5_meta.upper.levels[lv_idx])}hPa"
            if lv_idx < len(era5_meta.upper.levels)
            else str(lv_idx)
        )
        for ch_idx in col_ch_indices:
            ch_name = all_names[ch_idx] if ch_idx < len(all_names) else str(ch_idx)
            subplots.append((ch_idx, lv_idx, f"{ch_name} {lv_label}"))

    ncols = len(col_ch_indices)
    nrows_grid = len(row_lv_indices)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]

    for g_idx, group_epochs in enumerate(epoch_groups):
        if len(group_epochs) == 0:
            continue
        epoch_set = set(int(e) for e in group_epochs)
        label_str = f"epochs {int(group_epochs[0])}–{int(group_epochs[-1])}"

        fig, axes = plt.subplots(
            nrows_grid, ncols,
            figsize=(4 * ncols, 3.5 * nrows_grid),
            sharex=True, sharey=True,
            squeeze=False,
        )

        for sp_idx, (ch_idx, lv_idx, title) in enumerate(subplots):
            row, col = divmod(sp_idx, ncols)
            ax = axes[row][col]
            ax.set_title(title, fontsize=22, pad=4)

            for c_idx, (label, epoch_dfs) in enumerate(config_results.items()):
                color = colors[c_idx % len(colors)]
                ls = linestyles[c_idx % len(linestyles)]
                mk = markers[c_idx % len(markers)]

                values_per_epoch: list[np.ndarray] = []
                sample_stds_per_epoch: list[np.ndarray] = []

                for entry in epoch_dfs:
                    udf = _upper_df(entry)
                    if udf is None:
                        continue
                    ep = int(udf["epoch"].iloc[0])
                    if ep not in epoch_set:
                        continue
                    row_mask = (udf["channel_idx"] == ch_idx) & (udf["level_idx"] == lv_idx)
                    rows = udf[row_mask]
                    if rows.empty:
                        continue
                    vals = np.array([rows[c].iloc[0] for c in acols], dtype=float)
                    stds = np.array(
                        [rows[f"{c}_std"].iloc[0] for c in acols if f"{c}_std" in rows.columns],
                        dtype=float,
                    )
                    if len(vals) == len(angles):
                        values_per_epoch.append(vals)
                        if len(stds) == len(angles):
                            sample_stds_per_epoch.append(stds)

                _draw_line_with_bands(ax, angles, values_per_epoch, sample_stds_per_epoch, color, ls, mk, label)

            if row == nrows_grid - 1:
                ax.set_xlabel("Rotation angle (°)", fontsize=22)
            if col == 0:
                ax.set_ylabel("Equivariance error", fontsize=22)
            ax.tick_params(labelsize=20)
            ax.grid(True, linewidth=0.5, alpha=0.6)
            ax.set_ylim(0, 0.4)

        for sp_idx in range(len(subplots), nrows_grid * ncols):
            row, col = divmod(sp_idx, ncols)
            axes[row][col].set_visible(False)

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                       bbox_to_anchor=(0.5, 1.0), fontsize=20, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        group_tag = _GROUP_NAMES[g_idx % len(_GROUP_NAMES)]
        out_path = out_dir / f"equivariance_upper_{group_tag}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")

    print(f"Upper plots saved to {out_dir}/")


def plot_upper_max_vs_level(
    config_results: dict[str, list],
    run_name: str = "comparison",
    lead_time_days: int = 1,
    untrained_results: list | None = None,
) -> None:
    """
    Produce a single figure with one subplot per upper-level variable.
    Each subplot shows max(mean EE over rotation angles) vs. pressure level,
    averaged over ALL available epochs.

    Both configs use the same color; linestyle and marker distinguish them.
    """
    import matplotlib.lines as mlines

    out_dir = Path(f"experiments/weather/plots/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    upper_epochs: list[int] = sorted({
        int(entry["upper"]["epoch"].iloc[0])
        for dfs in config_results.values()
        for entry in dfs
        if isinstance(entry, dict) and entry.get("upper") is not None
    })
    if not upper_epochs:
        print("No upper-level data available — skipping max-vs-level plots.")
        return

    first_upper = next(
        (entry["upper"] for dfs in config_results.values()
         for entry in dfs
         if isinstance(entry, dict) and entry.get("upper") is not None),
        None,
    )
    if first_upper is None:
        return
    acols = _angle_cols(first_upper)
    n_levels = first_upper["level_idx"].nunique()

    var_col_order = ["q", "t", "u", "v", "z"]
    all_names = era5_meta.upper.names
    col_ch_indices = [all_names.index(v) for v in var_col_order if v in all_names]
    col_ch_names = [all_names[i] for i in col_ch_indices]

    pressure_levels = [
        int(era5_meta.upper.levels[lv]) if lv < len(era5_meta.upper.levels) else lv
        for lv in range(n_levels)
    ]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    config_linestyles = {label: "-" for label in config_results}
    config_markers = ["o", "s", "^", "D", "v"]

    epoch_set = set(upper_epochs)

    # ── figure: 2 rows (3 + 2) ───────────────────────────────────────────────
    n_cols_grid = 3
    fig, axes = plt.subplots(2, n_cols_grid, figsize=(5 * n_cols_grid, 10), squeeze=False)

    for col_pos, (ch_idx, ch_name) in enumerate(zip(col_ch_indices, col_ch_names)):
        grid_row, grid_col = divmod(col_pos, n_cols_grid)
        ax = axes[grid_row][grid_col]
        ax.set_title(ch_name, fontsize=24, pad=4)

        for c_idx, (label, epoch_dfs) in enumerate(config_results.items()):
            ls = config_linestyles[label]
            mk = config_markers[c_idx % len(config_markers)]
            color = colors[c_idx % len(colors)]
            max_per_level: list[float] = []
            for lv_idx in range(n_levels):
                values_per_epoch: list[np.ndarray] = []
                for entry in epoch_dfs:
                    udf = _upper_df(entry)
                    if udf is None:
                        continue
                    if int(udf["epoch"].iloc[0]) not in epoch_set:
                        continue
                    row_mask = (udf["channel_idx"] == ch_idx) & (udf["level_idx"] == lv_idx)
                    rows = udf[row_mask]
                    if rows.empty:
                        continue
                    vals = np.array([rows[c].iloc[0] for c in acols], dtype=float)
                    if len(vals) == len(acols):
                        values_per_epoch.append(vals)

                if values_per_epoch:
                    mean_v = np.mean(values_per_epoch, axis=0)
                    max_per_level.append(float(np.max(mean_v)))
                else:
                    max_per_level.append(np.nan)

            ax.plot(
                pressure_levels, max_per_level,
                linestyle=ls, marker=mk, markersize=5,
                color=color, linewidth=1.5, label=label,
            )

        ax.set_xlabel("Pressure level (hPa)", fontsize=20)
        if grid_col == 0:
            ax.set_ylabel("Max mean equivariance error", fontsize=18)
        ax.tick_params(labelsize=18)
        ax.grid(True, linewidth=0.5, alpha=0.6)
        ax.invert_xaxis()

    # Hide unused bottom-right cell.
    n_total = len(col_ch_indices)
    for spare in range(n_total, 2 * n_cols_grid):
        r, c = divmod(spare, n_cols_grid)
        axes[r][c].set_visible(False)

    # ── legend: one entry per config ──────────────────────────────────────────
    config_handles = [
        mlines.Line2D([], [], color=colors[i % len(colors)], linestyle="-",
                      marker=config_markers[i % len(config_markers)], markersize=5,
                      linewidth=2, label=label)
        for i, label in enumerate(config_results)
    ]
    fig.legend(
        handles=config_handles,
        loc="lower center", ncol=len(config_handles),
        bbox_to_anchor=(0.5, 0.0), fontsize=14, framealpha=0.8,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])

    out_path = out_dir / "equivariance_upper_max_vs_level.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")
    print(f"Max-vs-level plot saved to {out_dir}/")


def plot_upper_max_vs_level_collapsed(
    config_results: dict[str, list],
    run_name: str = "comparison",
    untrained_results: list | None = None,
) -> None:
    """
    Single-axis version of plot_upper_max_vs_level: all variables and configs
    are drawn on one plot.  Color = variable, linestyle = config.
    """
    import matplotlib.lines as mlines

    out_dir = Path(f"experiments/weather/plots/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    upper_epochs: list[int] = sorted({
        int(entry["upper"]["epoch"].iloc[0])
        for dfs in config_results.values()
        for entry in dfs
        if isinstance(entry, dict) and entry.get("upper") is not None
    })
    if not upper_epochs:
        print("No upper-level data available — skipping collapsed max-vs-level plot.")
        return

    first_upper = next(
        (entry["upper"] for dfs in config_results.values()
         for entry in dfs
         if isinstance(entry, dict) and entry.get("upper") is not None),
        None,
    )
    if first_upper is None:
        return
    acols = _angle_cols(first_upper)
    n_levels = first_upper["level_idx"].nunique()

    var_col_order = ["q", "t", "u", "v", "z"]
    all_names = era5_meta.upper.names
    col_ch_indices = [all_names.index(v) for v in var_col_order if v in all_names]
    col_ch_names = [all_names[i] for i in col_ch_indices]

    pressure_levels = [
        int(era5_meta.upper.levels[lv]) if lv < len(era5_meta.upper.levels) else lv
        for lv in range(n_levels)
    ]

    cmap = plt.get_cmap("tab10")
    var_colors = {ch_name: cmap(i) for i, ch_name in enumerate(col_ch_names)}
    config_linestyles = {label: ("-" if i == 0 else ":") for i, label in enumerate(config_results)}
    var_markers = ["o", "s", "^", "D", "v"]
    epoch_set = set(upper_epochs)

    # ── reference line ────────────────────────────────────────────────────────
    ref_line_value: float | None = None
    if untrained_results is not None:
        ref_vals: list[float] = []
        for entry in untrained_results:
            udf = _upper_df(entry)
            if udf is None:
                continue
            for ch_idx in col_ch_indices:
                for lv_idx in range(n_levels):
                    row_mask = (udf["channel_idx"] == ch_idx) & (udf["level_idx"] == lv_idx)
                    rows = udf[row_mask]
                    if rows.empty:
                        continue
                    vals = np.array([rows[c].iloc[0] for c in acols], dtype=float)
                    if len(vals) == len(acols):
                        ref_vals.append(float(np.max(vals)))
        if ref_vals:
            ref_line_value = float(np.mean(ref_vals))

    # ── single axis ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    for col_pos, (ch_idx, ch_name) in enumerate(zip(col_ch_indices, col_ch_names)):
        color = var_colors[ch_name]
        mk = var_markers[col_pos % len(var_markers)]

        for label, epoch_dfs in config_results.items():
            ls = config_linestyles[label]
            max_per_level: list[float] = []
            for lv_idx in range(n_levels):
                values_per_epoch: list[np.ndarray] = []
                for entry in epoch_dfs:
                    udf = _upper_df(entry)
                    if udf is None:
                        continue
                    if int(udf["epoch"].iloc[0]) not in epoch_set:
                        continue
                    row_mask = (udf["channel_idx"] == ch_idx) & (udf["level_idx"] == lv_idx)
                    rows = udf[row_mask]
                    if rows.empty:
                        continue
                    vals = np.array([rows[c].iloc[0] for c in acols], dtype=float)
                    if len(vals) == len(acols):
                        values_per_epoch.append(vals)

                if values_per_epoch:
                    mean_v = np.mean(values_per_epoch, axis=0)
                    max_per_level.append(float(np.max(mean_v)))
                else:
                    max_per_level.append(np.nan)

            ax.plot(pressure_levels, max_per_level,
                    linestyle=ls, marker=mk, markersize=5,
                    color=color, linewidth=1.5)

    if ref_line_value is not None:
        ax.axhline(ref_line_value, color="gray", linestyle="--", linewidth=1.2)

    ax.set_xlabel("Pressure level (hPa)", fontsize=20)
    ax.set_ylabel("Max mean equivariance error", fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.invert_xaxis()

    # ── two-section legend ────────────────────────────────────────────────────
    var_handles = [
        mlines.Line2D([], [], color=var_colors[ch_name], linewidth=2,
                      marker=var_markers[i % len(var_markers)], markersize=5, label=ch_name)
        for i, ch_name in enumerate(col_ch_names)
    ]
    config_handles = [
        mlines.Line2D([], [], color="black", linestyle=ls, linewidth=2, label=label)
        for label, ls in config_linestyles.items()
    ]
    if ref_line_value is not None:
        config_handles.append(
            mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5,
                          label="Untrained PEAR (reference)")
        )
    ax.legend(handles=var_handles + config_handles, fontsize=18, framealpha=0.8,
              loc="best", ncol=2)

    fig.tight_layout()
    out_path = out_dir / "equivariance_upper_max_vs_level_collapsed.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_upper_max_vs_level_avg(
    config_results: dict[str, list],
    run_name: str = "comparison",
    untrained_results: list | None = None,
) -> None:
    """
    Single-axis plot: for each config, one line showing the max EE per pressure
    level averaged over all variables.  Color/linestyle encode config.
    Reference line shows the same average for the untrained model.
    """
    import matplotlib.lines as mlines

    out_dir = Path(f"experiments/weather/plots/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    upper_epochs: list[int] = sorted({
        int(entry["upper"]["epoch"].iloc[0])
        for dfs in config_results.values()
        for entry in dfs
        if isinstance(entry, dict) and entry.get("upper") is not None
    })
    if not upper_epochs:
        print("No upper-level data available — skipping avg-over-vars plot.")
        return

    first_upper = next(
        (entry["upper"] for dfs in config_results.values()
         for entry in dfs
         if isinstance(entry, dict) and entry.get("upper") is not None),
        None,
    )
    if first_upper is None:
        return
    acols = _angle_cols(first_upper)
    n_levels = first_upper["level_idx"].nunique()

    var_col_order = ["q", "t", "u", "v", "z"]
    all_names = era5_meta.upper.names
    col_ch_indices = [all_names.index(v) for v in var_col_order if v in all_names]

    pressure_levels = [
        int(era5_meta.upper.levels[lv]) if lv < len(era5_meta.upper.levels) else lv
        for lv in range(n_levels)
    ]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    config_linestyles = {label: ("-" if i == 0 else ":") for i, label in enumerate(config_results)}
    markers = ["o", "s", "^", "D", "v"]
    epoch_set = set(upper_epochs)

    def _avg_max_per_level(epoch_dfs) -> list[float]:
        """For each level, average max_angle(EE) over all variables and epochs."""
        result = []
        for lv_idx in range(n_levels):
            level_vals: list[float] = []
            for ch_idx in col_ch_indices:
                values_per_epoch: list[np.ndarray] = []
                for entry in epoch_dfs:
                    udf = _upper_df(entry)
                    if udf is None:
                        continue
                    if int(udf["epoch"].iloc[0]) not in epoch_set:
                        continue
                    row_mask = (udf["channel_idx"] == ch_idx) & (udf["level_idx"] == lv_idx)
                    rows = udf[row_mask]
                    if rows.empty:
                        continue
                    vals = np.array([rows[c].iloc[0] for c in acols], dtype=float)
                    if len(vals) == len(acols):
                        values_per_epoch.append(vals)
                if values_per_epoch:
                    mean_v = np.mean(values_per_epoch, axis=0)
                    level_vals.append(float(np.max(mean_v)))
            result.append(float(np.mean(level_vals)) if level_vals else np.nan)
        return result

    # ── reference line ────────────────────────────────────────────────────────
    ref_line_value: float | None = None
    if untrained_results is not None:
        ref_vals: list[float] = []
        for entry in untrained_results:
            udf = _upper_df(entry)
            if udf is None:
                continue
            for ch_idx in col_ch_indices:
                for lv_idx in range(n_levels):
                    row_mask = (udf["channel_idx"] == ch_idx) & (udf["level_idx"] == lv_idx)
                    rows = udf[row_mask]
                    if rows.empty:
                        continue
                    vals = np.array([rows[c].iloc[0] for c in acols], dtype=float)
                    if len(vals) == len(acols):
                        ref_vals.append(float(np.max(vals)))
        if ref_vals:
            ref_line_value = float(np.mean(ref_vals))

    # ── single axis ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    # fig.suptitle("Max equivariance error vs pressure level (avg over variables)",
    #              fontsize=10, fontweight="bold")

    for c_idx, (label, epoch_dfs) in enumerate(config_results.items()):
        avg_per_level = _avg_max_per_level(epoch_dfs)
        ax.plot(pressure_levels, avg_per_level,
                linestyle=config_linestyles[label],
                marker=markers[c_idx % len(markers)], markersize=6,
                color=colors[c_idx % len(colors)], linewidth=2,
                label=label)

    if ref_line_value is not None:
        ax.axhline(ref_line_value, color="gray", linestyle="--", linewidth=1.5,
                   label="Untrained PEAR (reference)")

    ax.set_xlabel("Pressure level (hPa)", fontsize=19)
    ax.set_ylabel("Mean max equivariance error", fontsize=19)
    ax.tick_params(labelsize=13)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.invert_xaxis()
    ax.legend(fontsize=15, framealpha=0.8, loc="best")

    fig.tight_layout()
    out_path = out_dir / "equivariance_upper_max_vs_level_avg.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate equivariance error across training checkpoints for one or "
            "more configs.  When multiple configs are given, produces 3 comparison "
            "plots split by training phase (early / mid / late)."
        )
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="One or more paths to experiment config .py files.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated display labels for each config, e.g. 'PEAR,Equiv'. "
             "Defaults to the config file stem.",
    )
    parser.add_argument(
        "--epochs",
        default=None,
        help="Comma-separated epochs to evaluate, e.g. '0,100,200'. "
             "Default: 10,20,...,200.",
    )
    parser.add_argument("--lead-time-days", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument(
        "--sensitivity", type=int, default=120,
        help="Number of rotation steps (sensitivity=120 → one angle every 3°).",
    )
    parser.add_argument("--log-dir", default="submitit_logs")
    parser.add_argument(
        "--check-db", action="store_true",
        help="Check DB and print available epochs without submitting cluster jobs.",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Output sub-directory name under experiments/weather/plots/. "
             "Defaults to the stem of the first config.",
    )
    parser.add_argument(
        "--autoregressive", action="store_true",
        help="Wrap each model in AutoregressiveWrapper before evaluating equivariance.",
    )
    parser.add_argument(
        "--ar-target-hours", type=int, default=24,
        help="Total forecast horizon for autoregressive chaining (default: 24).",
    )
    parser.add_argument(
        "--run-local", action="store_true",
        help="Compute missing epochs in the current process instead of submitting SLURM jobs. "
             "Useful when already running inside a SLURM allocation.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to use for local computation, e.g. 'cuda', 'cuda:0', 'cpu'. "
             "Overrides ddp_setup(). Defaults to 'cuda' when --run-local is set and "
             "a GPU is available, otherwise falls back to ddp_setup().",
    )
    parser.add_argument(
        "--cache-dir", default=".local/equiv_cache",
        help="Directory for local result cache (default: .local/equiv_cache). "
             "Each (model, epoch, eval-params) tuple gets its own .pkl file.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable the local file cache; always recompute on the cluster.",
    )
    parser.add_argument(
        "--untrained-config", default=None,
        help="Path to a config .py for the untrained PEAR baseline. Used to draw "
             "a horizontal reference line on the max-vs-level plot.",
    )
    parser.add_argument(
        "--untrained-epoch", type=int, default=0,
        help="Epoch to load for the untrained reference model (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.device is not None:
        device_id = args.device
    elif args.run_local and torch.cuda.is_available():
        device_id = "cuda"
    else:
        device_id = ddp_setup()

    def oom_observer(device, alloc, device_alloc, device_free):
        print("saving allocated state during OOM")
        torch.cuda.memory._dump_snapshot("oom_snapshot_new.pickle")
    torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    epochs = (
        [int(e.strip()) for e in args.epochs.split(",")]
        if args.epochs is not None
        else list(range(10, 201, 10))
    )

    # Build label list: explicit --labels, else config stems.
    raw_labels = (
        [l.strip().replace("_", " ") for l in args.labels.split(",")]
        if args.labels is not None
        else [Path(c).stem for c in args.configs]
    )
    if len(raw_labels) != len(args.configs):
        raise ValueError(
            f"--labels has {len(raw_labels)} entries but {len(args.configs)} configs were given."
        )

    run_name = args.run_name or Path(args.configs[0]).stem
    cache_dir = None if args.no_cache else args.cache_dir

    # Query equivariance for each config.
    config_results: dict[str, list[pd.DataFrame]] = {}
    for label, config_path in zip(raw_labels, args.configs):
        print(f"\n=== Config: {label} ({config_path}) ===")
        create_config = load_create_config(config_path)
        print(f"  Evaluating at epochs: {epochs}")
        epoch_dfs = query_equivariance(
            create_config, epochs, device_id,
            lead_time_days=args.lead_time_days,
            max_batches=args.max_batches,
            sensitivity=args.sensitivity,
            log_dir=args.log_dir,
            model_name=label,
            check_db=args.check_db,
            autoregressive=args.autoregressive,
            ar_target_hours=args.ar_target_hours,
            cache_dir=cache_dir,
            run_local=args.run_local,
        )
        config_results[label] = epoch_dfs

    # Query untrained reference model when --untrained-config is provided.
    untrained_results = None
    if args.untrained_config is not None:
        print(f"\n=== Untrained reference: {args.untrained_config} (epoch {args.untrained_epoch}) ===")
        untrained_create_config = load_create_config(args.untrained_config)
        untrained_results = query_equivariance(
            untrained_create_config, [args.untrained_epoch], device_id,
            lead_time_days=args.lead_time_days,
            max_batches=args.max_batches,
            sensitivity=args.sensitivity,
            log_dir=args.log_dir,
            model_name="untrained",
            check_db=args.check_db,
            autoregressive=False,
            cache_dir=cache_dir,
            run_local=args.run_local,
        )

    plot_multi_config_by_epoch_group(
        config_results,
        n_groups=3,
        run_name=run_name,
        lead_time_days=args.lead_time_days,
    )
    plot_upper_by_epoch_group(
        config_results,
        n_groups=3,
        run_name=run_name,
        lead_time_days=args.lead_time_days,
    )
    plot_upper_max_vs_level(
        config_results,
        run_name=run_name,
        lead_time_days=args.lead_time_days,
        untrained_results=untrained_results,
    )
    plot_upper_max_vs_level_collapsed(
        config_results,
        run_name=run_name,
        untrained_results=untrained_results,
    )
    plot_upper_max_vs_level_avg(
        config_results,
        run_name=run_name,
        untrained_results=untrained_results,
    )
