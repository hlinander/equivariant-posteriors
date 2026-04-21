"""
Generate ACC & RMSE table and plots for weather models.
Connects to ducklake, caches data locally, produces LaTeX table with mean ± std.

Usage:
    python experiments/weather_pear/acc_and_rmse_table.py                # use cached data
    python experiments/weather_pear/acc_and_rmse_table.py --refresh      # re-fetch from ducklake
"""

import argparse
import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
LOCAL_DB = SCRIPT_DIR / "weather_models.db"
SECRET_FILE = REPO_ROOT / "secret.txt"

# Models to select from the remote database
MODEL_NAMES = [
    "SwinHPPanguPad",
    "Pangu",
    "PanguParametrized",
    "PanguPhysicsNemo",
    "GraphCastPhysicsNemo",
    "FengwuPhysicsNemo",
]

# Display name mapping
DISPLAY_NAMES = {
    "SwinHPPanguPad": "PEAR",
    "SwinHPPanguPadPatch": "PEAR",  # derived name (patch_size=16)
    "SwinHPPanguPadDH": "PEAR-DH",
    "Pangu": "Pangu",
    "PanguParametrized": "PanguParam",
    "PanguPhysicsNemo": "PanguPN",
    "PanguPhysicsNemoDH": "PanguPN-DH",
    "GraphCastPhysicsNemo": "GraphCast",
    "GraphCastPhysicsNemoDH": "GraphCast-DH",
    "FengwuPhysicsNemo": "FengWu",
    "FengwuPhysicsNemoDH": "FengWu-DH",
}

# Fixed color palette — consistent across all plots
MODEL_PALETTE = {
    "PEAR": "#1f77b4",       # blue (same as original paper)
    "Pangu": "#ff7f0e",      # orange (same as original paper)
    "PanguParam": "#2ca02c",
    "PanguPN": "#d62728",
    "GraphCast": "#9467bd",
    "FengWu": "#8c564b",
}

UNITS = {
    "msl": r"$\mathrm{Pa}$",
    "t2m": r"$\mathrm{K}$",
    "u10": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "v10": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "q": r"$\frac{\mathrm{g}}{\mathrm{kg}}$",
    "t": r"$\mathrm{K}$",
    "u": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "v": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "z": r"$\mathrm{gpm}$",
}


def fetch_and_cache(conn: duckdb.DuckDBPyConnection):
    """Fetch data from ducklake and store locally."""
    print("Reading secret.txt and attaching ducklake...")
    with open(SECRET_FILE) as f:
        conn.sql(f.read())

    model_list = ", ".join(f"'{m}'" for m in MODEL_NAMES)

    print("Fetching model IDs (filtered to x_hamli)...")
    conn.sql(f"""
        CREATE OR REPLACE TABLE model_ids AS
        SELECT DISTINCT t1.model_id
        FROM eqp.model_parameter_text AS t1
        JOIN eqp.model_parameter_text AS t2 ON t1.model_id = t2.model_id
        WHERE t1.name = 'train_config.model.name'
          AND t1.value IN ({model_list})
          AND t2.name = 'env.LOGNAME'
          AND t2.value = 'x_hamli'
    """)

    print("Fetching metrics...")
    conn.sql("""
        CREATE OR REPLACE TABLE local_metrics AS
        SELECT t1.*
        FROM eqp.checkpoint_sample_metric_float AS t1
        SEMI JOIN model_ids ON t1.model_id = model_ids.model_id
        WHERE t1.name LIKE 'acc_%' OR t1.name LIKE 'rmse_%'
           OR t1.name LIKE 'dh$acc_%' OR t1.name LIKE 'dh$rmse_%'
    """)

    print("Fetching parameters...")
    conn.sql("""
        CREATE OR REPLACE TABLE local_params AS
        SELECT t1.*
        FROM eqp.model_parameter_text AS t1
        SEMI JOIN model_ids ON t1.model_id = model_ids.model_id
    """)

    # Pivot parameters to one row per model_id
    conn.sql("""
        CREATE OR REPLACE TABLE model_info AS
        SELECT
            model_id,
            max(CASE WHEN name = 'train_config.model.name' THEN value END) AS model_name,
            max(CASE WHEN name = 'train_config.model.config.rel_pos_bias' THEN value END) AS rel_pos_bias,
            max(CASE WHEN name = 'train_config.model.config.patch_size' THEN value END) AS patch_size
        FROM local_params
        GROUP BY model_id
    """)

    n_metrics = conn.sql("SELECT count(*) FROM local_metrics").fetchone()[0]
    n_models = conn.sql("SELECT count(*) FROM model_info").fetchone()[0]
    print(f"Cached {n_metrics} metric rows, {n_models} models")


def build_dataframes(conn: duckdb.DuckDBPyConnection):
    """Build ACC and RMSE dataframes from local tables.

    Returns dict mapping variant name ('hp', 'dh') to (df_acc, df_rmse) tuples.
    HP = HealPix metrics (no dh$ prefix), DH = double-headed metrics (dh$ prefix).

    For each model_id (seed), we select the step with the lowest overall RMSE
    (average across all rmse_ metrics at that step), then use all metrics from
    that step.
    """
    # Find the best step per model_id: lowest average RMSE across all rmse metrics
    conn.sql("""
        CREATE OR REPLACE TABLE best_steps AS
        WITH step_rmse AS (
            SELECT
                t1.model_id,
                t1.step,
                avg(t1.mean) AS avg_rmse
            FROM local_metrics AS t1
            WHERE t1.name LIKE 'rmse_%' AND t1.name NOT LIKE 'dh$%'
            GROUP BY t1.model_id, t1.step
        )
        SELECT model_id, step
        FROM step_rmse
        QUALIFY row_number() OVER (PARTITION BY model_id ORDER BY avg_rmse) = 1
    """)

    print("\nBest steps per model:")
    print(conn.sql("""
        SELECT m.model_name, b.model_id, b.step, b.step / 365 / 10 AS epoch
        FROM best_steps b JOIN model_info m ON b.model_id = m.model_id
        ORDER BY m.model_name, b.model_id
    """))

    conn.sql("""
        CREATE OR REPLACE TABLE acc AS
        SELECT
            t1.name LIKE 'dh$%' AS dh,
            split_part(t1.name, '$', -1) AS metric,
            m.model_name AS model,
            mean AS acc,
            t1.step / 365 / 10 AS epoch,
            m.rel_pos_bias
        FROM local_metrics AS t1
        JOIN model_info AS m ON t1.model_id = m.model_id
        JOIN best_steps AS b ON t1.model_id = b.model_id AND t1.step = b.step
        WHERE split_part(t1.name, '$', -1) LIKE 'acc_%'
    """)

    conn.sql("""
        CREATE OR REPLACE TABLE rmse AS
        SELECT
            t1.name LIKE 'dh$%' AS dh,
            split_part(t1.name, '$', -1) AS metric,
            m.model_name AS model,
            mean AS rmse,
            t1.step / 365 / 10 AS epoch,
            m.rel_pos_bias
        FROM local_metrics AS t1
        JOIN model_info AS m ON t1.model_id = m.model_id
        JOIN best_steps AS b ON t1.model_id = b.model_id AND t1.step = b.step
        WHERE split_part(t1.name, '$', -1) LIKE 'rmse_%'
    """)

    # Show available models
    print("\nACC models:")
    print(conn.sql("""
        SELECT model, dh, min(epoch) as min_epoch, max(epoch) as max_epoch, count(*) as n
        FROM acc GROUP BY model, dh ORDER BY model, dh
    """))

    # Parse metric structure
    for table in ["acc", "rmse"]:
        conn.sql(f"""
            CREATE OR REPLACE TABLE data_{table} AS
            SELECT
                try_cast(split_part(split_part(metric, '.', 1), '_', -1) AS integer) AS pressure,
                try_cast(trim(split_part(metric, '.', 2), 'd') AS integer) AS days,
                string_split(split_part(metric, '.', 1), '_')[1:3] AS metric,
                *
            FROM {table}
        """)

    results = {}
    for variant, dh_val in [("hp", False), ("dh", True)]:
        df_acc = conn.sql(f"""
            SELECT days, acc, model, metric::varchar AS metric
            FROM data_acc WHERE dh = {dh_val}
            ORDER BY metric, model, days
        """).df()

        df_rmse = conn.sql(f"""
            SELECT days, rmse, model, metric::varchar AS metric
            FROM data_rmse WHERE dh = {dh_val}
            ORDER BY metric, model, days
        """).df()

        # Apply display names
        df_acc["model"] = df_acc["model"].map(lambda m: DISPLAY_NAMES.get(m, m))
        df_rmse["model"] = df_rmse["model"].map(lambda m: DISPLAY_NAMES.get(m, m))

        results[variant] = (df_acc, df_rmse)

    return results


def split_metric(s: str) -> dict:
    k, v, l = [p.strip() for p in s.strip("[]").split(",")]
    return {"kind": k.lower(), "var": v, "lev": l}


def _get_palette(df: pd.DataFrame) -> dict:
    """Get color palette for models present in the dataframe."""
    models = df["model"].unique()
    return {m: MODEL_PALETTE.get(m, "#333333") for m in models}


def make_plots(df_acc: pd.DataFrame, df_rmse: pd.DataFrame, outdir: Path, suffix: str = ""):
    """Generate faceted ACC and RMSE plots with CI bands."""
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")

    if not df_acc.empty:
        palette = _get_palette(df_acc)
        g = sns.FacetGrid(df_acc, col="metric", sharey=True, col_wrap=3)
        g.map_dataframe(sns.lineplot, x="days", y="acc", style="model", hue="model", palette=palette)
        g.add_legend()
        path = outdir / f"acc_iterated{suffix}.pdf"
        g.savefig(path)
        print(f"Saved {path}")
        plt.close()

    if not df_rmse.empty:
        palette = _get_palette(df_rmse)
        g = sns.FacetGrid(df_rmse, col="metric", sharey=False, col_wrap=3)
        g.map_dataframe(sns.lineplot, x="days", y="rmse", style="model", hue="model", palette=palette)
        g.add_legend()
        path = outdir / f"rmse_iterated{suffix}.pdf"
        g.savefig(path)
        print(f"Saved {path}")
        plt.close()


def fmt_val(mean: float, std: float, bold: bool, is_acc: bool) -> str:
    if pd.isna(mean):
        return ""
    if is_acc:
        txt = f"{mean:.3f}"
        stxt = f"{std:.3f}" if not pd.isna(std) and std > 0 else None
    else:
        e = int(np.floor(np.log10(abs(mean)))) if mean else 0
        if e < -2:
            txt = f"{mean / 10**e:.3g}\\times 10^{{{e}}}"
            stxt = (
                f"{std / 10**e:.2g}\\times 10^{{{e}}}"
                if not pd.isna(std) and std > 0
                else None
            )
        else:
            txt = f"{mean:.3g}"
            stxt = f"{std:.3g}" if not pd.isna(std) and std > 0 else None
    inner = f"{txt} \\pm {stxt}" if stxt else txt
    return rf"$\mathbf{{{inner}}}$" if bold else rf"${inner}$"


def make_table(df_acc: pd.DataFrame, df_rmse: pd.DataFrame) -> str:
    """Generate LaTeX table with mean ± std across seeds."""
    # Filter to days 1, 3, 5
    dfacc = df_acc[df_acc["days"].isin([1, 3, 5])].copy()
    dfrmse = df_rmse[df_rmse["days"].isin([1, 3, 5])].copy()

    for df in (dfacc, dfrmse):
        parts = df["metric"].map(split_metric).apply(pd.Series)
        df[["kind", "var", "lev"]] = parts

    models = sorted(dfacc["model"].unique().tolist())
    main = models

    idx_cols = ["var", "lev", "days"]
    acc_mean = dfacc.pivot_table(index=idx_cols, columns="model", values="acc", aggfunc="mean").reindex(columns=models)
    acc_std = dfacc.pivot_table(index=idx_cols, columns="model", values="acc", aggfunc="std").reindex(columns=models)
    rmse_mean = dfrmse.pivot_table(index=idx_cols, columns="model", values="rmse", aggfunc="mean").reindex(columns=models)
    rmse_std = dfrmse.pivot_table(index=idx_cols, columns="model", values="rmse", aggfunc="std").reindex(columns=models)

    rows = []
    union_idx = acc_mean.index.union(rmse_mean.index)
    for (var, lev), grp in union_idx.to_frame(index=False).groupby(["var", "lev"]):
        try:
            days = sorted(set(acc_mean.loc[var, lev].index))
        except KeyError:
            days = sorted(set(rmse_mean.loc[var, lev].index))
        unit_cell = rf"\multirow{{{len(days)}}}{{*}}{{{UNITS.get(lev, '')}}}"

        for i, day in enumerate(days):
            idx = (var, lev, day)
            am = acc_mean.loc[idx] if idx in acc_mean.index else pd.Series(np.nan, index=models)
            astd = acc_std.loc[idx] if idx in acc_std.index else pd.Series(np.nan, index=models)
            rm = rmse_mean.loc[idx] if idx in rmse_mean.index else pd.Series(np.nan, index=models)
            rstd = rmse_std.loc[idx] if idx in rmse_std.index else pd.Series(np.nan, index=models)

            bold_a = am.eq(am[main].max())
            bold_r = rm.eq(rm[main].min())

            label = rf"$_\mathrm{{{var}}}^\mathrm{{{lev}}}$" if i == 0 else ""
            unit_col = unit_cell if i == 0 else ""

            cells = [fmt_val(am[m], astd[m], bold_a[m], True) for m in models] + [
                fmt_val(rm[m], rstd[m], bold_r[m], False) for m in models
            ]
            rows.append(f"{label} & {day} & " + " & ".join(cells) + f" & {unit_col} \\\\")
        rows.append(r"\midrule")
    rows.pop()  # remove last midrule

    n = len(models)
    model_header = " & ".join(models)
    lines = [
        rf"\begin{{tabular}}{{l r {'r ' * n}|{'r ' * n}l}}",
        r"\toprule",
        rf"Variable & $\Delta t$ & \multicolumn{{{n}}}{{c}}{{ACC}} & \multicolumn{{{n}}}{{c}}{{RMSE}} & unit \\",
        rf"\cmidrule(lr){{3-{2 + n}}}\cmidrule(lr){{{3 + n}-{2 + 2 * n}}}",
        rf" & (days) & {model_header} & {model_header} & \\",
        r"\midrule",
        "\n".join(rows),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate ACC & RMSE table for weather models")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch data from ducklake")
    args = parser.parse_args()

    if args.refresh or not LOCAL_DB.exists():
        conn = duckdb.connect(str(LOCAL_DB))
        fetch_and_cache(conn)
    else:
        conn = duckdb.connect(str(LOCAL_DB))
        print(f"Using cached data from {LOCAL_DB}")

    variants = build_dataframes(conn)
    conn.close()

    for variant, (df_acc, df_rmse) in variants.items():
        label = "HealPix" if variant == "hp" else "Double-headed"
        print(f"\n{'='*60}")
        print(f"{label} ({variant}): ACC={len(df_acc)} rows, RMSE={len(df_rmse)} rows")
        models = sorted(df_acc['model'].unique())
        print(f"Models: {models}")

        if df_acc.empty and df_rmse.empty:
            print(f"No data for {variant}, skipping")
            continue

        make_plots(df_acc, df_rmse, SCRIPT_DIR, suffix=f"_{variant}")

        table = make_table(df_acc, df_rmse)
        print("\n" + table)

        outfile = SCRIPT_DIR / f"acc_rmse_table_{variant}.tex"
        outfile.write_text(table)
        print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
