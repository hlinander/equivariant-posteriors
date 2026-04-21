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
OVERLEAF_DIR = SCRIPT_DIR / "paper" / "overleaf"
OVERLEAF_FIG_DIR = OVERLEAF_DIR / "figures"

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

    _lineplot_kw = dict(estimator="mean", errorbar=("pi", 100))

    if not df_acc.empty:
        palette = _get_palette(df_acc)
        g = sns.FacetGrid(df_acc, col="metric", sharey=True, col_wrap=3)
        g.map_dataframe(sns.lineplot, x="days", y="acc", style="model", hue="model", palette=palette, **_lineplot_kw)
        g.add_legend()
        for path in [outdir / f"acc_iterated{suffix}.pdf", OVERLEAF_FIG_DIR / f"acc_iterated{suffix}.pdf"]:
            g.savefig(path)
            print(f"Saved {path}")
        plt.close()

    if not df_rmse.empty:
        palette = _get_palette(df_rmse)
        g = sns.FacetGrid(df_rmse, col="metric", sharey=False, col_wrap=3)
        g.map_dataframe(sns.lineplot, x="days", y="rmse", style="model", hue="model", palette=palette, **_lineplot_kw)
        g.add_legend()
        for path in [outdir / f"rmse_iterated{suffix}.pdf", OVERLEAF_FIG_DIR / f"rmse_iterated{suffix}.pdf"]:
            g.savefig(path)
            print(f"Saved {path}")
        plt.close()

    # Single-panel ACC plot for surface v10 (used on first page of paper)
    if not df_acc.empty:
        v10 = df_acc[df_acc["metric"] == "[acc, surface, v10]"]
        if not v10.empty:
            palette = _get_palette(v10)
            fig, ax = plt.subplots(figsize=(4.3, 4.3 * 0.818))
            sns.lineplot(v10, x="days", y="acc", style="model", hue="model", palette=palette, ax=ax, **_lineplot_kw)
            ax.set_title("surface v10")
            fig.tight_layout()
            for path in [outdir / f"surface_v10{suffix}.pdf", OVERLEAF_FIG_DIR / f"surface_v10{suffix}.pdf"]:
                fig.savefig(path)
                print(f"Saved {path}")
            plt.close()


def fmt_val(mean: float, std: float, bold: bool, is_acc: bool, scale: float = 1.0) -> str:
    """Format a value with optional ± std. scale divides the value (e.g. 1e-4)."""
    if pd.isna(mean):
        return ""
    mean = mean / scale
    std = std / scale if not pd.isna(std) else std
    if is_acc:
        txt = f"{mean:.3f}"
        stxt = f"{std:.3f}" if not pd.isna(std) and std > 0 else None
    else:
        txt = f"{mean:.3g}"
        stxt = f"{std:.3g}" if not pd.isna(std) and std > 0 else None
    inner = f"{txt} \\pm {stxt}" if stxt else txt
    return rf"$\mathbf{{{inner}}}$" if bold else rf"${inner}$"


def _detect_scale(values: pd.Series) -> tuple[float, str]:
    """If all non-NaN values share a common exponent < -2, return (scale, latex suffix).
    Otherwise return (1.0, "")."""
    vals = values.dropna()
    if vals.empty:
        return 1.0, ""
    median_exp = int(np.floor(np.log10(vals.abs().median())))
    if median_exp < -2:
        return 10.0 ** median_exp, rf" $\times 10^{{{median_exp}}}$"
    return 1.0, ""


def _make_single_table(
    df: pd.DataFrame, value_col: str, is_acc: bool, models: list[str]
) -> str:
    """Generate a single LaTeX table (ACC or RMSE) with mean ± std across seeds."""
    dfx = df[df["days"].isin([1, 3, 5])].copy()
    parts = dfx["metric"].map(split_metric).apply(pd.Series)
    dfx[["kind", "var", "lev"]] = parts

    idx_cols = ["var", "lev", "days"]
    vmean = dfx.pivot_table(index=idx_cols, columns="model", values=value_col, aggfunc="mean").reindex(columns=models)
    vstd = dfx.pivot_table(index=idx_cols, columns="model", values=value_col, aggfunc="std").reindex(columns=models)

    # Bold: highest ACC or lowest RMSE
    bold_fn = (lambda s: s.eq(s.max())) if is_acc else (lambda s: s.eq(s.min()))

    rows = []
    for (var, lev), _ in vmean.index.to_frame(index=False).groupby(["var", "lev"]):
        days = sorted(set(vmean.loc[var, lev].index))

        # Detect common scale for this variable block (RMSE only)
        if not is_acc:
            block_vals = vmean.loc[var, lev].values.flatten()
            scale, scale_suffix = _detect_scale(pd.Series(block_vals))
        else:
            scale, scale_suffix = 1.0, ""

        unit_str = UNITS.get(lev, "") + scale_suffix
        unit_cell = rf"\multirow{{{len(days)}}}{{*}}{{{unit_str}}}"

        for i, day in enumerate(days):
            idx = (var, lev, day)
            vm = vmean.loc[idx] if idx in vmean.index else pd.Series(np.nan, index=models)
            vs = vstd.loc[idx] if idx in vstd.index else pd.Series(np.nan, index=models)
            bold = bold_fn(vm)

            label = rf"$_\mathrm{{{var}}}^\mathrm{{{lev}}}$" if i == 0 else ""
            unit_col = unit_cell if i == 0 else ""

            cells = [fmt_val(vm[m], vs[m], bold[m], is_acc, scale=scale) for m in models]
            rows.append(f"{label} & {day} & " + " & ".join(cells) + f" & {unit_col} \\\\")
        rows.append(r"\midrule")
    rows.pop()

    n = len(models)
    metric_name = "ACC" if is_acc else "RMSE"
    model_header = " & ".join(models)
    lines = [
        rf"\begin{{tabular}}{{l r {'r ' * n}l}}",
        r"\toprule",
        rf"Variable & $\Delta t$ & \multicolumn{{{n}}}{{c}}{{{metric_name}}} & unit \\",
        rf"\cmidrule(lr){{3-{2 + n}}}",
        rf" & (days) & {model_header} & \\",
        r"\midrule",
        "\n".join(rows),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def make_tables(df_acc: pd.DataFrame, df_rmse: pd.DataFrame) -> tuple[str, str]:
    """Generate separate ACC and RMSE LaTeX tables."""
    models = sorted(set(df_acc["model"].unique()) | set(df_rmse["model"].unique()))
    acc_table = _make_single_table(df_acc, "acc", is_acc=True, models=models)
    rmse_table = _make_single_table(df_rmse, "rmse", is_acc=False, models=models)
    return acc_table, rmse_table


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

        acc_table, rmse_table = make_tables(df_acc, df_rmse)

        for name, table in [("acc", acc_table), ("rmse", rmse_table)]:
            print(f"\n{name.upper()} table:")
            print(table)
            for outdir in [SCRIPT_DIR, OVERLEAF_DIR]:
                outfile = outdir / f"{name}_table_{variant}.tex"
                outfile.write_text(table)
                print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
