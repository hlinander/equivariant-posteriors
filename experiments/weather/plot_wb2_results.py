import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from pathlib import Path
import json

parent = Path(sys.argv[1]).parent

ds = xr.open_dataset(sys.argv[1])

errors = dict()
for var_name, da in ds.data_vars.items():
    fig = plt.figure(figsize=(10, 10))
    da.sel(metric="spatial_mse").plot()
    plt.savefig(parent / f"{var_name}.png")

    fig = plt.figure(figsize=(10, 10))
    da.sel(metric="spatial_mse").plot.hist(yscale="log")
    plt.savefig(parent / f"{var_name}_hist.png")
    minv = da.sel(metric="spatial_mse").min()
    maxv = da.sel(metric="spatial_mse").max()
    mid = (minv + maxv) * 0.5

    v = da.sel(metric="spatial_mse")
    fig = plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots()
    # v.plot(ax=ax)
    np.log(v).plot(ax=ax)
    # v.where(v < v.quantile(0.99), drop=True).plot(ax=ax)
    # errors[var_name] = np.sqrt(v.where(v < v.quantile(0.9), drop=True)).mean().item()
    errors[var_name] = (
        v.mean().item()
    )  # np.sqrt(v.where(v < v.quantile(0.9), drop=True)).mean().item()

    plt.title(f"{var_name}")
    ax.set_title(f"{var_name}")
    plt.savefig(parent / f"{var_name}_filtered.png")


open(parent / f"rmse.json", "w").write(json.dumps(errors, indent=2))
