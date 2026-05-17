#!/usr/bin/env python
"""Print cached normalization stats across seeds to verify split stability."""

import numpy as np
import glob
import os

CACHE = "/proj/heal_pangu/users/x_tagty/climateset/cache"
SEEDS = [1, 2, 3, 4, 5]

def load_stats(path):
    d = np.load(path, allow_pickle=True).item()
    return d["mean"].squeeze(), d["std"].squeeze()

# --- Inputs ---
inp_dir = f"{CACHE}/inputs_grid_BC_CH4_SO2_CO2_ssp126_ssp370_ssp585_2015-2100_nohistoric_anthro-fires_sphere_stats"
print("=== INPUTS (BC, CH4, SO2, CO2) ===")
channels = ["BC", "CH4", "SO2", "CO2"]
means, stds = [], []
for s in SEEDS:
    m, sd = load_stats(f"{inp_dir}/train_stats_seed{s}_valfrac0.1.npy")
    means.append(m); stds.append(sd)
means, stds = np.stack(means), np.stack(stds)
print(f"{'ch':>5}  " + "  ".join(f"seed{s:>2}_mean  seed{s:>2}_std" for s in SEEDS) + "  cross_std_mean  cross_std_std")
for i, ch in enumerate(channels):
    vals = "  ".join(f"{means[j,i]:12.4e}  {stds[j,i]:12.4e}" for j in range(len(SEEDS)))
    print(f"{ch:>5}  {vals}  {means[:,i].std():14.4e}  {stds[:,i].std():13.4e}")

# --- Outputs per climate model ---
print("\n=== OUTPUTS (tas, pr) per climate model ===")
out_dirs = sorted(glob.glob(f"{CACHE}/outputs_*_grid_tas_pr_*_sphere_stats"))
for out_dir in out_dirs:
    model = os.path.basename(out_dir).split("_grid_")[0].replace("outputs_", "")
    avail = [s for s in SEEDS if os.path.exists(f"{out_dir}/train_stats_seed{s}_valfrac0.1.npy")]
    if not avail:
        continue
    means, stds = [], []
    for s in avail:
        m, sd = load_stats(f"{out_dir}/train_stats_seed{s}_valfrac0.1.npy")
        means.append(m); stds.append(sd)
    means, stds = np.stack(means), np.stack(stds)
    print(f"\n  {model}  (seeds {avail})")
    for i, var in enumerate(["tas", "pr"]):
        m_vals = "  ".join(f"{means[j,i]:.6f}" for j in range(len(avail)))
        s_vals = "  ".join(f"{stds[j,i]:.6f}" for j in range(len(avail)))
        print(f"    {var} mean: {m_vals}   std_across_seeds={means[:,i].std():.2e}")
        print(f"    {var} std:  {s_vals}   std_across_seeds={stds[:,i].std():.2e}")
