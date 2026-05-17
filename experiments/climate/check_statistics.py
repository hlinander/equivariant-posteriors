import numpy as np
import sys

files = [
    "/proj/heal_pangu/users/x_tagty/climateset/cache/outputs_AWI-CM-1-1-MR_r1i1p1f1_nside32_tas_pr_ssp126_ssp370_ssp585_2015-2100_nohistoric_250_km/train_stats_seed1_valfrac0.1.npy",
    "/proj/heal_pangu/users/x_tagty/climateset/cache/inputs_nside32_BC_CH4_SO2_CO2_ssp126_ssp370_ssp585_2015-2100_nohistoric_anthro-fires/train_stats_seed1_valfrac0.1.npy",
    "/proj/heal_pangu/users/x_tagty/climateset/cache/inputs_grid_BC_CH4_SO2_CO2_ssp126_ssp370_ssp585_2015-2100_nohistoric_no-fires/train_stats_seed1_valfrac0.1.npy",
    *sys.argv[1:],
]

for path in files:
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print('='*60)
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        item = data.item()
        if isinstance(item, dict):
            for k, v in item.items():
                print(f"  {k}: {v}")
        else:
            print(item)
    else:
        print(f"  shape: {data.shape}, dtype: {data.dtype}")
        print(f"  {data}")
