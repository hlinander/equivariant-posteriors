import os
from multiprocessing import Pool

from experiments.weather.data import DataHPConfig, DataHP


def hydrate_idx(idx_and_nside):
    idx, nside = idx_and_nside
    ds_hp = DataHP(DataHPConfig(nside=nside))
    ds_dh = DataHP(DataHPConfig(nside=nside, driscoll_healy=True))
    _ = ds_hp[idx]
    _ = ds_dh[idx]


ds_meta = DataHP(DataHPConfig())
n_samples_left = len(ds_meta)
batch_size = int(os.getenv("SLURM_CPUS_ON_NODE", f"{os.cpu_count()}"))
print(f"Starting hydration with batch size {batch_size}")
idx = 0
nside = 256
with Pool() as p:
    while n_samples_left > 0:
        n_requests = min(batch_size, n_samples_left)
        idx_and_nsides = zip(range(idx, idx + n_requests), [nside] * n_requests)
        p.map(hydrate_idx, idx_and_nsides)
        idx += n_requests
        n_samples_left -= n_requests
        print(n_samples_left)

# for idx in range(len(ds_hp)):
#     _ = ds_hp[idx]
#     _ = ds_dh[idx]
#     print(idx)
