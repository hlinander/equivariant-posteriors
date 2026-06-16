"""Determinant of real N x N matrices as a regression task.

Inputs are Gaussian-iid matrices A ~ N(0, sigma)^{NxN}; the target is the pair
(sign(det A), standardized log|det A|), computed exactly via slogdet. log|det|
is standardized to zero mean / unit variance using a fixed reference draw that
depends only on (n, sigma) — so the target scale is identical across seeds and
train sizes, and R^2 on log|det| equals 1 - MSE on the standardized target.

Each (n, n_train, seed) draws its own deterministic Gaussian train set (draws
are independent across n_train, not nested — torch's normal kernel is not
prefix-stable across sizes; 3 seeds handle the curve variance). The validation
set is a separate fixed stream, identical across all n_train at a given
(n, sigma, seed).
"""
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy

# Disjoint, fixed seed offsets so train / val / standardization streams never
# coincide for any user seed.
_VAL_SEED_OFFSET = 1 << 40
_REF_SEED = 999983
_REF_N = 16384


@dataclass(frozen=True)
class DataRealDetConfig:
    n: int = 3
    n_train: int = 1000  # learning-curve variable (train-set size)
    n_val: int = 10000  # fixed held-out set size
    sigma: float = 1.0  # entry scale: A_ij ~ N(0, sigma^2)
    seed: int = 0  # controls the Gaussian draws
    validation: bool = False  # True: return the held-out set

    @property
    def n_samples(self):
        return self.n_val if self.validation else self.n_train

    def serialize_human(self):
        return {**self.__dict__, "n_samples": self.n_samples}


def _sign_logabs(matrices):
    """Exact (sign>0 as {0,1} float, log|det|) for a batch of (B, n, n)."""
    sign, logabs = torch.linalg.slogdet(matrices)
    return (sign > 0).float(), logabs


class DataRealDet(torch.utils.data.Dataset):
    def __init__(self, data_config: DataRealDetConfig):
        n = data_config.n
        sigma = data_config.sigma

        # Standardization constants for log|det|, deterministic in (n, sigma):
        # a fixed reference draw independent of the train/val seed so every run
        # at this (n, sigma) shares one target scale.
        g_ref = torch.Generator().manual_seed(_REF_SEED + n)
        ref = torch.randn(_REF_N, n, n, generator=g_ref) * sigma
        _, ref_logabs = _sign_logabs(ref)
        self.ld_mu = ref_logabs.mean()
        self.ld_sd = ref_logabs.std().clamp_min(1e-6)

        if data_config.validation:
            g = torch.Generator().manual_seed(data_config.seed + _VAL_SEED_OFFSET)
            n_draw = data_config.n_val
        else:
            g = torch.Generator().manual_seed(data_config.seed)
            n_draw = data_config.n_train

        matrices = torch.randn(n_draw, n, n, generator=g) * sigma
        sign_label, logabs = _sign_logabs(matrices)
        logabs_std = (logabs - self.ld_mu) / self.ld_sd

        self.xs = matrices.reshape(n_draw, n * n)
        # target column 0 = sign label (0/1), column 1 = standardized log|det|
        self.ys = torch.stack([sign_label, logabs_std], dim=1)
        self.sample_ids = torch.arange(n_draw, dtype=torch.int32)

    @staticmethod
    def data_spec(config):
        return DataSpec(
            input_shape=torch.Size([config.n * config.n]),
            target_shape=torch.Size([2]),
            output_shape=torch.Size([2]),
        )

    def __getitem__(self, idx):
        return create_sample_legacy(self.xs[idx], self.ys[idx], self.sample_ids[idx])

    def __getitems__(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.xs.device)
        return dict(
            input=self.xs[idx],
            target=self.ys[idx],
            sample_id=self.sample_ids[idx],
        )

    def to(self, device):
        self.xs = self.xs.to(device)
        self.ys = self.ys.to(device)
        self.sample_ids = self.sample_ids.to(device)
        return self

    @staticmethod
    def collate_fn(batch):
        return batch

    def __len__(self):
        return self.xs.shape[0]
